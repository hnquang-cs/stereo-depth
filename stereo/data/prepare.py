"""Prepare the labelled training data once, from the official sources.

Downloading inside a GPU session burns GPU quota for as long as it takes, so
this runs once -- in a Kaggle notebook with no accelerator, or on any machine
-- and the result is attached as a dataset from then on::

    python -m stereo.data.prepare --out /kaggle/working/stereo-data

What it writes under ``--out``, and why:

    middlebury/MiddEval3/trainingQ  15 scenes at quarter size, with both views'
                                    disparity (the GT0 and GT1 archives)
    middlebury/2001 2003 2005 2006  the releases' own files, unchanged: their
                                    disparity encodings depend on the image size.
                                    2003, 2005 and 2006 in two sizes, so that a
                                    larger training width still has a wide copy
    middlebury/2014 2021            2014's 13 scenes that MiddEval3 does not hold,
                                    and all 24 of 2021, shrunk from 3000 and
                                    1920 px to at most ``--max-width``
    kitti2015/training              the 200 + 194 frame pairs that have ground
    kitti2012/training              truth (frame _10); the archives' other frames
                                    are unlabelled or the test set
    instereo2k/train, test          2,000 + 50 indoor pairs, shrunk from 1080 px,
                                    when a download of it is given (--instereo2k):
                                    its hosts, OneDrive and Baidu, cannot be scripted
    stereo_data_manifest.json       marks the directory; the notebook finds it

Every release is then checked by warping (:func:`stereo.data.check_label_scale`),
flipped as well as not, and the run fails rather than leave labels it could not
confirm.

KITTI is licensed CC BY-NC-SA 3.0, InStereo2K is for non-commercial use, and
Middlebury asks to be cited; a dataset made from this output is for private use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import warnings
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io import read_pfm, write_pfm

MIDDLEBURY = "https://vision.middlebury.edu/stereo/data"
MIDDEVAL3 = "https://vision.middlebury.edu/stereo/submit3/zip"
KITTI = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

MANIFEST = "stereo_data_manifest.json"

SCENES_2001 = ("sawtooth", "venus", "bull", "poster", "barn1", "barn2")
#: 2005 withholds the ground truth of Computer, Drumsticks and Dwarves.
SCENES_2005 = ("Art", "Books", "Dolls", "Laundry", "Moebius", "Reindeer")
SCENES_2006 = ("Aloe", "Baby1", "Baby2", "Baby3", "Bowling1", "Bowling2", "Cloth1", "Cloth2",
               "Cloth3", "Cloth4", "Flowerpots", "Lampshade1", "Lampshade2", "Midd1", "Midd2",
               "Monopoly", "Plastic", "Rocks1", "Rocks2", "Wood1", "Wood2")
#: 2014's scenes with public ground truth that MiddEval3's training set does not hold.
SCENES_2014 = ("Backpack", "Bicycle1", "Cable", "Classroom1", "Couch", "Flowers", "Mask",
               "Shopvac", "Sticks", "Storage", "Sword1", "Sword2", "Umbrella")
SCENES_2021 = ("artroom1", "artroom2", "bandsaw1", "bandsaw2", "chess1", "chess2", "chess3",
               "curule1", "curule2", "curule3", "ladder1", "ladder2", "octogons1", "octogons2",
               "pendulum1", "pendulum2", "podium1", "skates1", "skates2", "skiboots1",
               "skiboots2", "skiboots3", "traproom1", "traproom2")
#: Two sizes of the single-illumination archives, which hold the release's
#: default exposure. Third is 413-465 px wide, half 620-698.
SIZES_2005_2006 = ("ThirdSize", "HalfSize")
#: What a 2014 or 2021 scene needs, out of everything its directory holds.
FULL_SIZE_FILES = ("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "calib.txt")

#: Per benchmark: (archive, folders to take from it). Frames other than _10 have
#: no ground truth.
KITTI_ARCHIVES = {
    "kitti2015": (("data_scene_flow.zip", ("training/image_2/", "training/image_3/",
                                           "training/disp_occ_0/", "training/disp_noc_0/")),
                  ("data_scene_flow_calib.zip", ("training/calib_cam_to_cam/",))),
    "kitti2012": (("data_stereo_flow.zip", ("training/colored_0/", "training/colored_1/",
                                            "training/disp_occ/", "training/disp_noc/")),
                  ("data_stereo_flow_calib.zip", ("training/calib/",))),
}
KITTI_PAIRS = {"kitti2015": 200, "kitti2012": 194}


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #

#: Files at least this large are fetched as parallel byte ranges.
SEGMENTED_MIN_BYTES = 64 * 1024 * 1024
SEGMENTS = 8


def fetch(url: str, path: str) -> str:
    """Download ``url`` to ``path`` unless it is already there; resumable."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    size = ranged_size(url)
    if size is not None and size >= SEGMENTED_MIN_BYTES and shutil.which("curl"):
        return fetch_segmented(url, path, size)
    partial = path + ".part"
    if shutil.which("curl"):
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-C", "-", "-o", partial, url],
                       check=True)
    else:
        urllib.request.urlretrieve(url, partial)
    os.replace(partial, path)
    return path


def ranged_size(url: str) -> Optional[int]:
    """The file's size, if the server serves byte ranges of it; else ``None``."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as reply:
            if reply.headers.get("Accept-Ranges", "").lower() != "bytes":
                return None
            return int(reply.headers.get("Content-Length") or 0) or None
    except (OSError, ValueError):
        return None


def fetch_segmented(url: str, path: str, size: int, segments: int = SEGMENTS) -> str:
    """Download ``segments`` byte ranges in parallel, then join them.

    Measured from S3, KITTI's host: 1-2.5 MB/s per connection, so one
    connection per archive would take half an hour for KITTI alone. A range
    already complete from an interrupted run is kept.
    """
    bounds = [(i * size // segments, (i + 1) * size // segments - 1) for i in range(segments)]
    parts = [f"{path}.part{i}" for i in range(segments)]

    def get(index: int) -> None:
        (start, end), part = bounds[index], parts[index]
        if os.path.exists(part) and os.path.getsize(part) == end - start + 1:
            return
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-r", f"{start}-{end}",
                        "-o", part, url], check=True)
        if os.path.getsize(part) != end - start + 1:
            raise RuntimeError(f"{url}: range {start}-{end} arrived incomplete")

    with ThreadPoolExecutor(max_workers=segments) as pool:
        list(pool.map(get, range(segments)))
    with open(path + ".part", "wb") as joined:
        for part in parts:
            with open(part, "rb") as handle:
                shutil.copyfileobj(handle, joined, 16 * 1024 * 1024)
    if os.path.getsize(path + ".part") != size:
        raise RuntimeError(f"{url}: joined {os.path.getsize(path + '.part')} of {size} bytes")
    os.replace(path + ".part", path)
    for part in parts:
        os.remove(part)
    return path


def fetch_all(jobs: Sequence[Tuple[str, str]], workers: int = 6, label: str = "") -> None:
    """Download ``(url, path)`` pairs several at a time.

    Middlebury's server sends about 3 MB/s per connection, so one at a time
    would take several times longer than the bandwidth allows.
    """
    todo = [(url, path) for url, path in jobs if not os.path.exists(path)]
    if not todo:
        print(f"  {label}: {len(jobs)} files already downloaded")
        return
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(lambda job: fetch(*job), todo), start=1):
            if done % 25 == 0 or done == len(todo):
                print(f"  {label}: {done}/{len(todo)} files, {time.time() - started:.0f}s")
    size = sum(os.path.getsize(path) for _, path in todo)
    print(f"  {label}: {size / 1e6:.0f} MB in {time.time() - started:.0f}s")


def extract(archive: str, target: str, keep: Optional[Callable[[str], bool]] = None) -> int:
    """Extract the members ``keep`` accepts (all by default). Returns how many."""
    count = 0
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir() or (keep is not None and not keep(member.filename)):
                continue
            handle.extract(member, target)
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Shrinking the full-size releases
# --------------------------------------------------------------------------- #

def shrink_disparity(disparity: np.ndarray, factor: int) -> np.ndarray:
    """Disparity at ``1/factor`` size.

    Each ``factor x factor`` block becomes the median of its known pixels,
    divided by ``factor`` because disparity is a length. A median keeps one
    surface at a depth edge where a mean would invent one in between; a block
    less than half known becomes unknown (``inf``).
    """
    if factor == 1:
        return disparity.astype(np.float32)
    rows, cols = disparity.shape[0] // factor, disparity.shape[1] // factor
    blocks = (disparity[:rows * factor, :cols * factor]
              .reshape(rows, factor, cols, factor).transpose(0, 2, 1, 3)
              .reshape(rows, cols, factor * factor))
    known = np.isfinite(blocks) & (blocks > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)    # all-unknown blocks
        median = np.nanmedian(np.where(known, blocks, np.nan), axis=2)
    enough = known.sum(axis=2) * 2 >= factor * factor
    return np.where(enough, median / factor, np.inf).astype(np.float32)


def shrink_calib(text: str, factor: int, width: int, height: int) -> str:
    """A Middlebury ``calib.txt`` for the image shrunk by ``factor``.

    Pixel centres sit at integer coordinates, so a principal point maps as
    ``(c + 0.5) / factor - 0.5``; the focal length and every disparity-valued
    entry simply divide.
    """
    def number(value: float) -> str:
        return f"{value:.10g}"

    lines = []
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator:
            lines.append(line)
            continue
        if key in ("cam0", "cam1"):
            m = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)]
            if len(m) == 9:                 # [f 0 cx; 0 f cy; 0 0 1]
                m[0], m[4] = m[0] / factor, m[4] / factor
                m[2], m[5] = (m[2] + 0.5) / factor - 0.5, (m[5] + 0.5) / factor - 0.5
                value = "[{} {} {}; {} {} {}; {} {} {}]".format(*map(number, m))
        elif key in ("doffs", "vmin", "vmax", "dyavg", "dymax"):
            value = number(float(value) / factor)
        elif key == "ndisp":
            value = str(math.ceil(float(value) / factor))
        elif key == "width":
            value = str(width)
        elif key == "height":
            value = str(height)
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def shrink_scene(source: str, target: str, max_width: int) -> int:
    """Copy a PFM-labelled scene (2014, 2021) at ``1/k`` of its size.

    ``k`` is the smallest integer that brings the width to ``max_width`` or
    below. The images are block-averaged (INTER_AREA) over the same blocks the
    disparity is reduced over, so the two stay aligned. Returns ``k``.
    """
    left = cv2.imread(os.path.join(source, "im0.png"), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(f"no im0.png in {source}")
    height, width = left.shape[:2]
    factor = max(1, math.ceil(width / max_width))
    rows, cols = height // factor, width // factor
    os.makedirs(target, exist_ok=True)

    for name in ("im0.png", "im1.png"):
        image = left if name == "im0.png" else cv2.imread(os.path.join(source, name), cv2.IMREAD_COLOR)
        image = image[:rows * factor, :cols * factor]
        cv2.imwrite(os.path.join(target, name),
                    cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA))
    for name in ("disp0.pfm", "disp1.pfm"):
        path = os.path.join(source, name)
        if os.path.exists(path):
            write_pfm(os.path.join(target, name), shrink_disparity(read_pfm(path), factor))
    calib = os.path.join(source, "calib.txt")
    if os.path.exists(calib):
        with open(calib) as handle:
            text = shrink_calib(handle.read(), factor, cols, rows)
        with open(os.path.join(target, "calib.txt"), "w") as handle:
            handle.write(text)
    return factor


# --------------------------------------------------------------------------- #
# Middlebury
# --------------------------------------------------------------------------- #

def middlebury_downloads(cache: str) -> List[Tuple[str, str]]:
    """Every Middlebury file and archive to fetch, as ``(url, local path)``."""
    jobs = [(f"{MIDDEVAL3}/MiddEval3-{part}-Q.zip", os.path.join(cache, f"MiddEval3-{part}-Q.zip"))
            for part in ("data", "GT0", "GT1")]
    jobs += [(f"{MIDDLEBURY}/scenes2001/data/{scene}/{name}", os.path.join(cache, "2001", scene, name))
             for scene in SCENES_2001 for name in ("im2.ppm", "im6.ppm", "disp2.pgm", "disp6.pgm")]
    for scene in ("cones", "teddy"):
        jobs.append((f"{MIDDLEBURY}/scenes2003/newdata/{scene}/{scene}-png-2.zip",
                     os.path.join(cache, "2003", f"{scene}-png-2.zip")))
        jobs.append((f"{MIDDLEBURY}/scenes2003/newdata/full/{scene}H-ppm-2.zip",
                     os.path.join(cache, "2003", f"{scene}H-ppm-2.zip")))
    for year, scenes in (("2005", SCENES_2005), ("2006", SCENES_2006)):
        for size in SIZES_2005_2006:
            jobs += [(f"{MIDDLEBURY}/scenes{year}/{size}/zip-2views/{scene}-2views.zip",
                      os.path.join(cache, year, size, f"{scene}-2views.zip")) for scene in scenes]
    jobs += [(f"{MIDDLEBURY}/scenes2014/datasets/{scene}-perfect/{name}",
              os.path.join(cache, "2014", f"{scene}-perfect", name))
             for scene in SCENES_2014 for name in FULL_SIZE_FILES]
    jobs += [(f"{MIDDLEBURY}/scenes2021/data/{scene}/{name}", os.path.join(cache, "2021", scene, name))
             for scene in SCENES_2021 for name in FULL_SIZE_FILES]
    return jobs


def prepare_middlebury(out: str, cache: str, max_width: int = 960, workers: int = 6) -> str:
    """Download every Middlebury release with public ground truth into ``out/middlebury``."""
    root = os.path.join(out, "middlebury")
    print("\nmiddlebury")
    fetch_all(middlebury_downloads(cache), workers, "download")

    # MiddEval3: the training set only; the test set has no ground truth.
    for part in ("data", "GT0", "GT1"):
        extract(os.path.join(cache, f"MiddEval3-{part}-Q.zip"), root,
                keep=lambda name: "/trainingQ/" in name)
    for scene in SCENES_2001:
        os.makedirs(os.path.join(root, "2001", scene), exist_ok=True)
        for name in ("im2.ppm", "im6.ppm", "disp2.pgm", "disp6.pgm"):
            shutil.copy2(os.path.join(cache, "2001", scene, name), os.path.join(root, "2001", scene, name))
    for scene in ("cones", "teddy"):         # each archive holds one scene directory
        for archive in (f"{scene}-png-2.zip", f"{scene}H-ppm-2.zip"):
            extract(os.path.join(cache, "2003", archive), os.path.join(root, "2003"))
    for year, scenes in (("2005", SCENES_2005), ("2006", SCENES_2006)):
        for size in SIZES_2005_2006:
            for scene in scenes:
                extract(os.path.join(cache, year, size, f"{scene}-2views.zip"),
                        os.path.join(root, year, size))

    factors = []
    for year, scenes in (("2014", [f"{s}-perfect" for s in SCENES_2014]), ("2021", SCENES_2021)):
        for scene in scenes:
            factors.append(shrink_scene(os.path.join(cache, year, scene),
                                        os.path.join(root, year, scene), max_width))
    print(f"  shrunk {len(factors)} full-size scenes by {sorted(set(factors))} "
          f"to at most {max_width} px wide")
    return root


# --------------------------------------------------------------------------- #
# KITTI
# --------------------------------------------------------------------------- #

def kitti_member(name: str, folders: Sequence[str]) -> bool:
    """A labelled reference frame (_10), its ground truth, or its calibration."""
    if not name.startswith(tuple(folders)):
        return False
    return name.endswith("_10.png") if name.endswith(".png") else True


def prepare_kitti(out: str, cache: str, workers: int = 4) -> List[str]:
    """Download the KITTI 2012 and 2015 training pairs that have ground truth."""
    print("\nkitti")
    fetch_all([(f"{KITTI}/{archive}", os.path.join(cache, archive))
               for archives in KITTI_ARCHIVES.values() for archive, _ in archives],
              workers, "download")
    written = []
    for name, archives in KITTI_ARCHIVES.items():
        target = os.path.join(out, name)
        count = sum(extract(os.path.join(cache, archive), target,
                            keep=lambda member, folders=folders: kitti_member(member, folders))
                    for archive, folders in archives)
        print(f"  {name}: {count} files")
        written.append(target)
    return written


# --------------------------------------------------------------------------- #
# InStereo2K
# --------------------------------------------------------------------------- #

#: Forms an InStereo2K download may come in.
ARCHIVE_SUFFIXES = (".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tgz")


def unpack(archive: str, target: str) -> None:
    """Extract a .zip with Python, a tar with tar, anything else with 7z."""
    os.makedirs(target, exist_ok=True)
    lowered = archive.lower()
    if lowered.endswith(".zip"):
        extract(archive, target)
    elif lowered.endswith((".tar", ".tar.gz", ".tgz")):
        subprocess.run(["tar", "-xf", archive, "-C", target], check=True)
    else:
        tool = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
        if tool is None:
            raise RuntimeError(f"{archive} needs 7z to extract (apt-get install p7zip-full)")
        subprocess.run([tool, "x", "-y", f"-o{target}", archive], check=True,
                       stdout=subprocess.DEVNULL)


def find_instereo2k(search_root: str = "/kaggle/input", max_depth: int = 4) -> Optional[str]:
    """An attached InStereo2K download, found by its name."""
    from .discovery import walk_dirs

    if not os.path.isdir(search_root):
        return None
    for directory in walk_dirs(search_root, max_depth=max_depth):
        if "instereo" in os.path.basename(directory).lower():
            return directory
    return None


def shrink_instereo2k_scene(source: str, target: str, max_width: int) -> int:
    """Copy one InStereo2K scene at ``1/k`` size, keeping its x100 PNG encoding.

    The stored values are reduced in their own units, so whichever scale the
    files truly use survives the shrink unchanged. Returns ``k``.
    """
    from .instereo2k import DISPARITIES, VIEWS

    left = cv2.imread(os.path.join(source, VIEWS[0]), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(f"no {VIEWS[0]} in {source}")
    height, width = left.shape[:2]
    factor = max(1, math.ceil(width / max_width))
    rows, cols = height // factor, width // factor
    os.makedirs(target, exist_ok=True)
    for name in VIEWS:
        image = left if name == VIEWS[0] else cv2.imread(os.path.join(source, name), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(target, name),
                    cv2.resize(image[:rows * factor, :cols * factor], (cols, rows),
                               interpolation=cv2.INTER_AREA))
    for name in DISPARITIES:
        path = os.path.join(source, name)
        if not os.path.exists(path):
            continue
        stored = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if stored.ndim == 3:
            stored = stored[..., 0]
        shrunk = shrink_disparity(np.where(stored > 0, stored, np.inf), factor)
        encoded = np.where(np.isfinite(shrunk), np.round(shrunk), 0)
        cv2.imwrite(os.path.join(target, name), np.clip(encoded, 0, 65535).astype(np.uint16))
    return factor


def prepare_instereo2k(out: str, source: str, cache: str, max_width: int = 960) -> str:
    """Copy an InStereo2K download into ``out/instereo2k``.

    It has no scriptable source, so ``source`` is what was fetched by hand: the
    archive, or a folder holding it or its extracted contents -- such as the
    dataset uploaded to Kaggle.
    """
    from .instereo2k import find_scenes, scene_split

    print(f"\ninstereo2k  <- {source}")
    root = source
    if os.path.isfile(source):
        root = os.path.join(cache, "unpacked")
        unpack(source, root)
    elif not find_scenes(source):
        archives = sorted(os.path.join(directory, name) for directory, _, names in os.walk(source)
                          for name in names if name.lower().endswith(ARCHIVE_SUFFIXES))
        if not archives:
            raise RuntimeError(f"no InStereo2K scenes or archives under {source}")
        root = os.path.join(cache, "unpacked")
        for archive in archives:
            print(f"  unpacking {os.path.basename(archive)}")
            unpack(archive, root)

    scenes = find_scenes(root)
    if not scenes:
        raise RuntimeError(f"no InStereo2K scenes (left.png, right.png, left_disp.png) "
                           f"in {source}")
    counts: Dict[str, int] = defaultdict(int)
    factors = set()
    for scene in scenes:
        relative = os.path.relpath(scene, root)
        factors.add(shrink_instereo2k_scene(scene, os.path.join(out, "instereo2k", relative),
                                            max_width))
        counts[scene_split(root, scene) or "train"] += 1
    print(f"  {', '.join(f'{split} {n}' for split, n in sorted(counts.items()))} scenes, "
          f"shrunk by {sorted(factors)}")
    return os.path.join(out, "instereo2k")


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #

def verify(out: str) -> List[str]:
    """Check every prepared release by warping. Returns the problems found."""
    from .augmentation import HorizontalFlip, HorizontalFlipConfig
    from .base import DatasetMode
    from .kitti import KittiStereoDataset
    from .label_check import check_label_scale
    from .middlebury import MiddleburyDataset

    problems = []
    root = os.path.join(out, "middlebury")
    if os.path.isdir(root):
        plain = MiddleburyDataset(root, mode=DatasetMode.BENCHMARK)
        flipped = MiddleburyDataset(root, mode=DatasetMode.TRAIN,
                                    transform=HorizontalFlip(HorizontalFlipConfig(probability=1.0), 0))
        flipped.with_labels = True
        print(f"\nmiddlebury: {plain.summary()}")
        groups: Dict[str, List[int]] = defaultdict(list)
        for index, scene in enumerate(plain.entries):
            groups[scene.release].append(index)
        for release, indices in sorted(groups.items()):
            picks = indices[::max(1, len(indices) // 4)][:4]
            report = check_label_scale([plain[i] for i in picks])
            mirrored = [i for i in picks if plain.entries[i].disparity_right]
            flip = check_label_scale([flipped[i] for i in mirrored]) if mirrored else None
            ok = report.consistent and (flip is None or flip.consistent)
            print(f"  {release:<10} {len(indices):>3} scenes   labels x{report.best_factor:g}   "
                  f"flipped {'x%g' % flip.best_factor if flip else 'n/a'}   "
                  f"{'ok' if ok else 'WRONG'}")
            if not ok:
                problems.append(f"middlebury {release}: labels look off by "
                                f"x{report.best_factor:g} (flipped "
                                f"{flip.best_factor if flip else 'n/a'})")
            if not mirrored:
                problems.append(f"middlebury {release}: no right-view disparity")

    root = os.path.join(out, "instereo2k")
    if os.path.isdir(root):
        from .instereo2k import DISPARITY_SCALE, InStereo2kDataset

        plain = InStereo2kDataset(root, split=None, mode=DatasetMode.BENCHMARK)
        flipped = InStereo2kDataset(root, split=None, mode=DatasetMode.TRAIN,
                                    transform=HorizontalFlip(HorizontalFlipConfig(probability=1.0), 0))
        flipped.with_labels = True
        picks = list(range(0, len(plain), max(1, len(plain) // 6)))[:6]
        # The README says /100, torchvision reads /1024: the scan includes both.
        factors = (DISPARITY_SCALE / 1024, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
        report = check_label_scale([plain[i] for i in picks], factors=factors)
        flip = check_label_scale([flipped[i] for i in picks], factors=factors)
        ok = report.consistent and flip.consistent
        print(f"  instereo2k {len(plain):>4} pairs   labels x{report.best_factor:.3g}   "
              f"flipped x{flip.best_factor:.3g}   {'ok' if ok else 'WRONG'}")
        if not ok:
            hint = (" -- that is /1024, as torchvision reads it; set DISPARITY_SCALE = 1024 "
                    "in stereo/data/instereo2k.py"
                    if math.isclose(report.best_factor, DISPARITY_SCALE / 1024) else "")
            problems.append(f"instereo2k: labels look off by x{report.best_factor:.3g}{hint}")

    for name, expected in KITTI_PAIRS.items():
        path = os.path.join(out, name)
        if not os.path.isdir(path):
            continue
        dataset = KittiStereoDataset(path, version=name[-4:], mode=DatasetMode.BENCHMARK)
        step = max(1, len(dataset) // 4)
        report = check_label_scale([dataset[i] for i in list(range(0, len(dataset), step))[:4]])
        ok = len(dataset) == expected and report.consistent
        print(f"  {name:<10} {len(dataset):>3} pairs    labels x{report.best_factor:g}   "
              f"{'ok' if ok else 'WRONG'}")
        if len(dataset) != expected:
            problems.append(f"{name}: {len(dataset)} labelled pairs, expected {expected}")
        if not report.consistent:
            problems.append(f"{name}: labels look off by x{report.best_factor:g}")
    return problems


def directory_megabytes(path: str) -> float:
    return sum(os.path.getsize(os.path.join(directory, name))
               for directory, _, names in os.walk(path) for name in names) / 1e6


def find_prepared(search_root: str = "/kaggle/input", max_depth: int = 5) -> Optional[str]:
    """The directory of an attached prepared dataset, found by its manifest."""
    from .discovery import walk_dirs

    if not os.path.isdir(search_root):
        return None
    for directory in walk_dirs(search_root, max_depth=max_depth):
        if os.path.isfile(os.path.join(directory, MANIFEST)):
            return directory
    return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def prepare_all(out: str, cache: Optional[str] = None, max_width: int = 960,
                middlebury: bool = True, kitti: bool = True, workers: int = 6,
                keep_cache: bool = False, instereo2k: Optional[str] = None) -> str:
    """Download, assemble and check everything; writes the manifest last.

    ``instereo2k`` is a hand-made InStereo2K download (archive or folder), or
    ``None`` to leave it out.
    """
    out = os.path.abspath(out)
    cache = cache or os.path.join(tempfile.gettempdir(), "stereo-prepare-cache")
    if os.path.commonpath([out, os.path.abspath(cache)]) == out:
        raise ValueError("the download cache must be outside --out, or the archives "
                         "would end up in the dataset")
    os.makedirs(out, exist_ok=True)
    started = time.time()
    if middlebury:
        prepare_middlebury(out, os.path.join(cache, "middlebury"), max_width, workers)
    if kitti:
        prepare_kitti(out, os.path.join(cache, "kitti"))
    if instereo2k:
        prepare_instereo2k(out, instereo2k, os.path.join(cache, "instereo2k"), max_width)

    problems = verify(out)
    if problems:
        raise RuntimeError("the prepared data failed its checks:\n  " + "\n  ".join(problems))

    contents = {name: round(directory_megabytes(os.path.join(out, name)), 1)
                for name in ("middlebury", "kitti2015", "kitti2012", "instereo2k")
                if os.path.isdir(os.path.join(out, name))}
    with open(os.path.join(out, MANIFEST), "w") as handle:
        json.dump({"format": 1, "max_width": max_width, "megabytes": contents,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, handle, indent=2)
    if not keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
    print(f"\nprepared {sum(contents.values()):.0f} MB in {(time.time() - started) / 60:.1f} min "
          f"-> {out}")
    for name, size in contents.items():
        print(f"  {name:<12} {size:>8.1f} MB")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="directory to write the prepared data to")
    parser.add_argument("--cache", help="where downloads go (default: the system temp dir)")
    parser.add_argument("--max-width", type=int, default=960,
                        help="shrink 2014 and 2021 scenes to at most this many pixels wide")
    parser.add_argument("--skip-middlebury", action="store_true")
    parser.add_argument("--skip-kitti", action="store_true")
    parser.add_argument("--instereo2k", help="an InStereo2K download (archive or folder) to "
                                             "include; it cannot be fetched automatically")
    parser.add_argument("--workers", type=int, default=6, help="parallel Middlebury downloads")
    parser.add_argument("--keep-cache", action="store_true", help="keep the downloads afterwards")
    parser.add_argument("--verify-only", action="store_true",
                        help="check an already prepared --out and exit")
    args = parser.parse_args()
    if args.verify_only:
        problems = verify(args.out)
        print("\n" + ("\n".join(problems) if problems else "all checks passed"))
        raise SystemExit(1 if problems else 0)
    prepare_all(args.out, args.cache, args.max_width, not args.skip_middlebury,
                not args.skip_kitti, args.workers, args.keep_cache, args.instereo2k)


if __name__ == "__main__":
    main()
