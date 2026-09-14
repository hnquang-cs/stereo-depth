"""Dataset download and preparation.

Every URL here was checked to resolve and serve the stated content type.  Sizes
are the ``Content-Length`` reported by the servers at the time of writing.

    dataset         auto-download   archive size   notes
    --------------  --------------  -------------  --------------------------------
    middlebury      yes             ~110 MB (H)    MiddEval3, images + GT, no login
    eth3d           yes             ~1.1 GB        needs 7z to extract
    kitti2015       yes             ~1.7 GB        public S3 mirror
    kitti2012       yes             ~2.0 GB        public S3 mirror
    sceneflow       yes, but huge   45 GB + 93 GB  FlyingThings3D finalpass + disparity

Scene Flow is the only one where a plain download is impractical on most
machines; ``--dry-run`` prints the exact commands so it can be staged manually or
fetched from a Kaggle dataset mirror instead.

Usage::

    python -m stereo.data.download --list
    python -m stereo.data.download middlebury kitti2015 --root datasets
    python -m stereo.data.download --all --dry-run
    python -m stereo.data.download --verify --root datasets
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Archive:
    url: str
    filename: str
    approximate_bytes: int
    #: Directory, relative to the dataset root, that the archive is extracted into.
    extract_to: str = "."


@dataclass
class DatasetRecipe:
    name: str
    archives: List[Archive]
    #: Path, relative to the dataset root, that must exist afterwards.
    expected_path: str
    #: What to pass as ``--dataset-root`` to evaluate.py.
    usage_root: str
    protocol: str
    description: str = ""
    manual_note: str = ""
    requires_7z: bool = False


MIDDLEBURY_RESOLUTION_SUFFIX = {"F": "F", "H": "H", "Q": "Q"}

RECIPES: Dict[str, DatasetRecipe] = {
    "middlebury": DatasetRecipe(
        name="middlebury",
        archives=[
            Archive("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip",
                    "MiddEval3-data-H.zip", 109_900_544),
            Archive("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip",
                    "MiddEval3-GT0-H.zip", 53_097_428),
        ],
        expected_path="MiddEval3/trainingH",
        usage_root="middlebury/MiddEval3/trainingH",
        protocol="middlebury2014",
        description="Middlebury 2014 / MiddEval3, half resolution. 15 training scenes with "
                    "public ground truth; the 15 test scenes have images only.",
        manual_note="Swap -H for -F (full) or -Q (quarter) in both URLs for other resolutions. "
                    "The paper's Table V is the TEST split, scored by the Middlebury server; "
                    "only the TRAINING split can be scored locally.",
    ),
    "eth3d": DatasetRecipe(
        name="eth3d",
        archives=[
            Archive("https://www.eth3d.net/data/two_view_training.7z",
                    "two_view_training.7z", 1_100_000_000),
            Archive("https://www.eth3d.net/data/two_view_training_gt.7z",
                    "two_view_training_gt.7z", 14_863_383),
        ],
        expected_path="two_view_training",
        usage_root="eth3d/two_view_training",
        protocol="eth3d",
        description="ETH3D low-res two-view stereo, training split with sparse laser ground truth.",
        requires_7z=True,
        manual_note="Extraction needs 7z: 'brew install p7zip' or 'apt install p7zip-full'. "
                    "Both archives extract into the same two_view_training/ tree.",
    ),
    "kitti2015": DatasetRecipe(
        name="kitti2015",
        archives=[
            Archive("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip",
                    "data_scene_flow.zip", 1_681_488_619),
            Archive("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip",
                    "data_scene_flow_calib.zip", 1_631_055),
        ],
        expected_path="training/image_2",
        usage_root="kitti2015/training",
        protocol="kitti2015",
        description="KITTI 2015 stereo: 200 training pairs with semi-dense LiDAR-derived "
                    "disparity, 200 test pairs without.",
        manual_note="The calib archive supplies training/calib_cam_to_cam/, which the loader "
                    "needs for metric depth.",
    ),
    "kitti2012": DatasetRecipe(
        name="kitti2012",
        archives=[
            Archive("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip",
                    "data_stereo_flow.zip", 2_008_641_404),
        ],
        expected_path="training/colored_0",
        usage_root="kitti2012/training",
        protocol="kitti2012",
        description="KITTI 2012 stereo: 194 training pairs with LiDAR disparity.",
    ),
    "sceneflow": DatasetRecipe(
        name="sceneflow",
        archives=[
            Archive("https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/"
                    "Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar",
                    "flyingthings3d__frames_finalpass.tar", 45_212_712_960),
            Archive("https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/"
                    "Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2",
                    "flyingthings3d__disparity.tar.bz2", 93_213_362_434),
        ],
        expected_path="frames_finalpass/TEST",
        usage_root="sceneflow",
        protocol="sceneflow",
        description="Scene Flow FlyingThings3D, final pass. The paper's Table IV benchmark.",
        manual_note="45 GB of images plus 93 GB of disparity. Training here needs only the "
                    "images (label-free), so the disparity archive is required ONLY for "
                    "benchmarking. Download it separately, or skip it and train on images alone.",
    ),
}


def human_bytes(count: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if count < 1024 or unit == "TB":
            return f"{count:.1f} {unit}"
        count /= 1024.0


def _download(url: str, destination: str, dry_run: bool) -> None:
    """Resumable download via curl, falling back to wget then urllib."""
    if os.path.exists(destination):
        print(f"    already downloaded: {destination}")
        return
    partial = destination + ".part"
    if shutil.which("curl"):
        command = ["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", partial, url]
    elif shutil.which("wget"):
        command = ["wget", "-c", "-O", partial, url]
    else:
        command = [sys.executable, "-c",
                   f"import urllib.request;urllib.request.urlretrieve({url!r}, {partial!r})"]
    print(f"    $ {' '.join(command)}")
    if dry_run:
        return
    subprocess.run(command, check=True)
    os.replace(partial, destination)


def _extract(archive_path: str, target_dir: str, dry_run: bool, requires_7z: bool) -> None:
    os.makedirs(target_dir, exist_ok=True)
    if archive_path.endswith(".7z") or requires_7z:
        seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
        if seven_zip is None:
            raise RuntimeError(
                f"{archive_path} needs 7z to extract. Install it with "
                "'brew install p7zip' (macOS) or 'apt-get install p7zip-full' (Debian/Ubuntu).")
        command = [seven_zip, "x", "-y", f"-o{target_dir}", archive_path]
        print(f"    $ {' '.join(command)}")
        if not dry_run:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        return

    print(f"    extracting {archive_path} -> {target_dir}")
    if dry_run:
        return
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)
    elif ".tar" in archive_path:
        mode = "r:bz2" if archive_path.endswith(".bz2") else ("r:gz" if archive_path.endswith(".gz") else "r:")
        with tarfile.open(archive_path, mode) as archive:
            archive.extractall(target_dir)
    else:
        raise RuntimeError(f"do not know how to extract {archive_path}")


def prepare(name: str, root: str, dry_run: bool = False, keep_archives: bool = True) -> str:
    """Download and extract one dataset.  Returns the directory to point configs at."""
    if name not in RECIPES:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(RECIPES)}")
    recipe = RECIPES[name]
    dataset_dir = os.path.join(root, name)
    archive_dir = os.path.join(root, "_archives")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    total = sum(a.approximate_bytes for a in recipe.archives)
    print(f"\n[{name}] {recipe.description}")
    print(f"  download size ~{human_bytes(total)} -> {dataset_dir}")
    if recipe.manual_note:
        print(f"  note: {recipe.manual_note}")

    for archive in recipe.archives:
        archive_path = os.path.join(archive_dir, archive.filename)
        print(f"  {archive.filename} (~{human_bytes(archive.approximate_bytes)})")
        _download(archive.url, archive_path, dry_run)
        target = os.path.join(dataset_dir, archive.extract_to)
        _extract(archive_path, target, dry_run, recipe.requires_7z)
        if not keep_archives and not dry_run and os.path.exists(archive_path):
            os.remove(archive_path)

    usage = os.path.join(root, recipe.usage_root)
    print(f"  ready: --dataset-root {usage}  (protocol: {recipe.protocol})")
    return usage


def verify(root: str) -> Dict[str, bool]:
    """Check which datasets are present and correctly laid out."""
    results = {}
    print(f"checking {os.path.abspath(root)}")
    for name, recipe in RECIPES.items():
        path = os.path.join(root, name, recipe.expected_path)
        present = os.path.isdir(path)
        results[name] = present
        marker = "OK     " if present else "MISSING"
        print(f"  [{marker}] {name:12s} {path}")
        if present:
            print(f"                 use: --dataset-root {os.path.join(root, recipe.usage_root)} "
                  f"--protocol {recipe.protocol}")
    return results


def describe() -> str:
    lines = ["Available datasets:", ""]
    for name, recipe in RECIPES.items():
        total = sum(a.approximate_bytes for a in recipe.archives)
        lines.append(f"  {name:12s} ~{human_bytes(total):>10s}  protocol={recipe.protocol}")
        lines.append(f"               {recipe.description}")
        if recipe.manual_note:
            lines.append(f"               note: {recipe.manual_note}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare stereo benchmark datasets.")
    parser.add_argument("datasets", nargs="*", choices=sorted(RECIPES) + [], help="datasets to fetch")
    parser.add_argument("--root", default="datasets", help="directory to download into")
    parser.add_argument("--all", action="store_true", help="fetch every dataset")
    parser.add_argument("--list", action="store_true", help="describe the datasets and exit")
    parser.add_argument("--verify", action="store_true", help="check what is already prepared")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--delete-archives", action="store_true", help="remove archives after extraction")
    args = parser.parse_args()

    if args.list:
        print(describe())
        return
    if args.verify:
        verify(args.root)
        return

    names = sorted(RECIPES) if args.all else args.datasets
    if not names:
        parser.error("name at least one dataset, or pass --all / --list / --verify")
    for name in names:
        prepare(name, args.root, dry_run=args.dry_run, keep_archives=not args.delete_archives)


if __name__ == "__main__":
    main()
