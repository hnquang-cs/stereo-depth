"""Scene Flow (FlyingThings3D, Monkaa, Driving) from the official archives, once.

The release is 960 x 540 frames with float PFM disparity. FlyingThings3D's
disparity alone is 93 GB compressed and 104 GB unpacked -- more than a Kaggle
session's disk -- so the archives are *streamed*: fetched in byte ranges a few
hundred MB ahead of the reader, unpacked as they arrive, every file converted
and written small, and every range deleted once read.

Written per subset, under ``out/<subset>/``::

    frames_finalpass/TRAIN/A/0000/left/0006.webp   480 x 270, WebP quality 95
    disparity/TRAIN/A/0000/left/0006.png           480 x 270, both views
    frames_finalpass/TEST/...                      960 x 540, the official files
    disparity/TEST/.../left/....png                960 x 540, left view only
    disparity_encoding.json                        pixels = value / 128, 0 unknown
    sceneflow_index.json                           the frames, so that no session
                                                   walks the tree to find them

Training frames are halved: an exact 2x, so each label is the median of a 2 x 2
block aligned with the image's 2 x 2 average, and all three subsets fit in about
13 GB. TEST stays at full size, so the paper's Table IV is scored on its own
images at its own resolution. Only FlyingThings3D has a TEST split; Monkaa and
Driving have no split and are training data.

Measured: 8 MB/s per connection from the server, about 20 MB/s with four, and
bz2 unpacking at about 27 MB/s on one core -- a few hours for all three subsets.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io import parse_pfm

SCENEFLOW = "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data"
#: subset -> (server folder, archive prefix)
SUBSETS = {"flyingthings3d": ("FlyingThings3D", "flyingthings3d"),
           "monkaa": ("Monkaa", "monkaa"),
           "driving": ("Driving", "driving")}
#: Disparity is a uint16 PNG of pixels x this; 0 is unknown. Range 0-511.99 px.
DISPARITY_UNITS = 128
ENCODING_FILE = "disparity_encoding.json"
INDEX_FILE = "sceneflow_index.json"
PASS_DIR = "frames_finalpass"


def archive_urls(subset: str) -> Tuple[str, str]:
    """The (images, disparity) archives of one subset: finalpass WebP, and PFM."""
    folder, prefix = SUBSETS[subset]
    return (f"{SCENEFLOW}/{folder}/raw_data/{prefix}__frames_finalpass_webp.tar",
            f"{SCENEFLOW}/{folder}/derived_data/{prefix}__disparity.tar.bz2")


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #

class RemoteStream(io.RawIOBase):
    """A remote file read front to back, fetched in byte ranges ahead of the reader.

    Ranges are downloaded several at a time (each retried) into ``cache``, read in
    order and deleted once read, so disk use stays at ``ahead`` ranges however
    large the file -- which is what lets a 93 GB archive through a smaller disk.
    """

    def __init__(self, url: str, cache: str, size: Optional[int] = None,
                 chunk: int = 256 << 20, ahead: int = 6, connections: int = 4):
        super().__init__()
        from .prepare import ranged_size

        self.url, self.cache, self.chunk, self.ahead = url, cache, chunk, ahead
        if size is None and url.startswith("file://"):
            size = os.path.getsize(url[len("file://"):])
        self.size = size if size is not None else ranged_size(url)
        if not self.size:
            raise RuntimeError(f"{url} does not serve byte ranges, so it cannot be streamed")
        os.makedirs(cache, exist_ok=True)
        self.count = math.ceil(self.size / chunk)
        self.position = 0
        self._pool = ThreadPoolExecutor(max_workers=connections)
        self._pending: Dict[int, object] = {}
        self._index = 0
        self._handle = None
        self._schedule()

    def _path(self, index: int) -> str:
        return os.path.join(self.cache, f"range{index:05d}")

    def _fetch(self, index: int) -> str:
        start = index * self.chunk
        end = min(start + self.chunk, self.size) - 1
        path, partial = self._path(index), self._path(index) + ".part"
        for attempt in range(6):
            result = subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-r", f"{start}-{end}",
                                     "-o", partial, self.url], capture_output=True)
            if result.returncode == 0 and os.path.getsize(partial) == end - start + 1:
                os.replace(partial, path)
                return path
            time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"{self.url}: bytes {start}-{end} failed six times "
                           f"({result.stderr.decode(errors='replace').strip()[:200]})")

    def _schedule(self) -> None:
        for index in range(self._index, min(self.count, self._index + self.ahead)):
            if index not in self._pending:
                self._pending[index] = self._pool.submit(self._fetch, index)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while True:
            if self._handle is None:
                if self._index >= self.count:
                    return 0
                self._schedule()
                self._handle = open(self._pending.pop(self._index).result(), "rb")
            read = self._handle.readinto(buffer)
            if read:
                self.position += read
                return read
            self._handle.close()
            os.remove(self._path(self._index))
            self._handle = None
            self._index += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().close()


def stream_members(url: str, cache: str, compressed: bool, size: Optional[int] = None,
                   limit: Optional[int] = None) -> Iterator[Tuple[str, bytes]]:
    """``(name, bytes)`` for each regular file in a remote tar, in archive order.

    ``limit`` stops after that many files: a smoke test of the whole path.
    """
    stream = RemoteStream(url, cache, size=size)
    try:
        reader = io.BufferedReader(stream, buffer_size=4 << 20)
        with tarfile.open(fileobj=reader, mode="r|bz2" if compressed else "r|") as tar:
            count = 0
            for member in tar:
                if not member.isfile():
                    continue
                yield member.name, tar.extractfile(member).read()
                count += 1
                if limit is not None and count >= limit:
                    return
    finally:
        stream.close()


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def destination(name: str) -> Tuple[str, str]:
    """``(path below the subset directory, split)`` for an archive member.

    ``frames_finalpass_webp/TRAIN/A/0000/left/0006.webp`` keeps its tree as
    ``frames_finalpass/TRAIN/A/0000/left/0006.webp``. Monkaa and Driving have no
    split directory; they are training data.
    """
    parts = name.split("/")
    if parts[0].startswith("frames_"):
        parts[0] = PASS_DIR
    split = "TEST" if "TEST" in parts else "TRAIN"
    return "/".join(parts), split


def encode_disparity(disparity: np.ndarray) -> np.ndarray:
    """uint16 of pixels x DISPARITY_UNITS; unknown or out-of-range -> 0."""
    limit = np.iinfo(np.uint16).max / DISPARITY_UNITS
    known = np.isfinite(disparity) & (disparity > 0) & (disparity < limit)
    return np.where(known, np.round(disparity * DISPARITY_UNITS), 0).astype(np.uint16)


def convert_image(name: str, data: bytes, root: str) -> None:
    relative, split = destination(name)
    path = os.path.join(root, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if split == "TEST":
        with open(path, "wb") as handle:              # the official file, untouched
            handle.write(data)
        return
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {name}")
    height, width = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    small = cv2.resize(image[:height, :width], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, small, [cv2.IMWRITE_WEBP_QUALITY, 95])


def convert_disparity(name: str, data: bytes, root: str) -> None:
    from .prepare import shrink_disparity

    relative, split = destination(name)
    if split == "TEST" and "/right/" in name:
        return                                         # scoring reads the left view only
    disparity = parse_pfm(data, name)
    if split == "TRAIN":
        disparity = shrink_disparity(disparity, 2)     # aligned with the image's 2 x 2 average
    path = os.path.join(root, os.path.splitext(relative)[0] + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, encode_disparity(disparity))


def convert_archive(url: str, root: str, cache: str, compressed: bool, workers: int,
                    size: Optional[int] = None, limit: Optional[int] = None) -> int:
    """Stream one archive through the converters; returns the files converted."""
    convert = convert_disparity if compressed else convert_image
    started, count = time.time(), 0
    in_flight = threading.BoundedSemaphore(workers * 4)  # bounds memory: each holds a file
    errors: List[BaseException] = []

    def task(name: str, data: bytes) -> None:
        try:
            convert(name, data, root)
        except BaseException as error:                 # surfaced after the stream
            errors.append(error)
        finally:
            in_flight.release()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, data in stream_members(url, cache, compressed, size=size, limit=limit):
            if name.endswith((".webp", ".png", ".pfm")):
                in_flight.acquire()
                pool.submit(task, name, data)
                count += 1
                if count % 5000 == 0:
                    rate = count / (time.time() - started)
                    print(f"    {count} files, {rate:.0f}/s", flush=True)
            if errors:
                break
    if errors:
        raise RuntimeError(f"converting {url} failed: {errors[0]}") from errors[0]
    print(f"    {count} files in {(time.time() - started) / 60:.1f} min", flush=True)
    return count


# --------------------------------------------------------------------------- #
# The subset as the loader sees it
# --------------------------------------------------------------------------- #

def write_index(root: str) -> Dict[str, int]:
    """Record every frame that has both views and the left view's disparity.

    Keyed by the frames directory relative to ``root`` (``frames_finalpass/TRAIN``,
    or ``frames_finalpass`` for a subset with no split), as the loader looks it up.
    """
    from .sceneflow import find_scene_dirs

    pass_root = os.path.join(root, PASS_DIR)
    splits = [name for name in ("TRAIN", "TEST") if os.path.isdir(os.path.join(pass_root, name))]
    index: Dict[str, List[List[str]]] = {}
    for split in splits or [""]:
        frames = os.path.join(pass_root, split)
        disparity = os.path.join(root, "disparity", split)
        entries = []
        for scene in find_scene_dirs(frames):
            left, right = (os.path.join(frames, scene, view) for view in ("left", "right"))
            names = set(os.listdir(right))
            for name in sorted(os.listdir(left)):
                stem = os.path.splitext(name)[0]
                if name in names and os.path.isfile(os.path.join(disparity, scene, "left", stem + ".png")):
                    entries.append([scene, name])
        index[f"{PASS_DIR}/{split}" if split else PASS_DIR] = entries
    with open(os.path.join(root, INDEX_FILE), "w") as handle:
        json.dump(index, handle)
    with open(os.path.join(root, ENCODING_FILE), "w") as handle:
        json.dump({"format": "png16", "divide_by": DISPARITY_UNITS, "unknown": 0,
                   "train_size": "960x540 halved to 480x270",
                   "test_size": "960x540, the official frames"}, handle, indent=2)
    return {key: len(entries) for key, entries in index.items()}


def prepare_sceneflow(out: str, subsets: Sequence[str], cache: str, workers: Optional[int] = None,
                      limit: Optional[int] = None) -> Dict[str, Dict[str, int]]:
    """Stream, convert and index each subset into ``out/<subset>``.

    ``limit`` converts only the first files of each archive: a quick end-to-end
    check, not a usable dataset.
    """
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    unknown = set(subsets) - set(SUBSETS)
    if unknown:
        raise ValueError(f"unknown Scene Flow subset(s) {sorted(unknown)}; known: {sorted(SUBSETS)}")
    frames: Dict[str, Dict[str, int]] = {}
    for subset in subsets:
        root = os.path.join(out, subset)
        images, disparity = archive_urls(subset)
        print(f"\n{subset}")
        for url, compressed in ((images, False), (disparity, True)):
            print(f"  {url.rsplit('/', 1)[1]}")
            convert_archive(url, root, os.path.join(cache, subset), compressed, workers, limit=limit)
        shutil.rmtree(os.path.join(cache, subset), ignore_errors=True)
        frames[subset] = write_index(root)
        print(f"  frames: {frames[subset]}")
    return frames
