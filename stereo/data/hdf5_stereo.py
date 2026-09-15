"""Stereo pairs packed into an HDF5 file.

Some Kaggle mirrors ship a dataset as one ``.hdf5`` container rather than a tree
of PNGs. The container's internal naming is not standardised, so this module
does two things:

* :func:`inspect_hdf5` prints the file's structure -- every dataset, its shape
  and dtype -- which is the only reliable way to find out what a given container
  actually holds.
* :class:`Hdf5StereoDataset` reads stereo pairs out of it, auto-detecting the
  view arrays by name and shape, or using names you pass explicitly.

Two packings are recognised:

* **separate arrays** -- one array per view, e.g. ``left`` ``(N, H, W, 3)`` and
  ``right`` ``(N, H, W, 3)``
* **stacked array** -- both views in one array with a length-2 axis, e.g.
  ``images (N, 2, H, W, 3)``

Anything else is reported rather than guessed at.

**A container is not automatically a stereo dataset.** Many files named after a
dataset hold something else entirely -- optical flow, classification crops, or a
single view. :func:`inspect_hdf5` is what settles it; the loader refuses to
invent a pairing it cannot verify.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import DatasetMode, StereoDataset

HDF5_EXTENSIONS = (".h5", ".hdf5", ".hdf", ".he5")

#: Array names that plausibly hold each view, most specific first.
LEFT_NAME_HINTS = ("left", "left_image", "left_images", "images_left", "image_left",
                   "im0", "img0", "image_02", "cam0", "l")
RIGHT_NAME_HINTS = ("right", "right_image", "right_images", "images_right", "image_right",
                    "im1", "img1", "image_03", "cam1", "r")
#: Array names that hold ground truth. Never loaded outside BENCHMARK mode.
DISPARITY_NAME_HINTS = ("disparity", "disparities", "disp", "disparity_left", "disp0")


def _require_h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading an HDF5 dataset needs h5py: pip install h5py") from error
    return h5py


def find_hdf5_files(root: str, max_depth: int = 6) -> List[str]:
    """Every HDF5 file at or beneath ``root``."""
    if os.path.isfile(root):
        return [root] if root.lower().endswith(HDF5_EXTENSIONS) else []
    found = []
    for directory, _, filenames in os.walk(root):
        depth = os.path.relpath(directory, root).count(os.sep)
        if depth > max_depth:
            continue
        found.extend(os.path.join(directory, name) for name in sorted(filenames)
                     if name.lower().endswith(HDF5_EXTENSIONS))
    return sorted(found)


def list_arrays(path: str) -> List[Tuple[str, Tuple[int, ...], str]]:
    """``[(name, shape, dtype), ...]`` for every array in the file, groups included."""
    h5py = _require_h5py()
    arrays: List[Tuple[str, Tuple[int, ...], str]] = []

    def visit(name, node):
        if isinstance(node, h5py.Dataset):
            arrays.append((name, tuple(node.shape), str(node.dtype)))

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)
    return arrays


def inspect_hdf5(root: str, max_files: int = 5) -> str:
    """Human-readable structure of the HDF5 file(s) under ``root``.

    This is the cell to run when a container's contents are unknown: it is the
    only way to learn what the arrays are called and what shape they are, which
    is what the loader needs to pair the views.
    """
    files = find_hdf5_files(root)
    if not files:
        return f"no HDF5 file found under {root}"

    lines = []
    for path in files[:max_files]:
        size = os.path.getsize(path) / 1e9
        lines.append(f"{path}  ({size:.2f} GB)")
        try:
            arrays = list_arrays(path)
        except Exception as error:
            lines.append(f"  could not read: {type(error).__name__}: {error}")
            continue
        if not arrays:
            lines.append("  (no arrays)")
        for name, shape, dtype in arrays:
            lines.append(f"    {name:40s} shape={shape}  dtype={dtype}")
        lines.append("")
        lines.append("  How to read this:")
        lines.append("    a stereo container needs two arrays of shape (N, H, W, 3) or")
        lines.append("    (N, 3, H, W) -- one per view -- or one array with a length-2 axis.")
        lines.append("    If nothing here looks like that, this file is not stereo imagery.")
    if len(files) > max_files:
        lines.append(f"... and {len(files) - max_files} more HDF5 files")
    return "\n".join(lines)


def _looks_like_images(shape: Sequence[int]) -> bool:
    """``(N, H, W, 3)`` or ``(N, 3, H, W)`` with plausible spatial extents."""
    if len(shape) != 4:
        return False
    if shape[3] in (1, 3, 4) and shape[1] > 4 and shape[2] > 4:
        return True
    if shape[1] in (1, 3, 4) and shape[2] > 4 and shape[3] > 4:
        return True
    return False


def _match_name(name: str, hints: Sequence[str]) -> int:
    """Rank of the first hint the array's own name matches; ``len(hints)`` if none."""
    leaf = name.rsplit("/", 1)[-1].lower()
    for index, hint in enumerate(hints):
        if leaf == hint:
            return index
        if hint in leaf:
            return index + len(hints)
    return 2 * len(hints)


def detect_view_arrays(arrays: Sequence[Tuple[str, Tuple[int, ...], str]]):
    """Work out how the views are packed.

    Returns ``("separate", left_name, right_name)``,
    ``("stacked", name, view_axis)``, or ``None`` when neither can be established.
    """
    image_arrays = [(name, shape) for name, shape, _ in arrays if _looks_like_images(shape)]

    left = sorted((( _match_name(n, LEFT_NAME_HINTS), n, s) for n, s in image_arrays))
    right = sorted(((_match_name(n, RIGHT_NAME_HINTS), n, s) for n, s in image_arrays))
    if left and right:
        left_rank, left_name, left_shape = left[0]
        right_rank, right_name, right_shape = right[0]
        if (left_name != right_name and left_shape == right_shape
                and left_rank < 2 * len(LEFT_NAME_HINTS)
                and right_rank < 2 * len(RIGHT_NAME_HINTS)):
            return "separate", left_name, right_name

    # Both views stacked on a length-2 axis, e.g. (N, 2, H, W, 3).
    for name, shape, _ in arrays:
        if len(shape) == 5 and 2 in shape[1:3]:
            return "stacked", name, shape.index(2)
    return None


class Hdf5StereoDataset(StereoDataset):
    """Stereo pairs read from an HDF5 container.

    Args:
        root: the ``.hdf5`` file, or a directory containing exactly one.
        left_key / right_key: array names. ``None`` auto-detects.
        disparity_key: ground-truth array, read only in ``BENCHMARK`` mode.
        stacked_key / view_axis: read both views from one array instead.
    """

    def __init__(self, root: str, mode: DatasetMode = DatasetMode.TRAIN, transform=None,
                 name: Optional[str] = None, left_key: Optional[str] = None,
                 right_key: Optional[str] = None, disparity_key: Optional[str] = None,
                 stacked_key: Optional[str] = None, view_axis: int = 1,
                 focal_length: Optional[float] = None, baseline: Optional[float] = None):
        super().__init__(mode=mode, transform=transform, name=name or "hdf5_stereo")
        _require_h5py()   # fail early with a clear message

        files = find_hdf5_files(root)
        if not files:
            raise FileNotFoundError(f"no HDF5 file found under {root}")
        if len(files) > 1 and not os.path.isfile(root):
            raise RuntimeError(
                f"{len(files)} HDF5 files under {root}; point root at the one to use:\n  "
                + "\n  ".join(files[:10]))
        self.path = files[0]
        self._handle = None

        arrays = list_arrays(self.path)
        self.stacked_key, self.view_axis = stacked_key, view_axis
        self.left_key, self.right_key = left_key, right_key

        if not (stacked_key or (left_key and right_key)):
            detected = detect_view_arrays(arrays)
            if detected is None:
                raise RuntimeError(
                    f"could not find a stereo pair inside {self.path}.\n"
                    f"A stereo container needs two image arrays of shape (N, H, W, 3) or "
                    f"(N, 3, H, W) -- one per view -- or one array with a length-2 view axis.\n"
                    f"Pass left_key/right_key (or stacked_key) explicitly if the arrays are "
                    f"named unusually.\n\nWhat the file actually contains:\n"
                    + "\n".join(f"    {n:40s} shape={s}  dtype={d}" for n, s, d in arrays)
                    + "\n\nIf none of these are stereo imagery, this file is not a stereo "
                      "dataset and cannot be used for training.")
            if detected[0] == "separate":
                _, self.left_key, self.right_key = detected
            else:
                _, self.stacked_key, self.view_axis = detected

        self.disparity_key = disparity_key
        if self.disparity_key is None:
            for candidate, shape, _ in arrays:
                if _match_name(candidate, DISPARITY_NAME_HINTS) < len(DISPARITY_NAME_HINTS):
                    self.disparity_key = candidate
                    break
        if mode is DatasetMode.BENCHMARK and self.disparity_key is None:
            raise FileNotFoundError(
                f"{self.path} has no disparity array, so it cannot be benchmarked. "
                f"Arrays present: {[n for n, _, _ in arrays]}")

        self._length = self._read_length()
        self.focal_length = focal_length
        self.baseline = baseline

    # -- file handling -------------------------------------------------------- #

    def _file(self):
        """Open lazily and per worker: an h5py handle cannot cross a fork."""
        if self._handle is None:
            self._handle = _require_h5py().File(self.path, "r")
        return self._handle

    def _read_length(self) -> int:
        with _require_h5py().File(self.path, "r") as handle:
            key = self.stacked_key or self.left_key
            return int(handle[key].shape[0])

    def __getstate__(self):
        """DataLoader workers pickle the dataset; an open h5py handle cannot cross
        a fork, so it is dropped and reopened lazily in the worker."""
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    # -- decoding -------------------------------------------------------------- #

    @staticmethod
    def _to_hwc_float(array: np.ndarray) -> np.ndarray:
        image = np.asarray(array)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[2] > 4:
            image = np.transpose(image, (1, 2, 0))   # CHW -> HWC
        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        image = image[:, :, :3].astype(np.float32)
        if image.max() > 1.5:                         # 0-255 rather than 0-1
            image = image / 255.0
        return np.ascontiguousarray(np.clip(image, 0.0, 1.0))

    # -- dataset hooks --------------------------------------------------------- #

    def _num_samples(self) -> int:
        return self._length

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        handle = self._file()
        if self.stacked_key:
            block = handle[self.stacked_key][index]
            left = np.take(block, 0, axis=self.view_axis - 1)
            right = np.take(block, 1, axis=self.view_axis - 1)
        else:
            left = handle[self.left_key][index]
            right = handle[self.right_key][index]
        return self._to_hwc_float(left), self._to_hwc_float(right)

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"sample_id": f"{index:07d}"}
        if self.focal_length is not None:
            metadata["focal_length"] = self.focal_length
        if self.baseline is not None:
            metadata["baseline"] = self.baseline
        return metadata

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        disparity = np.asarray(self._file()[self.disparity_key][index]).astype(np.float32)
        while disparity.ndim > 2:
            disparity = disparity[..., 0] if disparity.shape[-1] == 1 else disparity[0]
        valid = np.isfinite(disparity) & (disparity > 0)
        return {"disparity_gt": np.where(valid, disparity, 0.0).astype(np.float32),
                "valid_gt_mask": valid.astype(np.float32)}

    def describe(self) -> str:
        if self.stacked_key:
            packing = f"stacked array {self.stacked_key!r} (view axis {self.view_axis})"
        else:
            packing = f"left={self.left_key!r} right={self.right_key!r}"
        return (f"file        : {self.path}\n"
                f"packing     : {packing}\n"
                f"pairs       : {self._length}\n"
                f"disparity   : {self.disparity_key or 'NOT PRESENT -- images only'}")
