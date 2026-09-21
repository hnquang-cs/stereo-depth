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


def summarise_arrays(arrays: Sequence[Tuple[str, Tuple[int, ...], str]],
                     max_groups: int = 8, examples: int = 1) -> List[str]:
    """Group arrays by shape and dtype instead of listing them all.

    A keyed container holds one array per image -- tens of thousands of them once
    a dataset ships pre-augmented variants -- so listing every name buries the
    one thing worth knowing: which shapes are present and how many of each.
    """
    groups: Dict[Tuple[Tuple[int, ...], str], List[str]] = {}
    for name, shape, dtype in arrays:
        groups.setdefault((shape, dtype), []).append(name)

    lines = []
    for (shape, dtype), names in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_groups]:
        lines.append(f"    {len(names):>7,} arrays  shape={shape}  dtype={dtype}")
        for name in names[:examples]:
            lines.append(f"            e.g. {name}")
    if len(groups) > max_groups:
        lines.append(f"    ... and {len(groups) - max_groups} more shape/dtype groups")
    return lines


def inspect_hdf5(root: str, max_files: int = 5) -> str:
    """Human-readable structure of the HDF5 file(s) under ``root``.

    Reports what the arrays look like and how many there are, which is what the
    loader needs to pair the views. Arrays are grouped by shape and dtype rather
    than listed individually, because a keyed container can hold tens of
    thousands and the full listing floods the log.
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
            continue
        lines.append(f"  {len(arrays):,} arrays in total:")
        lines.extend(summarise_arrays(arrays))
        lines.append("")
        lines.append("  A stereo container needs two image arrays -- one per view -- either as")
        lines.append("  (N, H, W, 3) stacks, as one array with a length-2 view axis, or as one")
        lines.append("  array per image paired by a 'left'/'right' marker in the name.")
    if len(files) > max_files:
        lines.append(f"... and {len(files) - max_files} more HDF5 files")
    return "\n".join(lines)


#: Markers identifying a view inside a per-sample array name, e.g.
#: ``data/flying/left\\train/0000/0006.png``. Ordered longest-first so that
#: "left" does not match inside a longer token before the longer one is tried.
KEYED_VIEW_MARKERS = (("left", "right"), ("l", "r"), ("im0", "im1"), ("cam0", "cam1"))
#: Markers for the matching ground-truth array, tried in order.
KEYED_DISPARITY_MARKERS = ("disp", "disparity")
#: Extensions the per-sample arrays are named with. Ground truth often differs
#: from the images (``.tif`` disparity beside ``.png`` images).
KEYED_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".pfm", "")


def _replace_last(text: str, old: str, new: str) -> Optional[str]:
    index = text.rfind(old)
    return text[:index] + new + text[index + len(old):] if index >= 0 else None


def detect_keyed_pairs(names: Sequence[str]):
    """Find per-sample arrays paired by a left/right marker in their names.

    Rather than assuming a template, every array name containing a view marker
    is tested by swapping that marker for its partner and checking whether the
    result also exists. That handles the real-world naming seen in the wild --
    ``data/flying/left\\train/0000/0006.png`` paired with
    ``data/flying/right\\train/0000/0006.png`` -- without needing the separator,
    prefix or extension to be configured.

    Returns ``(pairs, marker)`` where ``pairs`` is ``[(left_name, right_name), ...]``.
    """
    available = set(names)
    for left_marker, right_marker in KEYED_VIEW_MARKERS:
        pairs = []
        for name in names:
            if left_marker not in name.lower():
                continue
            # Match case-insensitively but substitute on the real text.
            lowered = name.lower()
            index = lowered.rfind(left_marker)
            candidate = name[:index] + right_marker + name[index + len(left_marker):]
            if candidate in available and candidate != name:
                pairs.append((name, candidate))
        if len(pairs) >= 2:
            return sorted(pairs), (left_marker, right_marker)
    return [], None


def keyed_disparity_name(left_name: str, marker: Tuple[str, str],
                         available: Sequence[str]) -> Optional[str]:
    """The ground-truth array matching a left-view array, if the file has one."""
    available = set(available)
    lowered = left_name.lower()
    index = lowered.rfind(marker[0])
    stem, extension = os.path.splitext(left_name)
    for disparity_marker in KEYED_DISPARITY_MARKERS:
        base = left_name[:index] + disparity_marker + left_name[index + len(marker[0]):]
        if base in available:
            return base
        base_stem = os.path.splitext(base)[0]
        for candidate_extension in KEYED_EXTENSIONS:
            if base_stem + candidate_extension in available:
                return base_stem + candidate_extension
    return None


def keyed_sample_id(left_name: str, marker: Tuple[str, str]) -> str:
    """A stable, readable id: the array name with the view marker removed."""
    lowered = left_name.lower()
    index = lowered.rfind(marker[0])
    key = left_name[index + len(marker[0]):].lstrip("\\/")
    return os.path.splitext(key)[0].replace("\\", "/")


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
                 focal_length: Optional[float] = None, baseline: Optional[float] = None,
                 split: Optional[str] = None, bgr: bool = True):
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
        self.bgr = bgr
        self.keyed_pairs: List[Tuple[str, str]] = []
        self.keyed_disparity: List[Optional[str]] = []
        self.keyed_ids: List[str] = []

        if not (stacked_key or (left_key and right_key)):
            detected = detect_view_arrays(arrays)
            if detected is None:
                # Per-sample arrays, one per image, paired by a marker in the name.
                detected = self._try_keyed(arrays, split, mode)
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
        if self.disparity_key is None and not self.keyed_pairs:
            for candidate, shape, _ in arrays:
                if _match_name(candidate, DISPARITY_NAME_HINTS) < len(DISPARITY_NAME_HINTS):
                    self.disparity_key = candidate
                    break

        has_disparity = bool(self.disparity_key) or any(self.keyed_disparity)
        if mode is DatasetMode.BENCHMARK and not has_disparity:
            raise FileNotFoundError(
                f"{self.path} has no disparity data, so it cannot be benchmarked. "
                f"Arrays present: {[n for n, _, _ in arrays][:20]}")

        self._length = len(self.keyed_pairs) if self.keyed_pairs else self._read_length()
        self.focal_length = focal_length
        self.baseline = baseline

    def _try_keyed(self, arrays, split: Optional[str], mode: DatasetMode):
        """Detect and index per-sample ("keyed") arrays. Returns a sentinel or None."""
        names = [name for name, _, _ in arrays]
        pairs, marker = detect_keyed_pairs(names)
        if not pairs:
            return None

        ids = [keyed_sample_id(left, marker) for left, _ in pairs]

        # Keys commonly carry the split as their first component ("train/...",
        # "val/..."). Honour it, so the evaluation split can be held out of
        # training rather than silently mixed in.
        available_splits = sorted({i.split("/")[0].lower() for i in ids
                                   if "/" in i and i.split("/")[0].lower()
                                   in ("train", "val", "test", "validation")})
        if split and available_splits:
            wanted = {"TRAIN": ("train",), "TEST": ("val", "test", "validation")}[split.upper()]
            selected = [k for k in range(len(ids)) if ids[k].split("/")[0].lower() in wanted]
            if not selected:
                raise FileNotFoundError(
                    f"{self.path} has no {split!r} split. Splits present: {available_splits}")
            pairs = [pairs[k] for k in selected]
            ids = [ids[k] for k in selected]

        self.keyed_pairs = pairs
        self.keyed_ids = ids
        self.keyed_disparity = [keyed_disparity_name(left, marker, names) for left, _ in pairs]
        self.keyed_splits = available_splits
        return "keyed", marker[0], marker[1]

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

    def _to_hwc_float(self, array: np.ndarray) -> np.ndarray:
        image = np.asarray(array)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[2] > 4:
            image = np.transpose(image, (1, 2, 0))   # CHW -> HWC
        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        image = image[:, :, :3].astype(np.float32)
        if getattr(self, "bgr", False):
            image = image[:, :, ::-1]      # the container stores BGR; this repo uses RGB
        if image.max() > 1.5:                         # 0-255 rather than 0-1
            image = image / 255.0
        return np.ascontiguousarray(np.clip(image, 0.0, 1.0))

    # -- dataset hooks --------------------------------------------------------- #

    def _num_samples(self) -> int:
        return self._length

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        handle = self._file()
        if self.keyed_pairs:
            left_name, right_name = self.keyed_pairs[index]
            return (self._to_hwc_float(handle[left_name][()]),
                    self._to_hwc_float(handle[right_name][()]))
        if self.stacked_key:
            block = handle[self.stacked_key][index]
            left = np.take(block, 0, axis=self.view_axis - 1)
            right = np.take(block, 1, axis=self.view_axis - 1)
        else:
            left = handle[self.left_key][index]
            right = handle[self.right_key][index]
        return self._to_hwc_float(left), self._to_hwc_float(right)

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        sample_id = self.keyed_ids[index] if self.keyed_ids else f"{index:07d}"
        metadata: Dict[str, Any] = {"sample_id": sample_id}
        if self.focal_length is not None:
            metadata["focal_length"] = self.focal_length
        if self.baseline is not None:
            metadata["baseline"] = self.baseline
        return metadata

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        if self.keyed_pairs:
            name = self.keyed_disparity[index]
            if name is None:
                raise FileNotFoundError(
                    f"no disparity array for {self.keyed_ids[index]!r} in {self.path}")
            disparity = np.asarray(self._file()[name][()]).astype(np.float32)
        else:
            disparity = np.asarray(self._file()[self.disparity_key][index]).astype(np.float32)
        while disparity.ndim > 2:
            disparity = disparity[..., 0] if disparity.shape[-1] == 1 else disparity[0]
        valid = np.isfinite(disparity) & (disparity > 0)
        return {"disparity_gt": np.where(valid, disparity, 0.0).astype(np.float32),
                "valid_gt_mask": valid.astype(np.float32)}

    def describe(self) -> str:
        if self.keyed_pairs:
            packing = (f"keyed: one array per image, {len(self.keyed_pairs)} pairs\n"
                       f"              e.g. {self.keyed_pairs[0][0]}")
        elif self.stacked_key:
            packing = f"stacked array {self.stacked_key!r} (view axis {self.view_axis})"
        else:
            packing = f"left={self.left_key!r} right={self.right_key!r}"
        return (f"file        : {self.path}\n"
                f"packing     : {packing}\n"
                f"pairs       : {self._length}\n"
                f"disparity   : {self._disparity_summary()}"
                + (f"\nsplits      : {', '.join(self.keyed_splits)}"
                   if getattr(self, "keyed_splits", None) else ""))

    def _disparity_summary(self) -> str:
        if self.keyed_pairs:
            found = sum(1 for d in self.keyed_disparity if d)
            if not found:
                return "NOT PRESENT -- images only"
            return f"{found}/{len(self.keyed_pairs)} samples, e.g. {next(d for d in self.keyed_disparity if d)}"
        return self.disparity_key or "NOT PRESENT -- images only"
