"""KITTI stereo: the 2012/2015 benchmarks and the raw/Eigen-split recordings.

Three layouts, discovered structurally rather than assumed:

**benchmark 2015** ``training/image_2/*_10.png`` + ``image_3``, with
``disp_occ_0`` / ``disp_noc_0`` disparity and ``calib_cam_to_cam``.

**benchmark 2012** ``training/colored_0`` + ``colored_1``, ``disp_occ`` /
``disp_noc``, ``calib``.

**raw / Eigen split** the recordings the monocular-depth literature uses::

    <date>/<date>_drive_NNNN_sync/image_02/data/*.png
    <date>/<date>_drive_NNNN_sync/image_03/data/*.png
    <date>/calib_cam_to_cam.txt

``image_02`` and ``image_03`` are the rectified colour cameras, so they form a
rectified stereo pair -- which is all label-free training needs. The raw
recordings carry no disparity ground truth (the Eigen protocol evaluates
*depth* against projected LiDAR, a different benchmark the paper does not
report), so this layout is training-only and says so if asked to benchmark.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .base import DatasetMode, StereoDataset
from .discovery import (describe_missing_stereo, describe_tree, find_view_dir_pairs,
                        paired_filenames, view_image_dir)
from .io import read_image, read_kitti_calib, read_kitti_disparity

#: Disparity directory per benchmark version and occlusion convention.
DISPARITY_DIRS = {
    ("2015", "occ"): "disp_occ_0", ("2015", "noc"): "disp_noc_0",
    ("2012", "occ"): "disp_occ", ("2012", "noc"): "disp_noc",
}
#: View directories that identify each layout.
LAYOUT_VIEWS = {"2015": ("image_2", "image_3"), "2012": ("colored_0", "colored_1"),
                "raw": ("image_02", "image_03")}


class KittiEntry(NamedTuple):
    pair_dir: str      # directory holding the two view directories
    views: Tuple[str, str]
    filename: str

    @property
    def sample_id(self) -> str:
        return f"{os.path.basename(self.pair_dir)}_{os.path.splitext(self.filename)[0]}"


class KittiStereoDataset(StereoDataset):
    """KITTI stereo, benchmark or raw.

    Args:
        root: the ``training``/``testing`` directory, or any directory containing
            KITTI recordings -- the layout is found by searching.
        version: ``"2015"``, ``"2012"``, ``"raw"``, or ``None`` to detect.
        occlusion: ``"occ"`` (all pixels, the D1-all convention) or ``"noc"``.
            Benchmark layouts only.
        reference_frames_only: restrict to the ``_10`` frames that carry labels.
            Defaults to ``True`` in ``BENCHMARK`` mode on a benchmark layout.
    """

    def __init__(self, root: str, version: Optional[str] = None,
                 mode: DatasetMode = DatasetMode.TRAIN, transform=None, occlusion: str = "occ",
                 reference_frames_only: Optional[bool] = None, name: Optional[str] = None):
        if occlusion not in ("occ", "noc"):
            raise ValueError(f"occlusion must be 'occ' or 'noc', got {occlusion!r}")
        self.root = root
        self.occlusion = occlusion
        self.version = self._detect_version(root, version)
        super().__init__(mode=mode, transform=transform, name=name or f"kitti{self.version}")

        if reference_frames_only is None:
            reference_frames_only = (mode == DatasetMode.BENCHMARK and self.version != "raw")

        self.entries = self._index(reference_frames_only)
        if not self.entries:
            raise RuntimeError(
                f"no KITTI {self.version} stereo pairs found under {root}.\n"
                f"Looked for {LAYOUT_VIEWS[self.version]} directories containing images.\n\n"
                f"What is actually there:\n{describe_tree(root)}")

        if mode is DatasetMode.BENCHMARK and self.version == "raw":
            raise RuntimeError(
                "KITTI raw / Eigen-split recordings carry no disparity ground truth, so they "
                "cannot be used for disparity benchmarking. (The Eigen protocol scores *depth* "
                "against projected LiDAR, which is a different benchmark, and the paper reports "
                "no KITTI accuracy at all -- only runtimes, in its Table III.)\n"
                "Use this dataset for label-free training, and benchmark on kitti2015, "
                "middlebury2014 or eth3d.")

    # -- layout detection ---------------------------------------------------- #

    @staticmethod
    def _detect_version(root: str, version: Optional[str]) -> str:
        if version is not None:
            if version not in LAYOUT_VIEWS:
                raise ValueError(f"version must be one of {sorted(LAYOUT_VIEWS)}, got {version!r}")
            return version
        # Prefer a benchmark layout when present: it is the one with ground truth.
        for candidate in ("2015", "2012", "raw"):
            if find_view_dir_pairs(root, [LAYOUT_VIEWS[candidate]], max_depth=6):
                return candidate
        explanation = describe_missing_stereo(root)
        raise RuntimeError(
            f"could not identify a KITTI stereo layout under {root}.\n"
            f"Looked for any of {list(LAYOUT_VIEWS.values())} as sibling directories.\n\n"
            + (explanation + "\n\n" if explanation else "")
            + f"What is actually there:\n{describe_tree(root)}")

    def _index(self, reference_frames_only: bool) -> List[KittiEntry]:
        views = LAYOUT_VIEWS[self.version]
        entries: List[KittiEntry] = []
        for pair_dir, found_views in find_view_dir_pairs(self.root, [views], max_depth=6):
            left_dir = view_image_dir(pair_dir, found_views[0])
            right_dir = view_image_dir(pair_dir, found_views[1])
            for filename in paired_filenames(left_dir, right_dir):
                if reference_frames_only and not filename.endswith("_10.png"):
                    continue
                entries.append(KittiEntry(pair_dir, found_views, filename))
        return entries

    # -- paths ---------------------------------------------------------------- #

    def _image_path(self, entry: KittiEntry, side: int) -> str:
        return os.path.join(view_image_dir(entry.pair_dir, entry.views[side]), entry.filename)

    def _calibration(self, entry: KittiEntry) -> Dict[str, float]:
        """Search upward for the calibration file belonging to this recording."""
        candidates = []
        if self.version == "raw":
            # <date>/calib_cam_to_cam.txt, one level above the drive directory.
            candidates.append(os.path.join(os.path.dirname(entry.pair_dir), "calib_cam_to_cam.txt"))
            candidates.append(os.path.join(entry.pair_dir, "calib_cam_to_cam.txt"))
        else:
            stem = entry.filename.split("_")[0]
            for folder in ("calib_cam_to_cam", "calib"):
                candidates.append(os.path.join(entry.pair_dir, folder, stem + ".txt"))
        for path in candidates:
            if os.path.isfile(path):
                return read_kitti_calib(path)
        return {}

    def _disparity_path(self, entry: KittiEntry) -> Optional[str]:
        directory = DISPARITY_DIRS.get((self.version, self.occlusion))
        if directory is None:
            return None
        path = os.path.join(entry.pair_dir, directory, entry.filename)
        return path if os.path.isfile(path) else None

    # -- dataset hooks --------------------------------------------------------- #

    def _num_samples(self) -> int:
        return len(self.entries)

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        entry = self.entries[index]
        return read_image(self._image_path(entry, 0)), read_image(self._image_path(entry, 1))

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        metadata: Dict[str, Any] = {"sample_id": entry.sample_id}
        metadata.update(self._calibration(entry))
        return metadata

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        path = self._disparity_path(self.entries[index])
        if path is None:
            raise FileNotFoundError(
                f"no {self.occlusion} disparity for {self.entries[index].sample_id} "
                f"under {self.entries[index].pair_dir}")
        disparity, valid = read_kitti_disparity(path)
        return {"disparity_gt": disparity, "valid_gt_mask": valid.astype(np.float32)}
