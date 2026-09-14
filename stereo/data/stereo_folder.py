"""The unlabeled-first dataset: a folder of left/right images and nothing else.

    dataset/
    |-- left/
    |   |-- 000001.png
    |   '-- ...
    '-- right/
        |-- 000001.png
        '-- ...

This is the interface a custom stereo camera is expected to produce, and the
whole training pipeline works with exactly this and no labels.

Optionally, a ``calib.txt`` (Middlebury/ETH3D syntax) or a ``calib.json`` at the
dataset root supplies focal length and baseline for metric depth.  Calibration
is metadata, not supervision: it converts predicted disparity to metres and is
never an optimisation target.

For benchmarking a custom capture that *does* have ground truth, a
``left_disparity/`` directory of ``.npz``/``.pfm``/16-bit ``.png`` files can be
provided; it is read only in ``DatasetMode.BENCHMARK``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import DatasetMode, StereoDataset
from .io import list_images, read_image, read_kitti_disparity, read_middlebury_calib, read_pfm

LEFT_DIR = "left"
RIGHT_DIR = "right"
LEFT_DISPARITY_DIR = "left_disparity"


class StereoFolderDataset(StereoDataset):
    """Paired ``left/`` and ``right/`` image folders.

    Args:
        root: dataset directory.
        mode: :class:`~stereo.data.base.DatasetMode`.
        transform: per-sample transform (training modes only).
        name: name recorded in metadata.
        focal_length / baseline: override or supply calibration.
    """

    def __init__(self, root: str, mode: DatasetMode = DatasetMode.TRAIN, transform=None,
                 name: Optional[str] = None, focal_length: Optional[float] = None,
                 baseline: Optional[float] = None):
        super().__init__(mode=mode, transform=transform, name=name or os.path.basename(os.path.normpath(root)))
        self.root = root
        self.left_dir = os.path.join(root, LEFT_DIR)
        self.right_dir = os.path.join(root, RIGHT_DIR)
        self.disparity_dir = os.path.join(root, LEFT_DISPARITY_DIR)

        left_names = list_images(self.left_dir)
        right_names = set(list_images(self.right_dir))
        self.names = [name for name in left_names if name in right_names]
        if not self.names:
            raise RuntimeError(f"no matching left/right image pairs under {root}")
        if len(self.names) != len(left_names):
            missing = len(left_names) - len(self.names)
            print(f"[StereoFolderDataset] {root}: skipping {missing} left images without a right partner")

        self.calibration = _load_calibration(root)
        if focal_length is not None:
            self.calibration["focal_length"] = focal_length
        if baseline is not None:
            self.calibration["baseline"] = baseline

    def _num_samples(self) -> int:
        return len(self.names)

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        name = self.names[index]
        return (read_image(os.path.join(self.left_dir, name)),
                read_image(os.path.join(self.right_dir, name)))

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"sample_id": os.path.splitext(self.names[index])[0]}
        metadata.update(self.calibration)
        return metadata

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        name = os.path.splitext(self.names[index])[0]
        disparity, valid = _read_any_disparity(self.disparity_dir, name)
        return {"disparity_gt": disparity, "valid_gt_mask": valid.astype(np.float32)}


def _load_calibration(root: str) -> Dict[str, float]:
    calib_txt = os.path.join(root, "calib.txt")
    if os.path.exists(calib_txt):
        return read_middlebury_calib(calib_txt)
    calib_json = os.path.join(root, "calib.json")
    if os.path.exists(calib_json):
        with open(calib_json) as handle:
            raw = json.load(handle)
        return {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}
    return {}


def _read_any_disparity(directory: str, stem: str) -> Tuple[np.ndarray, np.ndarray]:
    npz_path = os.path.join(directory, stem + ".npz")
    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            disparity = data[list(data.keys())[0]].astype(np.float32)
        valid = np.isfinite(disparity) & (disparity > 0)
        return np.where(valid, disparity, 0.0).astype(np.float32), valid

    pfm_path = os.path.join(directory, stem + ".pfm")
    if os.path.exists(pfm_path):
        disparity = read_pfm(pfm_path)
        valid = np.isfinite(disparity) & (disparity > 0)
        return np.where(valid, disparity, 0.0).astype(np.float32), valid

    png_path = os.path.join(directory, stem + ".png")
    if os.path.exists(png_path):
        return read_kitti_disparity(png_path)

    raise FileNotFoundError(f"no disparity ground truth for {stem!r} under {directory}")
