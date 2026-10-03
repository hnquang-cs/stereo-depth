"""InStereo2K: 2,050 indoor stereo pairs with dense structured-light disparity.

Bao et al., "InStereo2K: a large real dataset for stereo matching in indoor
scenes", Sci. China Inf. Sci. 2020; https://github.com/YuhuaXu/StereoDataset.
2,000 pairs for training and 50 for testing, 1080 x 860 px, each scene a
directory of::

    left.png  right.png  left_disp.png  right_disp.png

Disparity is a 16-bit PNG of ``disparity x 100``, with 0 for unknown, per the
dataset's README. torchvision's loader divides by 1024 instead, which its own
issue tracker reports as a bug (pytorch/vision#7129). The two differ by 10x, so
the label check (:func:`stereo.data.check_label_scale`) settles it on the
real files before anything trains on them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import DatasetMode, StereoDataset
from .discovery import GROUND_TRUTH_DIR_NAMES, describe_tree, walk_dirs
from .io import read_image

#: Stored value / this = disparity in pixels (the dataset's README).
DISPARITY_SCALE = 100.0
VIEWS = ("left.png", "right.png")
DISPARITIES = ("left_disp.png", "right_disp.png")
SPLITS = ("train", "test")


def find_scenes(root: str) -> List[str]:
    """Scene directories under ``root`` -- views and left disparity -- at any depth."""
    scenes = []
    for directory in walk_dirs(root, skip_names=GROUND_TRUTH_DIR_NAMES):
        try:
            with os.scandir(directory) as entries:
                files = {entry.name for entry in entries if entry.is_file()}
        except OSError:
            continue
        if {*VIEWS, DISPARITIES[0]} <= files:
            scenes.append(directory)
    return sorted(scenes)


def scene_split(root: str, scene: str) -> Optional[str]:
    """``train`` or ``test``, from the scene's path below ``root``."""
    parts = [part.lower() for part in os.path.relpath(scene, root).split(os.sep)]
    return next((split for split in SPLITS if split in parts), None)


def read_disparity(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """A disparity PNG in pixels, and where it is known."""
    stored = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if stored is None:
        raise FileNotFoundError(f"could not read disparity: {path}")
    if stored.ndim == 3:
        stored = stored[..., 0]
    valid = stored > 0
    return (stored.astype(np.float32) / DISPARITY_SCALE), valid


class InStereo2kDataset(StereoDataset):
    """InStereo2K scenes, found at any depth.

    Args:
        split: ``"train"``, ``"test"``, or ``None`` for both. A mirror that drops
            the split directories counts as all-train.
    """

    def __init__(self, root: str, split: Optional[str] = "train",
                 mode: DatasetMode = DatasetMode.TRAIN, transform=None, name: Optional[str] = None):
        super().__init__(mode=mode, transform=transform, name=name or "instereo2k")
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS} or None, not {split!r}")
        self.root = root
        scenes = find_scenes(root)
        self.scenes = [scene for scene in scenes
                       if split is None or (scene_split(root, scene) or "train") == split]
        if not self.scenes:
            raise RuntimeError(
                f"no InStereo2K {split or ''} scenes under {root}: looked for directories "
                f"holding {', '.join(VIEWS)} and {DISPARITIES[0]}"
                + (f" ({len(scenes)} found in other splits)" if scenes else "")
                + f".\n\nWhat is actually there:\n{describe_tree(root)}")

    def _num_samples(self) -> int:
        return len(self.scenes)

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        scene = self.scenes[index]
        return tuple(read_image(os.path.join(scene, view)) for view in VIEWS)

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        return {"sample_id": os.path.relpath(self.scenes[index], self.root)}

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        scene = self.scenes[index]
        disparity, valid = read_disparity(os.path.join(scene, DISPARITIES[0]))
        ground_truth = {"disparity_gt": disparity, "valid_gt_mask": valid.astype(np.float32)}
        right_path = os.path.join(scene, DISPARITIES[1])
        # The right view's labels serve only the horizontal flip, which only
        # training applies.
        if self.mode is DatasetMode.TRAIN and os.path.isfile(right_path):
            right, right_valid = read_disparity(right_path)
            ground_truth["disparity_gt_right"] = right
            ground_truth["valid_gt_mask_right"] = right_valid.astype(np.float32)
        return ground_truth
