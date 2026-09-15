"""Middlebury 2014 / MiddEval3 stereo.

Layout (one directory per scene)::

    root/<scene>/im0.png          left image
                 im1.png          right image
                 disp0GT.pfm      left ground-truth disparity (inf = invalid)
                 mask0nocc.png    255 = non-occluded valid, 128 = occluded
                 calib.txt        cam0/cam1/doffs/baseline/ndisp

MiddEval3 ships resolution variants (``trainingF``/``trainingH``/``trainingQ``);
point ``root`` at the one the protocol calls for.  The paper reports the
**test** set through the official leaderboard, whose ground truth is not public;
this loader therefore serves the *training* set, and the evaluation code labels
the split accordingly.

Depth from disparity on Middlebury uses the dataset's own formula, which
includes the principal-point offset::

    Z = baseline * f / (d + doffs)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import DatasetMode, StereoDataset
from .discovery import describe_tree, find_view_file_pairs
from .io import read_image, read_middlebury_calib, read_middlebury_disparity, read_nonocc_mask


class MiddleburyDataset(StereoDataset):
    """Middlebury-style scene directories (also used for ETH3D, which shares the layout)."""

    left_name = "im0.png"
    right_name = "im1.png"
    disparity_name = "disp0GT.pfm"
    nonocc_name = "mask0nocc.png"

    def __init__(self, root: str, mode: DatasetMode = DatasetMode.TRAIN, transform=None,
                 name: Optional[str] = None, scenes: Optional[List[str]] = None):
        super().__init__(mode=mode, transform=transform, name=name or "middlebury")
        self.root = root
        self.scenes = scenes if scenes is not None else _index_scenes(root, self.left_name, self.right_name)
        if not self.scenes:
            raise RuntimeError(
                f"no {self.name} scenes found under {root}.\n"
                f"Looked for directories containing both {self.left_name} and "
                f"{self.right_name}, at any depth.\n\n"
                f"What is actually there:\n{describe_tree(root)}")

    def _scene_dir(self, index: int) -> str:
        return os.path.join(self.root, self.scenes[index])

    def _num_samples(self) -> int:
        return len(self.scenes)

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        scene = self._scene_dir(index)
        return (read_image(os.path.join(scene, self.left_name)),
                read_image(os.path.join(scene, self.right_name)))

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"sample_id": self.scenes[index]}
        calib_path = os.path.join(self._scene_dir(index), "calib.txt")
        if os.path.exists(calib_path):
            metadata.update(read_middlebury_calib(calib_path))
        return metadata

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        scene = self._scene_dir(index)
        disparity, valid = read_middlebury_disparity(os.path.join(scene, self.disparity_name))
        ground_truth = {"disparity_gt": disparity, "valid_gt_mask": valid.astype(np.float32)}

        nonocc_path = os.path.join(scene, self.nonocc_name)
        if os.path.exists(nonocc_path):
            nonocc = read_nonocc_mask(nonocc_path)
            ground_truth["nonocc_mask"] = (nonocc & valid).astype(np.float32)
        return ground_truth


def _index_scenes(root: str, left_name: str, right_name: str) -> List[str]:
    """Scene directories, found at any depth.

    Mirrors commonly wrap the scenes in extra folders (``MiddEval3/trainingH/``,
    a resolution folder, or just the dataset's own name), so the scenes are
    located by looking for the two view files rather than by assuming they sit
    directly under ``root``. Paths are returned relative to ``root``.
    """
    if not os.path.isdir(root):
        return []
    scenes = [os.path.relpath(path, root)
              for path in find_view_file_pairs(root, left_name, right_name)]
    return sorted(scenes)
