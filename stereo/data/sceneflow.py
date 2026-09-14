"""Scene Flow FlyingThings3D.

Expected layout (the official download, extracted side by side)::

    root/
    |-- frames_finalpass/TRAIN|TEST/<A|B|C>/<scene>/left|right/*.png
    '-- disparity/        TRAIN|TEST/<A|B|C>/<scene>/left|right/*.pfm

``frames_cleanpass`` is used instead if ``pass_name="cleanpass"``.

Disparity sign: the official PFM holds positive left-referenced disparity, which
is this repository's convention, so it is read as-is.  (The reference
implementation negates it because it decodes PFM through OpenCV, which applies
the signed scale factor from the header; :func:`stereo.data.io.read_pfm` uses
the sign only to pick the byte order, as the format specifies.)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .base import DatasetMode, StereoDataset
from .io import read_image, read_middlebury_disparity

#: Scene Flow renders with a virtual camera of focal length 1050 px (35 mm lens)
#: and a 1 m baseline.  Used only for optional depth conversion.
SCENEFLOW_FOCAL_LENGTH = 1050.0
SCENEFLOW_BASELINE = 1.0


class SceneFlowEntry(NamedTuple):
    letter: str
    scene: str
    filename: str

    @property
    def sample_id(self) -> str:
        return f"{self.letter}_{self.scene}_{os.path.splitext(self.filename)[0]}"


class SceneFlowDataset(StereoDataset):
    """FlyingThings3D.

    Args:
        root: directory containing ``frames_finalpass`` and ``disparity``.
        split: ``"TRAIN"`` or ``"TEST"``.  The paper's Table IV reports the
            **TEST** split.
        pass_name: ``"finalpass"`` or ``"cleanpass"``.
    """

    def __init__(self, root: str, split: str = "TRAIN", mode: DatasetMode = DatasetMode.TRAIN,
                 transform=None, pass_name: str = "finalpass", name: Optional[str] = None):
        super().__init__(mode=mode, transform=transform, name=name or f"sceneflow_{split.lower()}")
        self.root = root
        self.split = split.upper()
        self.frames_dir = os.path.join(root, f"frames_{pass_name}", self.split)
        self.disparity_dir = os.path.join(root, "disparity", self.split)
        self.entries = _index_sceneflow(self.frames_dir)
        if not self.entries:
            raise RuntimeError(f"no Scene Flow samples found under {self.frames_dir}")

    def _image_path(self, entry: SceneFlowEntry, view: str) -> str:
        return os.path.join(self.frames_dir, entry.letter, entry.scene, view, entry.filename)

    def _disparity_path(self, entry: SceneFlowEntry, view: str) -> str:
        stem = os.path.splitext(entry.filename)[0]
        return os.path.join(self.disparity_dir, entry.letter, entry.scene, view, stem + ".pfm")

    def _num_samples(self) -> int:
        return len(self.entries)

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        entry = self.entries[index]
        return read_image(self._image_path(entry, "left")), read_image(self._image_path(entry, "right"))

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        return {
            "sample_id": self.entries[index].sample_id,
            "focal_length": SCENEFLOW_FOCAL_LENGTH,
            "baseline": SCENEFLOW_BASELINE,
        }

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        disparity, valid = read_middlebury_disparity(self._disparity_path(self.entries[index], "left"))
        return {"disparity_gt": disparity, "valid_gt_mask": valid.astype(np.float32)}


def _index_sceneflow(frames_dir: str) -> List[SceneFlowEntry]:
    """Index every frame that exists for both views."""
    entries: List[SceneFlowEntry] = []
    if not os.path.isdir(frames_dir):
        return entries
    for letter in sorted(os.listdir(frames_dir)):
        letter_dir = os.path.join(frames_dir, letter)
        if not os.path.isdir(letter_dir):
            continue
        for scene in sorted(os.listdir(letter_dir)):
            left_dir = os.path.join(letter_dir, scene, "left")
            right_dir = os.path.join(letter_dir, scene, "right")
            if not (os.path.isdir(left_dir) and os.path.isdir(right_dir)):
                continue
            for filename in sorted(os.listdir(left_dir)):
                if filename.endswith(".png") and os.path.exists(os.path.join(right_dir, filename)):
                    entries.append(SceneFlowEntry(letter, scene, filename))
    return entries
