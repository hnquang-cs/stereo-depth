"""KITTI 2012 and KITTI 2015 stereo.

Layout (official ``data_stereo_flow`` / ``data_scene_flow`` downloads)::

    KITTI 2012: training/colored_0/*_10.png, colored_1/*_10.png,
                disp_occ/*_10.png, disp_noc/*_10.png, calib/*.txt
    KITTI 2015: training/image_2/*_10.png,   image_3/*_10.png,
                disp_occ_0/*_10.png, disp_noc_0/*_10.png, calib_cam_to_cam/*.txt

Only the reference frame ``_10`` has ground truth, so only those pairs are
indexed for benchmarking.  In training mode every stereo pair in the image
directories is used, ground truth or not -- images are all training needs.

The ``testing/`` split has no public ground truth and can only be used for
label-free training or for producing a submission.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import DatasetMode, StereoDataset
from .io import list_images, read_image, read_kitti_calib, read_kitti_disparity

_LAYOUTS = {
    "2012": {"left": "colored_0", "right": "colored_1", "disp_occ": "disp_occ",
             "disp_noc": "disp_noc", "calib": "calib"},
    "2015": {"left": "image_2", "right": "image_3", "disp_occ": "disp_occ_0",
             "disp_noc": "disp_noc_0", "calib": "calib_cam_to_cam"},
}


class KittiStereoDataset(StereoDataset):
    """KITTI stereo.

    Args:
        root: the ``training`` or ``testing`` directory.
        version: ``"2012"`` or ``"2015"``.
        occlusion: ``"occ"`` (all pixels, the D1-all convention) or ``"noc"``
            (non-occluded only).  Ground truth only.
        reference_frames_only: restrict to the ``_10`` frames that have labels.
    """

    def __init__(self, root: str, version: str = "2015", mode: DatasetMode = DatasetMode.TRAIN,
                 transform=None, occlusion: str = "occ", reference_frames_only: Optional[bool] = None,
                 name: Optional[str] = None):
        super().__init__(mode=mode, transform=transform, name=name or f"kitti{version}")
        if version not in _LAYOUTS:
            raise ValueError(f"version must be '2012' or '2015', got {version!r}")
        if occlusion not in ("occ", "noc"):
            raise ValueError(f"occlusion must be 'occ' or 'noc', got {occlusion!r}")
        layout = _LAYOUTS[version]
        self.root = root
        self.version = version
        self.left_dir = os.path.join(root, layout["left"])
        self.right_dir = os.path.join(root, layout["right"])
        self.disparity_dir = os.path.join(root, layout[f"disp_{occlusion}"])
        self.calib_dir = os.path.join(root, layout["calib"])

        if reference_frames_only is None:
            reference_frames_only = mode == DatasetMode.BENCHMARK
        right_names = set(list_images(self.right_dir))
        names = [n for n in list_images(self.left_dir) if n in right_names]
        if reference_frames_only:
            names = [n for n in names if n.endswith("_10.png")]
        self.names = names
        if not self.names:
            raise RuntimeError(f"no KITTI {version} stereo pairs found under {root}")

    def _num_samples(self) -> int:
        return len(self.names)

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        name = self.names[index]
        return (read_image(os.path.join(self.left_dir, name)),
                read_image(os.path.join(self.right_dir, name)))

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        name = self.names[index]
        metadata: Dict[str, Any] = {"sample_id": os.path.splitext(name)[0]}
        calib_path = os.path.join(self.calib_dir, name.split("_")[0] + ".txt")
        if os.path.exists(calib_path):
            metadata.update(read_kitti_calib(calib_path))
        return metadata

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        path = os.path.join(self.disparity_dir, self.names[index])
        disparity, valid = read_kitti_disparity(path)
        return {"disparity_gt": disparity, "valid_gt_mask": valid.astype(np.float32)}
