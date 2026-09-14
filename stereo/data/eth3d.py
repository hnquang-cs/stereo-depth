"""ETH3D two-view stereo.

The ``two_view_training`` release uses the same per-scene file names as
Middlebury (``im0.png`` / ``im1.png`` / ``disp0GT.pfm`` / ``mask0nocc.png`` /
``calib.txt``), so it is a thin specialisation rather than a separate loader.

ETH3D ground truth is sparse laser scan data projected into the image, so most
pixels are invalid; the ``valid_gt_mask`` is essential and the official
evaluation only scores the measured pixels.
"""

from __future__ import annotations

from typing import Optional

from .base import DatasetMode
from .middlebury import MiddleburyDataset


class Eth3dDataset(MiddleburyDataset):
    def __init__(self, root: str, mode: DatasetMode = DatasetMode.TRAIN, transform=None,
                 name: Optional[str] = None, scenes=None):
        super().__init__(root=root, mode=mode, transform=transform, name=name or "eth3d", scenes=scenes)
