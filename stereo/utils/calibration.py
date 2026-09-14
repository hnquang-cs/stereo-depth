"""Optional camera calibration carried alongside a stereo pair.

Calibration is *metadata*, not supervision: it turns a predicted disparity into
metres.  It never enters a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..data.base import metadata_tensor


@dataclass
class StereoCalibration:
    focal_length: Optional[float] = None      # pixels, fx of the left camera
    baseline: Optional[float] = None          # metres
    principal_point_x: Optional[float] = None
    principal_point_y: Optional[float] = None
    #: Middlebury's x-offset between the two principal points; depth uses d + doffs.
    doffs: float = 0.0

    @property
    def is_metric(self) -> bool:
        return self.focal_length is not None and self.baseline is not None


def calibration_from_batch(metadata: Dict[str, List[Any]], device=None):
    """Extract ``(focal_length, baseline, doffs)`` tensors, or ``None`` if incomplete."""
    focal = metadata_tensor(metadata, "focal_length", None, device)
    baseline = metadata_tensor(metadata, "baseline", None, device)
    if focal is None or baseline is None:
        return None
    doffs = metadata_tensor(metadata, "doffs", 0.0, device)
    return focal, baseline, doffs
