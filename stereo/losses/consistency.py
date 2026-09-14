"""Left-right consistency and the occlusion mask derived from it.

Geometry (derived in :mod:`stereo.geometry`): for a point visible in both views,

    d_L(x) = d_R(x - d_L(x))

so the residual ``d_L(x) - d_R(x - d_L(x))`` is zero exactly where the two
predictions agree geometrically.  Where it is large the pixel is either occluded
(visible in the left view only) or simply mis-matched -- either way its
photometric residual and its teacher pseudo-label are untrustworthy.

The same quantity therefore serves three jobs, and they are deliberately
separated into three functions so the mask logic is auditable:
  * :class:`LeftRightConsistencyLoss` -- a training regulariser,
  * :func:`occlusion_mask` -- reliability gating of the photometric term,
  * the pseudo-label filter in :mod:`stereo.losses.pseudo_label`.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..geometry import left_right_difference, warp_left_to_right, warp_right_to_left


class LeftRightConsistencyLoss(nn.Module):
    """Mean absolute left-right disparity disagreement, both directions."""

    def forward(self, disparity_left: torch.Tensor, disparity_right: torch.Tensor) -> Dict[str, torch.Tensor]:
        difference_left, valid_left = left_right_difference(disparity_left, disparity_right)
        warped_left_disp, valid_right = warp_left_to_right(disparity_left, disparity_right)
        difference_right = disparity_right - warped_left_disp

        left_error = torch.abs(difference_left)
        right_error = torch.abs(difference_right)
        loss = ((left_error * valid_left).sum() + (right_error * valid_right).sum()) \
            / (valid_left.sum() + valid_right.sum() + 1e-6)
        return {
            "loss": loss,
            "error_left": left_error,
            "error_right": right_error,
            "valid_left": valid_left,
            "valid_right": valid_right,
        }


def occlusion_mask(disparity_reference: torch.Tensor, disparity_other: torch.Tensor,
                   direction: str = "left", threshold: float = 1.0) -> torch.Tensor:
    """``(B, 1, H, W)`` mask that is 1 where the two views agree to within ``threshold`` pixels.

    Args:
        disparity_reference: disparity of the reference view.
        disparity_other: disparity of the opposite view.
        direction: ``"left"`` if the reference is the left view, ``"right"`` otherwise.
        threshold: agreement tolerance in pixels of the reference map.
    """
    if direction == "left":
        warped, valid = warp_right_to_left(disparity_other, disparity_reference)
    elif direction == "right":
        warped, valid = warp_left_to_right(disparity_other, disparity_reference)
    else:
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")
    agree = (torch.abs(disparity_reference - warped) < threshold).to(disparity_reference.dtype)
    return agree * valid
