"""Teacher pseudo-label loss and its reliability filter (Stage 2).

The teacher's disparity is *not* ground truth: it is the EMA model's own
prediction, and accepting all of it is how self-training collapses.  A teacher
pixel is used only when every cheap geometric check agrees that it is
trustworthy:

    reliable = matchability confidence >= tau_conf
             & left-right agreement    <  tau_lr  pixels
             & photometric residual    <  tau_photo
             & the warp landed inside the source image
             & disparity is inside the model's search range

All of these are computed from images and teacher predictions only.  None of
them touches ground truth, and the resulting target is detached before it
reaches the student.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PseudoLabelFilterConfig:
    """Thresholds of the reliability filter (all label-free)."""
    confidence_threshold: float = 0.5
    lr_threshold: float = 1.0
    photometric_threshold: float = 0.15
    use_confidence: bool = True
    use_lr: bool = True
    use_photometric: bool = True


def build_pseudo_label_mask(disparity_teacher: torch.Tensor,
                            config: PseudoLabelFilterConfig,
                            max_disparity: float,
                            confidence: Optional[torch.Tensor] = None,
                            lr_error: Optional[torch.Tensor] = None,
                            photometric_residual: Optional[torch.Tensor] = None,
                            valid_warp: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``(B, 1, H, W)`` float mask selecting usable teacher pixels.

    All inputs are expected at the resolution of ``disparity_teacher``; the
    confidence map, which lives at cost-volume resolution, is upsampled here.
    """
    mask = ((disparity_teacher > 0.0) & (disparity_teacher < max_disparity)).to(disparity_teacher.dtype)

    if valid_warp is not None:
        mask = mask * valid_warp
    if config.use_confidence and confidence is not None:
        if confidence.shape[-2:] != disparity_teacher.shape[-2:]:
            confidence = F.interpolate(confidence, size=disparity_teacher.shape[-2:],
                                       mode="bilinear", align_corners=False)
        mask = mask * (confidence >= config.confidence_threshold).to(mask.dtype)
    if config.use_lr and lr_error is not None:
        mask = mask * (lr_error < config.lr_threshold).to(mask.dtype)
    if config.use_photometric and photometric_residual is not None:
        mask = mask * (photometric_residual < config.photometric_threshold).to(mask.dtype)
    return mask


class PseudoLabelLoss(nn.Module):
    """Masked smooth-L1 between the student's disparity and the teacher's.

    Smooth L1 (Huber) rather than L1 because surviving teacher outliers should
    not dominate; ``beta`` is in pixels of the map being compared.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, disparity_student: torch.Tensor, disparity_teacher: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if disparity_student.shape != disparity_teacher.shape:
            raise ValueError(f"student {tuple(disparity_student.shape)} and teacher "
                             f"{tuple(disparity_teacher.shape)} disparities must have the same shape")
        # The teacher target never carries gradient.
        target = disparity_teacher.detach()
        mask = mask.detach()
        per_pixel = F.smooth_l1_loss(disparity_student, target, reduction="none", beta=self.beta)
        total = mask.sum()
        loss = (per_pixel * mask).sum() / (total + 1e-6)
        return {
            "loss": loss,
            "valid_ratio": total / mask.numel(),
        }
