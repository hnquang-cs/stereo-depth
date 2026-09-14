"""Photometric reconstruction loss -- the primary label-free training signal.

Formulation
-----------
Following Monodepth (Godard et al., 2017), the appearance-matching loss between
a reconstructed view ``I_hat`` and the real view ``I`` combines SSIM and L1::

    L_photo = alpha * (1 - SSIM(I, I_hat)) / 2  +  (1 - alpha) * |I - I_hat|

with ``alpha = 0.85`` and SSIM computed over 3x3 blocks.

Deviations from the local `monodepth` reference, and why
--------------------------------------------------------
* That implementation uses a plain MSE reprojection loss with no SSIM term.  MSE
  is strongly dominated by a few high-residual pixels (occlusions, specular
  highlights), which is exactly the failure mode stereo photometric training has
  to survive, so the published SSIM+L1 form is used instead.
* That implementation warps with ``grid_sample`` on a ``[-1, 1]`` grid while
  adding a disparity expressed as a fraction of image width, which is off by a
  factor of two.  Warping here goes through :mod:`stereo.geometry`, in pixels.

Every reduction is masked: photometric residuals are only meaningful where the
warp actually landed inside the source image.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import warp_left_to_right, warp_right_to_left

SSIM_C1 = 0.01 ** 2
SSIM_C2 = 0.03 ** 2


def ssim(pred: torch.Tensor, target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Per-pixel structural similarity over ``kernel_size`` blocks.

    Uses average pooling with reflection padding so the output keeps the input
    resolution.  Returns values in ``[-1, 1]``; ``(1 - SSIM) / 2`` is the loss.
    """
    pad = kernel_size // 2
    pred_p = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    target_p = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    pool = lambda x: F.avg_pool2d(x, kernel_size, stride=1)
    mu_p, mu_t = pool(pred_p), pool(target_p)
    sigma_p = pool(pred_p ** 2) - mu_p ** 2
    sigma_t = pool(target_p ** 2) - mu_t ** 2
    sigma_pt = pool(pred_p * target_p) - mu_p * mu_t

    numerator = (2 * mu_p * mu_t + SSIM_C1) * (2 * sigma_pt + SSIM_C2)
    denominator = (mu_p ** 2 + mu_t ** 2 + SSIM_C1) * (sigma_p + sigma_t + SSIM_C2)
    return numerator / (denominator + 1e-12)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean of ``values`` over ``mask`` (``mask`` is broadcast over channels)."""
    if mask.shape[1] == 1 and values.shape[1] != 1:
        mask = mask.expand_as(values)
    total = mask.sum()
    return (values * mask).sum() / (total + eps)


def border_mask(like: torch.Tensor, border: int) -> torch.Tensor:
    """``(B, 1, H, W)`` mask that is 0 within ``border`` pixels of the image edge.

    Reconstruction near the image border is unreliable regardless of disparity
    (the convolutional receptive field runs off the image and, for the left
    border specifically, no right-image evidence exists at all).
    """
    batch, _, height, width = like.shape
    mask = torch.ones((batch, 1, height, width), dtype=like.dtype, device=like.device)
    if border > 0:
        mask[..., :border, :] = 0.0
        mask[..., -border:, :] = 0.0
        mask[..., :, :border] = 0.0
        mask[..., :, -border:] = 0.0
    return mask


class PhotometricLoss(nn.Module):
    """SSIM + L1 appearance matching for one warping direction.

    Args:
        alpha: SSIM weight (0.85 in Monodepth).
        border: pixels of image border to exclude.
    """

    def __init__(self, alpha: float = 0.85, border: int = 2):
        super().__init__()
        self.alpha = alpha
        self.border = border

    def residual(self, reconstructed: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-pixel photometric residual and its two components, ``(B, 1, H, W)``."""
        l1 = torch.abs(reconstructed - target).mean(dim=1, keepdim=True)
        ssim_loss = torch.clamp((1.0 - ssim(reconstructed, target)) / 2.0, 0.0, 1.0).mean(dim=1, keepdim=True)
        combined = self.alpha * ssim_loss + (1.0 - self.alpha) * l1
        return {"photometric": combined, "l1": l1, "ssim": ssim_loss}

    def forward(self, target: torch.Tensor, source: torch.Tensor, disparity: torch.Tensor,
                direction: str, extra_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Reconstruct ``target`` from ``source`` and score it.

        Args:
            target: the view being reconstructed, ``(B, 3, H, W)``.
            source: the other view.
            disparity: disparity referenced to ``target``, in pixels.
            direction: ``"left"`` (target is the left view, sample right at
                ``x - d``) or ``"right"`` (target is the right view, sample left
                at ``x + d``).
            extra_mask: optional additional ``(B, 1, H, W)`` reliability mask,
                for example a non-occlusion mask.

        Returns:
            ``loss``, ``l1``, ``ssim`` scalars plus the per-pixel ``residual``,
            the ``reconstruction`` and the ``mask`` that was reduced over.
        """
        if direction == "left":
            reconstructed, valid = warp_right_to_left(source, disparity)
        elif direction == "right":
            reconstructed, valid = warp_left_to_right(source, disparity)
        else:
            raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

        mask = valid * border_mask(target, self.border)
        if extra_mask is not None:
            mask = mask * extra_mask

        parts = self.residual(reconstructed, target)
        return {
            "loss": masked_mean(parts["photometric"], mask),
            "l1": masked_mean(parts["l1"], mask),
            "ssim": masked_mean(parts["ssim"], mask),
            "residual": parts["photometric"],
            "reconstruction": reconstructed,
            "mask": mask,
            "valid_warp": valid,
        }
