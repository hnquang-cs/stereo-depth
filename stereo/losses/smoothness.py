"""Edge-aware disparity smoothness.

Formulation (Monodepth, Godard et al. 2017, eq. 4)::

    L_smooth = mean( |d_x d~| * exp(-|d_x I|) + |d_y d~| * exp(-|d_y I|) )

where the image gradient magnitude is averaged over colour channels.

Disparity normalisation
-----------------------
``d~`` is the *mean-normalised* disparity ``d / (mean(d) + eps)`` rather than the
raw disparity.  Without it the network can shrink the whole disparity map toward
zero to reduce the regulariser -- a real degenerate solution for self-supervised
stereo, where the photometric term is weak in textureless regions.  This is the
normalisation used by Monodepth2 (Wang et al., "Excessive Invariance"); the
original Monodepth instead relies on disparity being bounded in ``[0, 1]``,
which does not apply here because this network predicts disparity in pixels.

Deviation from the local `monodepth` reference: that implementation omits the
absolute values on both the disparity and the image gradients, which makes the
penalty signed -- it can be driven arbitrarily negative by a disparity map that
decreases monotonically to the right.  The published formulation is used here.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_x(tensor: torch.Tensor) -> torch.Tensor:
    """Forward difference along x, replicate-padded to keep the shape."""
    padded = F.pad(tensor, (0, 1, 0, 0), mode="replicate")
    return padded[..., :, 1:] - padded[..., :, :-1]


def gradient_y(tensor: torch.Tensor) -> torch.Tensor:
    """Forward difference along y, replicate-padded to keep the shape."""
    padded = F.pad(tensor, (0, 0, 0, 1), mode="replicate")
    return padded[..., 1:, :] - padded[..., :-1, :]


class SmoothnessLoss(nn.Module):
    """Edge-aware first-order smoothness on mean-normalised disparity."""

    def __init__(self, normalize: bool = True, eps: float = 1e-7):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, disparity: torch.Tensor, image: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            disparity: ``(B, 1, H, W)``.
            image: ``(B, 3, H, W)`` guidance image at the *same* resolution
                (resize it before calling; the caller knows the right mode).
            mask: optional ``(B, 1, H, W)`` weighting.
        """
        if disparity.shape[-2:] != image.shape[-2:]:
            raise ValueError(f"disparity {tuple(disparity.shape)} and image {tuple(image.shape)} "
                             "must have the same resolution")
        if self.normalize:
            mean_disparity = disparity.mean(dim=[1, 2, 3], keepdim=True)
            disparity = disparity / (mean_disparity + self.eps)

        disparity_dx = torch.abs(gradient_x(disparity))
        disparity_dy = torch.abs(gradient_y(disparity))
        image_dx = torch.mean(torch.abs(gradient_x(image)), dim=1, keepdim=True)
        image_dy = torch.mean(torch.abs(gradient_y(image)), dim=1, keepdim=True)

        smoothness = disparity_dx * torch.exp(-image_dx) + disparity_dy * torch.exp(-image_dy)
        if mask is None:
            return smoothness.mean()
        return (smoothness * mask).sum() / (mask.sum() + 1e-6)
