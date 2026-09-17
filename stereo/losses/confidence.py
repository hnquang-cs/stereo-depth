"""Label-free supervision for the matchability head.

What the original does
----------------------
Matchability is a *parameter-free* function of the aggregated cost volume (the
negative entropy of ``softmin(cost)``), so it has no weights of its own -- it is
shaped entirely by whatever shapes the cost volume.  In the paper that shaping
comes from the Noise-Sampling Cross-Entropy loss, which pushes the cost curve to
be unimodal **at the ground-truth disparity**.  That supervision is unavailable
here by construction.

What this module does instead
-----------------------------
It replaces the ground-truth peak target with a label-free *reliability* target
``r in {0, 1}`` computed from geometry alone: a pixel is reliable when its two
disparity predictions agree left-to-right, its warp is valid, and its
photometric residual is small.  The confidence ``exp(matchability)`` is then
regressed onto ``r`` with binary cross-entropy.

Honest caveats, restated in the README:
  * This is a new loss, not something the paper does.  Its effect on disparity
    accuracy has not been measured in this repository (no benchmark run was
    possible here), so it is weighted low by default and can be switched off.
  * The target is a function of the model's own outputs, so it is self-referential.
    Two things stop the trivial "confident everywhere" solution: the photometric
    term in ``r`` is not satisfiable by an arbitrary self-consistent disparity
    field in textured regions, and the loss is enabled only after the
    photometric warm-up.  Neither is a proof; a collapse monitor on the mean
    confidence is logged during training so it can be caught.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceLoss(nn.Module):
    """BCE between ``exp(matchability)`` and a detached label-free reliability mask.

    Args:
        eps: clamp keeping the log finite.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, matchability: torch.Tensor, reliability: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
            matchability: ``(B, 1, h, w)`` raw matchability at cost-volume resolution.
            reliability: ``(B, 1, H, W)`` target in ``[0, 1]``; resized to match.
        """
        target = reliability.detach()
        if target.shape[-2:] != matchability.shape[-2:]:
            # Area interpolation gives the fraction of reliable full-resolution
            # pixels inside each cost-volume cell, which is the right soft target.
            target = F.interpolate(target, size=matchability.shape[-2:], mode="area")

        # Computed in float32, outside autocast, for two reasons:
        #   * torch bans binary_cross_entropy under autocast outright (it is
        #     numerically unsafe in half precision and raises a RuntimeError),
        #   * exp() of a float16 matchability loses precision exactly where it
        #     matters, near 0, where confidence approaches 1.
        # This mirrors how the cost volume, soft argmin and matchability layers
        # already force float32 under mixed precision.
        with torch.autocast(device_type=matchability.device.type, enabled=False):
            confidence = torch.exp(matchability.float()).clamp(self.eps, 1.0 - self.eps)
            loss = F.binary_cross_entropy(confidence, target.float().clamp(0.0, 1.0))
        return {
            "loss": loss,
            "mean_confidence": confidence.mean().detach(),
            "mean_reliability": target.float().mean().detach(),
        }
