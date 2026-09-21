"""Label-free shaping of the cost volume.

The paper trains its cost volume with a Noise-Sampling Cross-Entropy loss whose
target is a peak at the **ground-truth** disparity. Removing that -- as this
project must -- leaves the cost volume with only the indirect gradient that
reaches it through soft-argmin, which says "move the expected disparity" rather
than "the match is at index k". Measured on a synthetic scene, that is not
enough: the coarse disparity sat at the midpoint of its search range and moved
1.4 px in 600 steps while the refinement network reduced the photometric loss by
other means. The network reconstructed well and matched nothing.

The fix is that **the true matching cost needs no labels**. For every candidate
disparity you can warp the source view and measure the photometric residual
directly; the resulting volume is a genuine, dense, per-pixel cost. Distilling
the learned cost volume toward it is the label-free analogue of the paper's NSCE
loss -- same cross-entropy form, with the target anchored on photometric
evidence instead of on ground truth.

Nothing here reads ground truth: the target is computed from the two input
images alone.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def photometric_cost_volume(reference: torch.Tensor, source: torch.Tensor,
                            num_disparities: int, direction: str = "left") -> torch.Tensor:
    """Per-disparity photometric residual between the two views.

    ``volume[b, d, y, x]`` is the appearance mismatch when pixel ``x`` of the
    reference view is matched at disparity ``d``. Lower is a better match, so it
    is directly comparable with the network's cost volume.

    Uses mean absolute difference over colour channels. SSIM would need a window
    per disparity and costs far more for little gain at this resolution.

    Args:
        reference: ``(B, 3, H, W)`` view the disparity is referenced to.
        source: ``(B, 3, H, W)`` the other view.
        num_disparities: number of candidates, in pixels of this resolution.
        direction: ``"left"`` matches at ``x - d``, ``"right"`` at ``x + d``.

    Returns:
        ``(B, D, H, W)`` cost, and a ``(B, D, H, W)`` validity mask.
    """
    width = reference.shape[-1]
    costs, valid = [], []
    for disparity in range(num_disparities):
        if disparity == 0:
            shifted = source
            inside = torch.ones_like(source[:, :1])
        elif direction == "left":
            # reference pixel x matches source pixel x - d
            shifted = F.pad(source[..., : width - disparity], (disparity, 0))
            inside = F.pad(torch.ones_like(source[:, :1, :, : width - disparity]), (disparity, 0))
        else:
            shifted = F.pad(source[..., disparity:], (0, disparity))
            inside = F.pad(torch.ones_like(source[:, :1, :, disparity:]), (0, disparity))
        costs.append(torch.abs(reference - shifted).mean(dim=1))
        valid.append(inside[:, 0])
    return torch.stack(costs, dim=1), torch.stack(valid, dim=1)


class CostVolumeLoss(nn.Module):
    """Cross-entropy from the network's cost volume to a photometric target.

    Mirrors the paper's NSCE formulation -- a soft target over disparities,
    cross-entropy against ``log_softmax(-cost)`` -- but builds the target from
    photometric evidence rather than ground truth.

    Args:
        temperature: sharpness of the photometric target. Lower makes it more
            peaked and more confident; too low and photometric noise becomes a
            hard, wrong label.
        min_confidence: skip pixels whose photometric target is nearly uniform.
            In textureless regions every disparity matches equally well and the
            target carries no information, so forcing the cost volume to imitate
            it would inject noise.
    """

    def __init__(self, temperature: float = 0.05, min_confidence: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.min_confidence = min_confidence

    def forward(self, cost: torch.Tensor, reference: torch.Tensor, source: torch.Tensor,
                direction: str = "left") -> Dict[str, torch.Tensor]:
        """Args:
            cost: ``(B, D, h, w)`` the network's aggregated cost volume.
            reference / source: images at the **cost volume's** resolution.
        """
        num_disparities = cost.shape[1]
        with torch.no_grad():
            photometric, valid = photometric_cost_volume(reference, source, num_disparities,
                                                          direction)
            # Softmin over candidates: the distribution photometric evidence implies.
            target = F.softmin(photometric / self.temperature, dim=1)
            target = target * valid
            target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-6)

            # How peaked that target is, as 1 - normalised entropy. Flat means the
            # region is ambiguous (textureless or repetitive) and teaches nothing.
            #
            # Normalised by the number of *valid* candidates, not by D. Near the
            # image border only a few disparities are valid at all, so a target
            # that is uniform over those few still has low absolute entropy and
            # would look confident -- measured at 13.5% of a blank image being
            # "supervised" before this correction.
            entropy = -(target * torch.log(target.clamp(min=1e-8))).sum(dim=1, keepdim=True)
            candidates = valid.sum(dim=1, keepdim=True).clamp(min=2.0)
            confidence = 1.0 - entropy / torch.log(candidates)
            weight = (confidence > self.min_confidence).to(cost.dtype)

        log_probability = F.log_softmax(-cost, dim=1)
        per_pixel = -(target * log_probability).sum(dim=1, keepdim=True)
        total = weight.sum()
        loss = (per_pixel * weight).sum() / total.clamp(min=1.0)
        return {
            "loss": loss,
            "target_confidence": confidence.mean().detach(),
            "supervised_ratio": (total / weight.numel()).detach(),
        }
