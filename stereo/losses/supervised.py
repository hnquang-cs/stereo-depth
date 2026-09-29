"""Supervised losses. These are the only terms that read ground truth.

Both are masked means over labelled pixels, so a batch mixing labelled and
unlabelled samples needs no branching: an unlabelled sample contributes an
all-zero mask and therefore nothing.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Width of the Laplacian NSCE places on the true disparity, in disparity bins.
#: 0.3 is the reference implementation's LAMBDA.
NSCE_LAMBDA = 0.3


def labels_from_batch(batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    """Pull the supervised targets out of a batch, or ``None`` if it has none.

    The ground-truth key names live here rather than in the training loop, so the
    loop never mentions them and the leakage audit can keep treating the whole of
    ``stereo/training/loop.py`` as a zone where labels must not appear.
    """
    if "disparity_gt" not in batch:
        return None
    return {"disparity_gt": batch["disparity_gt"],
            "valid_gt_mask": batch.get("valid_gt_mask")}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over ``mask``, or 0 when nothing is masked in."""
    total = mask.sum()
    if float(total) == 0.0:
        return values.sum() * 0.0
    return (values * mask).sum() / total


class DisparityLoss(nn.Module):
    """Smooth L1 between the predicted and true disparity.

    Args:
        beta: the smooth-L1 transition point, in pixels.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, disparity: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        error = F.smooth_l1_loss(disparity, target, beta=self.beta, reduction="none")
        return {"loss": masked_mean(error, mask),
                "epe": masked_mean((disparity - target).abs().detach(), mask)}


class NsceLoss(nn.Module):
    """Noise-Sampling Cross Entropy on the cost volume (arXiv:2005.08806, eq. 12).

    Cross-entropy from the cost volume's ``softmin`` distribution to a sharp
    Laplacian centred on the true disparity, which pushes the volume toward a
    single peak in the right place. Without it the volume is shaped only by the
    gradient that survives the soft-argmin, which can say "move the expected
    disparity" but never "candidate k is the correct one".

    This is the paper's term, and it needs ground truth -- that is why the
    label-free configuration has no equivalent and simply omits it.

    The cost volume is at ``1/downsample`` resolution and its bins are in
    ``downsample``-pixel units, so the target disparity is divided by
    ``downsample`` and the ground truth resampled to match.
    """

    def __init__(self, lambda_: float = NSCE_LAMBDA):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, cost: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                downsample: int) -> Dict[str, torch.Tensor]:
        size = cost.shape[-2:]
        # Nearest, not bilinear: averaging disparity across a depth discontinuity
        # invents a value that is true nowhere, and the same for the mask.
        small_target = F.interpolate(target, size=size, mode="nearest") / downsample
        small_mask = F.interpolate(mask, size=size, mode="nearest")

        candidates = torch.arange(cost.shape[1], dtype=cost.dtype,
                                  device=cost.device).view(1, -1, 1, 1)
        laplacian = torch.softmax(-(candidates - small_target).abs() / self.lambda_, dim=1)
        log_probability = F.log_softmax(-cost, dim=1)
        cross_entropy = -(laplacian * log_probability).sum(dim=1, keepdim=True)

        # Only where the true disparity is inside the search range; outside it the
        # Laplacian is truncated and the target is a fiction.
        in_range = (small_target < cost.shape[1]).to(cost.dtype)
        return {"loss": masked_mean(cross_entropy, small_mask * in_range),
                "in_range_ratio": (small_mask * in_range).sum() / small_mask.sum().clamp(min=1.0)}
