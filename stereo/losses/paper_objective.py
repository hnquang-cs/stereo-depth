"""The paper's supervised objective, as `mmstereo/config_sceneflow.yaml` sets it.

    L = sum over the two output scales of
          100 * disparity   (smooth-L1, std-mean scaled, per sample)
        + 0.2 * NSCE        (cost-volume scale only)
        +  20 * smoothness  (cost-volume scale only)

Three details are easy to get wrong and are taken from the reference
implementation rather than from the text:

1. **Deep supervision.** The disparity loss is applied to BOTH outputs -- the
   full-resolution refined disparity and the coarse soft-argmin disparity -- not
   only the final one.
2. **Coarse ground truth is max-pooled**, then divided by the scale factor, so it
   lands in the coarse map's own pixel units. Max, not mean: at a depth
   discontinuity a 4x4 block takes the nearest surface rather than inventing a
   disparity between the two.
3. **Smoothness and NSCE apply only at the coarse scale.** The reference returns
   a null loss for smoothness when ``scale == 1``, and NSCE needs a cost volume,
   which only the coarse output carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .smoothness import SmoothnessLoss
from .supervised import NsceLoss

#: A frame with less labelled area than this contributes nothing. Sparse ground
#: truth makes the per-sample std/mean below meaningless, and the reference skips
#: such frames outright.
MIN_VALID_FRACTION = 0.03


def downsample_disparity(disparity: torch.Tensor, factor: int) -> torch.Tensor:
    """Ground truth at ``1/factor`` resolution, in that resolution's pixel units.

    Max-pooled rather than averaged: averaging across a depth discontinuity
    produces a disparity that is true on neither side of it, while the max takes
    the nearer surface.
    """
    if factor == 1:
        return disparity
    return F.max_pool2d(disparity, kernel_size=factor, stride=factor) / factor


@dataclass
class PaperLossWeights:
    """`config_sceneflow.yaml` verbatim."""
    disparity: float = 100.0
    nsce: float = 0.2
    smoothness: float = 20.0
    #: Divide each sample's disparity loss by ``mean + 2*std`` of its own ground
    #: truth. Without it a frame full of large disparities dominates a batch
    #: simply by being closer to the camera.
    stdmean_scaled: bool = True


class PaperObjective(nn.Module):
    """The supervised objective, evaluated at every output scale."""

    def __init__(self, weights: Optional[PaperLossWeights] = None):
        super().__init__()
        self.weights = weights or PaperLossWeights()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.smoothness = SmoothnessLoss(normalize=False)
        # The paper's 0.2 weight is tuned for the RAW cross-entropy, and the
        # reference excludes disparity 0 (cost_volume[:, 1:]).
        self.nsce = NsceLoss(normalize=False, skip_zero_bin=True)

    def disparity_loss(self, disparity: torch.Tensor, target: torch.Tensor,
                       valid: torch.Tensor) -> torch.Tensor:
        """Smooth-L1, per sample, scaled by that sample's own ground truth spread."""
        per_pixel = self.smooth_l1(disparity, target)
        threshold = MIN_VALID_FRACTION * disparity.shape[-1] * disparity.shape[-2]
        total = disparity.sum() * 0.0
        counted = 0
        for index in range(disparity.shape[0]):
            keep = valid[index] > 0.5
            if float(keep.sum()) < threshold:
                continue                       # too little ground truth to scale by
            values = per_pixel[index][keep]
            if self.weights.stdmean_scaled:
                truth = target[index][keep]
                std, mean = torch.std_mean(truth)
                total = total + values.mean() / (mean + 2.0 * std)
            else:
                total = total + values.mean()
            counted += 1
        # Divided by the batch size, not by the number counted, so a batch where
        # most frames lack labels contributes proportionally less.
        return total / max(disparity.shape[0], 1), counted

    def forward(self, outputs: Dict[str, torch.Tensor], image: torch.Tensor,
                disparity_gt: torch.Tensor, valid_gt: torch.Tensor,
                downsample: int) -> Dict[str, object]:
        """Args:
            outputs: one view's model output -- ``disparity``, ``disparity_small``
                and ``cost``.
            image: the reference view, for the smoothness guidance.
            disparity_gt, valid_gt: ``(B, 1, H, W)`` ground truth at full resolution.
            downsample: the cost volume's scale factor.
        """
        logs: Dict[str, float] = {}
        parts: Dict[str, float] = {}
        total = outputs["disparity"].sum() * 0.0

        # -- full resolution: disparity only -------------------------------- #
        loss, counted = self.disparity_loss(outputs["disparity"], disparity_gt, valid_gt)
        contribution = self.weights.disparity * loss
        total = total + contribution
        parts["disp_1x"] = float(contribution.detach())
        logs["labelled_frames"] = counted / max(outputs["disparity"].shape[0], 1)
        with torch.no_grad():
            keep = valid_gt > 0.5
            logs["epe"] = float((outputs["disparity"] - disparity_gt).abs()[keep].mean()) \
                if bool(keep.any()) else float("nan")

        # -- cost-volume scale: disparity, smoothness, NSCE ------------------ #
        small_gt = downsample_disparity(disparity_gt, downsample)
        small_valid = downsample_disparity(valid_gt, downsample) > 0.0
        small = outputs["disparity_small"]

        loss, _ = self.disparity_loss(small, small_gt, small_valid.to(small.dtype))
        contribution = self.weights.disparity * loss
        total = total + contribution
        parts[f"disp_{downsample}x"] = float(contribution.detach())

        if self.weights.smoothness > 0.0:
            guidance = F.interpolate(image, size=small.shape[-2:], mode="bilinear",
                                     align_corners=False)
            max_small = max(small.shape[-1], 1)
            smooth = self.smoothness(small / max_small, guidance)
            contribution = self.weights.smoothness * smooth
            total = total + contribution
            parts["smooth"] = float(contribution.detach())

        if self.weights.nsce > 0.0 and "cost" in outputs:
            terms = self.nsce(outputs["cost"], disparity_gt, valid_gt.to(small.dtype), downsample)
            contribution = self.weights.nsce * terms["loss"]
            total = total + contribution
            parts["nsce"] = float(contribution.detach())
            logs["nsce_in_range"] = float(terms["in_range_ratio"])

        logs["total"] = float(total.detach())
        logs.update({f"part/{name}": value for name, value in parts.items()})
        return {"loss": total, "logs": logs, "parts": parts}
