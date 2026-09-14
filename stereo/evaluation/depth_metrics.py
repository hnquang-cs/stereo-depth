"""Metric-depth evaluation against real depth ground truth (LiDAR, structured light).

GROUND TRUTH IS USED HERE.  Evaluation only.

The paper reports disparity metrics, not depth metrics, so everything in this
module is **secondary/diagnostic** and is labelled as such in the benchmark
output.  The definitions are the standard ones (Eigen et al., 2014):

    AbsRel  = mean(|z - z*| / z*)
    SqRel   = mean((z - z*)^2 / z*)
    RMSE    = sqrt(mean((z - z*)^2))
    RMSElog = sqrt(mean((log z - log z*)^2))
    delta_k = fraction with max(z/z*, z*/z) < 1.25^k

No scale alignment
------------------
A calibrated stereo network predicts *metric* disparity, so median scaling or
least-squares scale fitting against ground truth would be measuring something
other than the model.  :func:`depth_metrics` therefore has no scaling step.
:func:`median_scale_factor` exists so that a protocol which genuinely requires
alignment can apply it explicitly and record that it did; it is never called by
default.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

DELTA_THRESHOLDS = (1.25, 1.25 ** 2, 1.25 ** 3)


def depth_valid_mask(depth_gt: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 80.0,
                     gt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Valid-pixel mask for depth evaluation.

    Sparse ground truth (projected LiDAR) is *not* interpolated to create more
    evaluation points: only the measured pixels selected by ``gt_mask`` count.
    """
    valid = torch.isfinite(depth_gt) & (depth_gt > min_depth) & (depth_gt < max_depth)
    if gt_mask is not None:
        valid = valid & (gt_mask > 0.5)
    return valid


def median_scale_factor(depth: torch.Tensor, depth_gt: torch.Tensor, valid: torch.Tensor) -> float:
    """``median(z*) / median(z)``.

    Provided for protocols that mandate scale alignment.  Applying it to a
    calibrated stereo prediction inflates the result and must be reported.
    """
    predicted = depth[valid]
    target = depth_gt[valid]
    if predicted.numel() == 0:
        return 1.0
    return float(torch.median(target) / torch.clamp(torch.median(predicted), min=1e-9))


def depth_metrics(depth: torch.Tensor, depth_gt: torch.Tensor, valid: torch.Tensor,
                  scale_factor: float = 1.0) -> Dict[str, float]:
    """Standard depth error metrics over ``valid``.  Returns ``{}`` if empty.

    Args:
        scale_factor: multiplied into the prediction.  Leave at 1.0 unless the
            protocol explicitly calls for alignment; the benchmark output records
            whichever value was used.
    """
    predicted = depth.double()[valid] * scale_factor
    target = depth_gt.double()[valid]
    if predicted.numel() == 0:
        return {}

    predicted = torch.clamp(predicted, min=1e-6)
    target = torch.clamp(target, min=1e-6)

    difference = predicted - target
    abs_rel = float((torch.abs(difference) / target).mean())
    sq_rel = float(((difference ** 2) / target).mean())
    rmse = float(torch.sqrt((difference ** 2).mean()))
    rmse_log = float(torch.sqrt(((torch.log(predicted) - torch.log(target)) ** 2).mean()))
    mae = float(torch.abs(difference).mean())

    ratio = torch.maximum(predicted / target, target / predicted)
    metrics = {
        "num_valid": float(predicted.numel()),
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "mae": mae,
        "scale_factor": float(scale_factor),
    }
    for index, threshold in enumerate(DELTA_THRESHOLDS, start=1):
        metrics[f"delta_{index}"] = float((ratio < threshold).double().mean())
    return metrics


class DepthAccumulator:
    """Running mean of per-image depth metrics, weighted by valid-pixel count."""

    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.total_pixels = 0.0
        self.num_images = 0

    def update(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        weight = metrics["num_valid"]
        self.total_pixels += weight
        self.num_images += 1
        for key, value in metrics.items():
            if key in ("num_valid", "scale_factor"):
                continue
            self.sums[key] = self.sums.get(key, 0.0) + value * weight

    def compute(self) -> Dict[str, float]:
        if self.total_pixels == 0:
            return {"num_images": float(self.num_images), "num_valid_pixels": 0.0}
        results = {key: value / self.total_pixels for key, value in self.sums.items()}
        results["num_images"] = float(self.num_images)
        results["num_valid_pixels"] = self.total_pixels
        return results
