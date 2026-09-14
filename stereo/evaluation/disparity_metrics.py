"""Disparity metrics, defined to match the original protocols exactly.

GROUND TRUTH IS USED HERE.  Nothing in this module is reachable from the
training loop; it is imported only by :mod:`stereo.evaluation`.

Where each definition comes from
--------------------------------
``epe``, ``bad_0.25``, ``bad_0.5``, ``bad_1.0``
    The reference implementation (``mmstereo/metrics``).  ``DisparityError``
    accumulates ``sum |d - d_gt|`` and ``sum valid`` across the whole run and
    divides at the end, so its EPE is a **global per-pixel mean**, not a mean of
    per-image means.  ``DisparityCorrect(t)`` counts ``|d - d_gt| <= t``; the
    paper's "% Bad (1.0)" is ``100 * (1 - correct_1.0)``, i.e. the fraction with
    ``|d - d_gt| > 1.0``.

``bad_2.0``, ``bad_4.0``, ``avgerr``, ``rms``, ``A90``, ``A95``
    The Middlebury 2014 / MiddEval3 evaluation SDK (``evaldisp``), which is what
    the paper's Table V reports.  ``bad_t`` is ``err > t``; ``avgerr`` is the
    mean absolute error; ``rms`` is the root mean square error; ``A90``/``A95``
    are the 90th and 95th percentiles of the absolute error.  Middlebury scores
    each image and then averages, so these are computed **per image**.

``d1``
    The KITTI convention: a pixel is an outlier when
    ``|d - d_gt| > 3`` **and** ``|d - d_gt| / d_gt > 0.05``.

Both aggregations are always produced and always labelled, because mixing them
silently is one of the easiest ways to publish a wrong comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

#: Thresholds the reference implementation logs.
REFERENCE_BAD_THRESHOLDS: Sequence[float] = (0.25, 0.5, 1.0)
#: Thresholds the Middlebury leaderboard reports.
MIDDLEBURY_BAD_THRESHOLDS: Sequence[float] = (0.5, 1.0, 2.0, 4.0)

KITTI_D1_ABSOLUTE = 3.0
KITTI_D1_RELATIVE = 0.05


def disparity_valid_mask(disparity_gt: torch.Tensor,
                         max_disparity: Optional[float] = None,
                         min_disparity: float = 1e-3,
                         gt_mask: Optional[torch.Tensor] = None,
                         ignore_edge: bool = False) -> torch.Tensor:
    """Valid-pixel mask, following the reference implementation.

    ``mmstereo/utils.get_disparity_valid_mask`` keeps pixels with
    ``d_gt > 1e-3`` and ``d_gt < max_disparity``, optionally also dropping
    pixels whose match falls outside the image (``d_gt >= x - 1``).

    Args:
        disparity_gt: ``(B, 1, H, W)`` ground-truth disparity.
        max_disparity: upper bound; ``None`` disables the check.
        min_disparity: lower bound (exclusive).
        gt_mask: dataset-provided validity (sparse ground truth, occlusion masks).
        ignore_edge: also require ``d_gt < x - 1``.

    Returns:
        bool tensor of the same shape.
    """
    valid = disparity_gt > min_disparity
    if max_disparity is not None:
        valid = valid & (disparity_gt < max_disparity)
    if gt_mask is not None:
        valid = valid & (gt_mask > 0.5)
    if ignore_edge:
        width = disparity_gt.shape[-1]
        columns = torch.arange(width, device=disparity_gt.device, dtype=disparity_gt.dtype) - 1.0
        valid = valid & (disparity_gt < columns.view(1, 1, 1, width))
    return valid


def _errors(disparity: torch.Tensor, disparity_gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Flat float64 vector of absolute disparity errors over the valid pixels."""
    return torch.abs(disparity.double() - disparity_gt.double())[valid]


def per_image_disparity_metrics(disparity: torch.Tensor, disparity_gt: torch.Tensor,
                                valid: torch.Tensor,
                                bad_thresholds: Sequence[float] = MIDDLEBURY_BAD_THRESHOLDS,
                                quantiles: Sequence[float] = (0.90, 0.95)) -> Dict[str, float]:
    """Metrics for a single image.  Returns ``{}`` when no pixel is valid."""
    errors = _errors(disparity, disparity_gt, valid)
    count = errors.numel()
    if count == 0:
        return {}

    metrics: Dict[str, float] = {
        "num_valid": float(count),
        "epe": float(errors.mean()),
        "avgerr": float(errors.mean()),
        "rms": float(torch.sqrt((errors ** 2).mean())),
    }
    for threshold in bad_thresholds:
        metrics[f"bad_{threshold:g}"] = float((errors > threshold).double().mean() * 100.0)
    for quantile in quantiles:
        metrics[f"A{int(round(quantile * 100))}"] = float(torch.quantile(errors, quantile))

    gt_values = disparity_gt.double()[valid]
    relative = errors / torch.clamp(gt_values, min=1e-6)
    d1 = ((errors > KITTI_D1_ABSOLUTE) & (relative > KITTI_D1_RELATIVE)).double().mean() * 100.0
    metrics["d1"] = float(d1)
    return metrics


@dataclass
class DisparityAccumulator:
    """Accumulates both aggregation conventions over a benchmark run.

    ``global_*``  -- one mean over every valid pixel of every image, which is
                     what the reference implementation's torchmetrics do.
    ``image_*``   -- the mean of the per-image metrics, which is what the
                     Middlebury and KITTI evaluations do.
    """
    bad_thresholds: Sequence[float] = field(default_factory=lambda: tuple(
        sorted(set(REFERENCE_BAD_THRESHOLDS) | set(MIDDLEBURY_BAD_THRESHOLDS))))
    quantiles: Sequence[float] = (0.90, 0.95)

    error_sum: float = 0.0
    squared_error_sum: float = 0.0
    pixel_count: int = 0
    bad_counts: Dict[float, float] = field(default_factory=dict)
    d1_count: float = 0.0
    per_image: List[Dict[str, float]] = field(default_factory=list)

    def update(self, disparity: torch.Tensor, disparity_gt: torch.Tensor, valid: torch.Tensor,
               sample_id: Optional[str] = None) -> Dict[str, float]:
        image_metrics = per_image_disparity_metrics(disparity, disparity_gt, valid,
                                                    self.bad_thresholds, self.quantiles)
        if not image_metrics:
            return {}
        if sample_id is not None:
            image_metrics = {"sample_id": sample_id, **image_metrics}
        self.per_image.append(image_metrics)

        errors = _errors(disparity, disparity_gt, valid)
        self.error_sum += float(errors.sum())
        self.squared_error_sum += float((errors ** 2).sum())
        self.pixel_count += int(errors.numel())
        for threshold in self.bad_thresholds:
            self.bad_counts[threshold] = self.bad_counts.get(threshold, 0.0) + float((errors > threshold).sum())

        gt_values = disparity_gt.double()[valid]
        relative = errors / torch.clamp(gt_values, min=1e-6)
        self.d1_count += float(((errors > KITTI_D1_ABSOLUTE) & (relative > KITTI_D1_RELATIVE)).sum())
        return image_metrics

    def compute(self) -> Dict[str, float]:
        if self.pixel_count == 0:
            return {"num_images": float(len(self.per_image)), "num_valid_pixels": 0.0}

        results: Dict[str, float] = {
            "num_images": float(len(self.per_image)),
            "num_valid_pixels": float(self.pixel_count),
            "global_epe": self.error_sum / self.pixel_count,
            "global_rms": float(np.sqrt(self.squared_error_sum / self.pixel_count)),
            "global_d1": 100.0 * self.d1_count / self.pixel_count,
        }
        for threshold, count in self.bad_counts.items():
            results[f"global_bad_{threshold:g}"] = 100.0 * count / self.pixel_count
            results[f"global_correct_{threshold:g}"] = 100.0 * (1.0 - count / self.pixel_count)

        keys = {key for image in self.per_image for key in image if key != "sample_id"}
        for key in sorted(keys):
            values = [image[key] for image in self.per_image if key in image]
            if values and key != "num_valid":
                results[f"image_{key}"] = float(np.mean(values))
        return results
