"""Estimate the disparity search range a dataset needs -- from images alone.

The search range is baked into the weights, so it has to be chosen *before*
training, and choosing it badly is expensive in both directions: too small
clips the far-field, too large costs accuracy, because every extra candidate is
another chance at a spurious match. Measured on a real Middlebury pair at
640x384 (true disparity 6.2-51.7 px), block-matching error in native pixels:

    cap   48    64    96   128   192   256   320
    MAE 6.42  6.23  6.97  7.44  8.48  9.73 10.83

So the rule ``min(width // 2, 384)`` -- which at width 640 gives 320 -- is a
safe *upper bound* and close to the worst available choice. The range should
come from the disparity the data actually contains.

That can be measured without ground truth. Block matching on the left/right
pair alone gives a per-pixel disparity estimate, and a ratio test discards the
pixels where the match is ambiguous (textureless or repetitive regions, which
would otherwise vote at random and inflate the estimate). The high percentile
of what survives is the range the data needs.

Uses only the rectified left and right images. No ground truth is read, so this
is safe to run on the training split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from ..geometry import RESIZE_ALIGN_CORNERS
from ..losses.cost_volume_loss import MATCH_WINDOW, pooled_photometric_cost_volume

#: A match counts as reliable when the best cost is at least this much better
#: than the best cost outside the peak. Classic ratio test; 0.8 is Lowe's value.
DEFAULT_RATIO = 0.8
#: Bins either side of the peak excluded when looking for the runner-up.
PEAK_EXCLUSION = 2


@dataclass
class DisparityRangeEstimate:
    """What :func:`calibrate_disparity_range` measured."""
    canonical_width: int
    percentile: float
    #: Disparity at ``percentile`` of reliable pixels, in canonical-width pixels.
    disparity_at_percentile: float
    max_reliable_disparity: float
    #: Fraction of pixels that passed the ratio test.
    reliable_fraction: float
    #: Recommended ``num_disparities``: the percentile plus margin, rounded up.
    recommended: int
    pairs_used: int
    #: True if the estimate is pressed against the search ceiling, meaning the
    #: true range may be larger than this and the number is a lower bound.
    saturated: bool

    def __str__(self) -> str:
        note = "  (SATURATED -- widen search_fraction)" if self.saturated else ""
        return (f"disparity range at {self.canonical_width}px width, from {self.pairs_used} pairs:\n"
                f"  p{self.percentile:g} of reliable matches : {self.disparity_at_percentile:.1f} px\n"
                f"  max reliable                : {self.max_reliable_disparity:.1f} px\n"
                f"  reliable pixels             : {self.reliable_fraction * 100:.1f}%\n"
                f"  recommended num_disparities : {self.recommended}{note}")


def _reliable_disparities(left: torch.Tensor, right: torch.Tensor, num_bins: int,
                          downsample: int, window: int, ratio: float):
    """Block-match one pair, return the disparities that pass the ratio test."""
    cost, _ = pooled_photometric_cost_volume(left, right, num_bins, downsample, "left", window)
    best_cost, peak = cost.min(dim=1, keepdim=True)

    bins = torch.arange(cost.shape[1], device=cost.device).view(1, -1, 1, 1)
    away = (bins - peak).abs() > PEAK_EXCLUSION
    # No runner-up exists for a volume this narrow; treat every match as ambiguous.
    if not bool(away.any()):
        return torch.empty(0, device=cost.device), 0.0
    runner_up = cost.masked_fill(~away, float("inf")).min(dim=1, keepdim=True).values

    finite = torch.isfinite(runner_up) & torch.isfinite(best_cost)
    reliable = finite & (best_cost < ratio * runner_up)
    disparities = (peak.float() * downsample)[reliable]
    return disparities, float(reliable.float().mean())


@torch.no_grad()
def calibrate_disparity_range(pairs: Iterable, canonical_width: int = 640,
                              percentile: float = 99.0, margin: float = 1.15,
                              search_fraction: float = 0.5, downsample: int = 4,
                              window: int = MATCH_WINDOW, max_pairs: int = 32,
                              ratio: float = DEFAULT_RATIO,
                              device: Optional[torch.device] = None) -> DisparityRangeEstimate:
    """Measure the disparity range a dataset needs, using no ground truth.

    Args:
        pairs: iterable of ``(left, right)`` tensors ``(B, 3, H, W)`` in [0, 1],
            or of dicts with ``"left"`` and ``"right"`` keys -- a DataLoader
            over the training split works directly.
        canonical_width: width the answer is expressed in. The recommendation is
            the ``num_disparities`` to pair with this ``canonical_width``.
        percentile: percentile of reliable disparities to cover. 99 deliberately
            ignores the last 1%, which is dominated by mismatches.
        margin: safety factor applied to the percentile before rounding.
        search_fraction: how wide to search, as a fraction of width. This is the
            ceiling on what can be discovered, so it is deliberately generous.
        max_pairs: stop after this many batches. The estimate is a high
            percentile over millions of pixels and converges quickly.

    Returns:
        :class:`DisparityRangeEstimate`.
    """
    num_bins = max(int(canonical_width * search_fraction) // downsample, 1)
    ceiling = num_bins * downsample

    collected, reliable_fractions, used = [], [], 0
    for batch in pairs:
        if used >= max_pairs:
            break
        if isinstance(batch, dict):
            left, right = batch["left"], batch["right"]
        else:
            left, right = batch[0], batch[1]
        if device is not None:
            left, right = left.to(device), right.to(device)
        if left.ndim == 3:
            left, right = left[None], right[None]

        height = max(int(round(left.shape[-2] * canonical_width / left.shape[-1])), 64)
        resize = lambda t: F.interpolate(t.float(), size=(height, canonical_width),
                                         mode="bilinear", align_corners=RESIZE_ALIGN_CORNERS)
        disparities, fraction = _reliable_disparities(
            resize(left), resize(right), num_bins, downsample, window, ratio)
        if disparities.numel():
            collected.append(disparities.cpu())
        reliable_fractions.append(fraction)
        used += 1

    if not collected:
        raise ValueError("no reliable matches found -- are the images rectified and non-empty?")

    values = torch.cat(collected)
    at_percentile = float(torch.quantile(values, percentile / 100.0))
    largest = float(values.max())

    recommended = int(at_percentile * margin)
    recommended = max(((recommended + downsample - 1) // downsample) * downsample, downsample)
    return DisparityRangeEstimate(
        canonical_width=canonical_width, percentile=percentile,
        disparity_at_percentile=at_percentile, max_reliable_disparity=largest,
        reliable_fraction=sum(reliable_fractions) / len(reliable_fractions),
        recommended=recommended, pairs_used=used,
        saturated=at_percentile >= ceiling - downsample)
