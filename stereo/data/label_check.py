"""Are a dataset's disparity labels consistent with its images?

A mirror that resizes stereo images must rescale disparity by the same factor,
because disparity is a length in pixels. Mirrors that resize the images and copy
the disparity through unchanged exist, and the failure is silent: supervised
training fits labels that are wrong by a constant factor, which looks like the
model failing rather than the data being broken.

The check needs no reference. For correctly scaled labels, shifting the right
view by the labelled disparity reconstructs the left view; for labels wrong by a
factor, it does not. Scanning candidate factors and reporting which reconstructs
best gives the factor the labels appear to be off by -- 1.0 when they are right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch

from ..geometry import warp_right_to_left

#: Factors scanned. Powers of two either side of 1, plus the ratios a resize to a
#: square from a 4:3 or 16:9 source typically produces.
DEFAULT_FACTORS = (0.125, 0.25, 0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)


@dataclass
class LabelScaleReport:
    """What :func:`check_label_scale` measured."""
    best_factor: float
    errors: Dict[float, float]
    pairs_used: int

    @property
    def consistent(self) -> bool:
        """True when the labels reconstruct best at their own scale."""
        return self.best_factor == 1.0

    def __str__(self) -> str:
        ordered = sorted(self.errors.items())
        detail = "  ".join(f"x{factor:g}:{error:.4f}" for factor, error in ordered)
        if self.consistent:
            return (f"disparity labels are consistent with the images "
                    f"({self.pairs_used} pairs)\n    {detail}")
        return (f"WARNING: disparity labels look wrong by a factor of about "
                f"{self.best_factor:g}.\n"
                f"    Shifting the right view by label x{self.best_factor:g} reconstructs the left "
                f"better than\n    by the label itself, which means this mirror resized its images "
                f"without\n    rescaling disparity. Supervised training would fit labels that are "
                f"wrong.\n    ({self.pairs_used} pairs)  {detail}")


@torch.no_grad()
def check_label_scale(pairs: Iterable, factors: Sequence[float] = DEFAULT_FACTORS,
                      max_pairs: int = 8) -> LabelScaleReport:
    """Measure which scaling of the labels best reconstructs the left view.

    Args:
        pairs: iterable of dicts with ``left``, ``right`` and ``disparity_gt``
            (a benchmark-mode dataset or loader works directly).
        factors: candidate scalings.
        max_pairs: stop after this many; the answer is a factor of two or more
            apart from its neighbours, so a handful of pairs settles it.
    """
    totals = {factor: 0.0 for factor in factors}
    counts = {factor: 0 for factor in factors}
    used = 0
    for sample in pairs:
        if used >= max_pairs:
            break
        left, right = sample["left"], sample["right"]
        disparity = sample.get("disparity_gt")
        if disparity is None:
            continue
        if left.ndim == 3:
            left, right, disparity = left[None], right[None], disparity[None]
        valid_gt = sample.get("valid_gt_mask")
        if valid_gt is not None and valid_gt.ndim == 3:
            valid_gt = valid_gt[None]

        for factor in factors:
            warped, valid = warp_right_to_left(right, disparity * factor)
            if valid_gt is not None:
                valid = valid * valid_gt
            error = (warped - left).abs().mean(dim=1, keepdim=True)
            keep = valid > 0.5
            if bool(keep.any()):
                totals[factor] += float(error[keep].mean())
                counts[factor] += 1
        used += 1

    errors = {factor: (totals[factor] / counts[factor] if counts[factor] else float("inf"))
              for factor in factors}
    if not any(counts.values()):
        raise ValueError("no labelled pairs with a valid warp region were found")
    return LabelScaleReport(best_factor=min(errors, key=errors.get), errors=errors, pairs_used=used)
