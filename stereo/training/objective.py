"""The training objective: the paper's supervised loss.

:class:`~stereo.losses.PaperObjective` holds the loss itself. This module adapts
it to the trainer's batch shape -- pulling the labels out, choosing the reference
view, and reporting the per-term breakdown the log prints.

The self-supervised objective that used to live here (photometric
reconstruction, left-right consistency) has been removed. It is recorded in
docs/REPORT.md: measured against ground truth it plateaued around 10 px on real
pairs, where the supervised loss reaches 0.67 px on the same data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..config import LossWeights
from ..losses import PaperLossWeights, PaperObjective


@dataclass
class ObjectiveState:
    """Schedule values for one iteration."""
    iteration: int = 0
    epoch: int = 0
    #: Ramps terms in over the first iterations; 1.0 once past the warm-up.
    warmup_scale: float = 1.0


class SupervisedObjective:
    """The paper's loss, applied to the trainer's batches.

    Args:
        weights: see :class:`~stereo.config.LossWeights`.
        downsample: the cost volume's scale factor, which the loss needs to put
            the ground truth into the coarse map's pixel units.
    """

    def __init__(self, weights: LossWeights, downsample: int = 4):
        self.weights = weights
        self.downsample = downsample
        self.loss = PaperObjective(PaperLossWeights(
            disparity=weights.disparity, nsce=weights.nsce,
            smoothness=weights.smoothness, stdmean_scaled=weights.stdmean_scaled))

    def __call__(self, outputs: Dict[str, Dict[str, torch.Tensor]],
                 images: Dict[str, torch.Tensor],
                 state: ObjectiveState,
                 labels: Optional[Dict[str, torch.Tensor]] = None,
                 valid_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Args:
            outputs: ``{"left": {...}}`` from the model.
            images: ``{"left", "right"}`` -- the clean (un-jittered) pair.
            labels: ``{"disparity_gt", "valid_gt_mask"}``. Required: this
                objective is supervised and has nothing to optimise without them.
            valid_mask: 1 on real pixels, 0 on rows the collate padded onto a
                ragged batch.
        """
        if labels is None or "disparity_gt" not in labels:
            raise ValueError(
                "supervised training needs disparity labels, and this batch has none. "
                "Check that the dataset provides ground truth and that it was built "
                "with with_labels=True.")

        target = labels["disparity_gt"]
        valid = labels.get("valid_gt_mask")
        valid = torch.ones_like(target) if valid is None else valid
        if valid_mask is not None:
            valid = valid * valid_mask

        result = self.loss(outputs["left"], images["left"], target, valid, self.downsample)

        with torch.no_grad():
            disparity = outputs["left"]["disparity"]
            result["logs"].update({
                "disparity_mean": float(disparity.mean()),
                "disparity_min": float(disparity.min()),
                "disparity_max": float(disparity.max()),
                "labelled_ratio": float(valid.mean()),
            })
        return result


#: The previous name, kept so older checkpoints and configs still import.
LabelFreeObjective = SupervisedObjective
