"""Diagnostic evaluation of the matchability/confidence head.

GROUND TRUTH IS USED HERE.  Evaluation only.

The paper does not publish a confidence benchmark -- matchability appears only
as the gate of its on-robot post-processing (``confidence = exp(matchability)``,
threshold 0.25) -- so there is no original protocol to reproduce.  These are
therefore clearly-labelled secondary metrics:

``sparsification AUC``
    Remove pixels in order of decreasing confidence and plot the EPE of what
    remains.  Lower area means confidence ranks errors well.  The *oracle* curve
    removes pixels in order of decreasing true error; the difference between the
    two ("sparsification error") is the part that is not explained by the error
    distribution itself.

``roc_auc``
    Probability that a randomly chosen correct pixel is scored above a randomly
    chosen incorrect one, with "correct" defined by a disparity threshold.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


# numpy renamed trapz -> trapezoid in 2.0; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def sparsification_curve(confidence: torch.Tensor, error: torch.Tensor,
                         fractions: Sequence[float] = tuple(np.linspace(0.0, 0.95, 20))) -> Dict[str, list]:
    """EPE of the retained pixels as the least-confident ones are removed."""
    order = torch.argsort(confidence, descending=True)
    sorted_error = error[order]
    sorted_oracle = torch.sort(error, descending=False).values

    total = sorted_error.numel()
    curve, oracle = [], []
    for fraction in fractions:
        keep = max(1, int(round(total * (1.0 - fraction))))
        curve.append(float(sorted_error[:keep].mean()))
        oracle.append(float(sorted_oracle[:keep].mean()))
    return {"fractions": list(map(float, fractions)), "epe": curve, "oracle_epe": oracle}


def confidence_metrics(confidence: torch.Tensor, disparity: torch.Tensor, disparity_gt: torch.Tensor,
                       valid: torch.Tensor, correct_threshold: float = 1.0) -> Dict[str, float]:
    """Sparsification AUC and ROC AUC for one image.  Returns ``{}`` if empty."""
    confidence_flat = confidence.double()[valid]
    error_flat = torch.abs(disparity.double() - disparity_gt.double())[valid]
    if confidence_flat.numel() < 2:
        return {}

    curve = sparsification_curve(confidence_flat, error_flat)
    auc = float(_trapezoid(curve["epe"], curve["fractions"]))
    oracle_auc = float(_trapezoid(curve["oracle_epe"], curve["fractions"]))

    correct = (error_flat <= correct_threshold)
    num_correct = int(correct.sum())
    num_wrong = int(correct.numel() - num_correct)
    if num_correct == 0 or num_wrong == 0:
        roc_auc = float("nan")
    else:
        # Rank-based (Mann-Whitney) AUC, ties averaged.
        ranks = torch.argsort(torch.argsort(confidence_flat)).double() + 1.0
        roc_auc = float((ranks[correct].sum() - num_correct * (num_correct + 1) / 2.0)
                        / (num_correct * num_wrong))

    return {
        "num_valid": float(confidence_flat.numel()),
        "sparsification_auc": auc,
        "oracle_sparsification_auc": oracle_auc,
        "sparsification_error_auc": auc - oracle_auc,
        "roc_auc": roc_auc,
        "mean_confidence": float(confidence_flat.mean()),
    }
