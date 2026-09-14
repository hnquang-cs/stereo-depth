"""Matchability-based post-processing, kept strictly out of the network.

The paper: "Output depth pixels are considered valid when they pass a confidence
check, and the containing 'depth-region' is sufficiently large (2000 px or
larger)", where ``confidence = exp(matchability)`` and the threshold is 0.25.

Two steps, in order:

1. **Confidence gate** -- drop pixels whose ``exp(matchability)`` is below
   ``confidence_threshold`` (0.25 in the paper).
2. **Region-size gate** -- group the surviving pixels into connected components
   of similar disparity and drop components smaller than ``min_region_pixels``
   (2000 in the paper).

The paper does not state how similar two neighbouring disparities must be to
belong to the same region, so ``region_disparity_tolerance`` is an explicit
assumption of this implementation, documented and configurable, defaulting to
1 pixel.

The paper's benchmark tables use the **raw** network output, so post-processing
is off by default and reported as a separate row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PostProcessConfig:
    enabled: bool = False
    #: exp(matchability) threshold; 0.25 in the paper.
    confidence_threshold: float = 0.25
    #: Minimum connected-component size in full-resolution pixels; 2000 in the paper.
    min_region_pixels: int = 2000
    #: Maximum disparity step within one region. Not specified by the paper.
    region_disparity_tolerance: float = 1.0
    #: Value written into invalidated pixels.
    invalid_value: float = 0.0


def upsample_confidence(confidence: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bring cost-volume-resolution confidence up to full resolution."""
    if confidence.shape[-2:] == tuple(size):
        return confidence
    return F.interpolate(confidence, size=size, mode="bilinear", align_corners=False)


def _region_labels(disparity: np.ndarray, mask: np.ndarray, tolerance: float) -> np.ndarray:
    """Label 4-connected components of ``mask`` whose neighbours differ by < ``tolerance``.

    Two-pass union-find over the right and down neighbours.  Plain NumPy so the
    behaviour is obvious and it has no OpenCV version dependency.
    """
    height, width = disparity.shape
    parent = np.arange(height * width, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    flat_index = np.arange(height * width).reshape(height, width)
    right_link = mask[:, :-1] & mask[:, 1:] & (np.abs(disparity[:, :-1] - disparity[:, 1:]) < tolerance)
    for row, column in zip(*np.nonzero(right_link)):
        union(int(flat_index[row, column]), int(flat_index[row, column + 1]))
    down_link = mask[:-1, :] & mask[1:, :] & (np.abs(disparity[:-1, :] - disparity[1:, :]) < tolerance)
    for row, column in zip(*np.nonzero(down_link)):
        union(int(flat_index[row, column]), int(flat_index[row + 1, column]))

    labels = np.array([find(i) for i in range(height * width)], dtype=np.int64).reshape(height, width)
    return np.where(mask, labels, -1)


def postprocess_disparity(disparity: torch.Tensor, confidence: torch.Tensor,
                          config: PostProcessConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the confidence and region-size gates.

    Args:
        disparity: ``(B, 1, H, W)`` full-resolution disparity.
        confidence: ``(B, 1, h, w)`` ``exp(matchability)``; upsampled internally.
        config: thresholds.

    Returns:
        ``(disparity, valid)`` where invalidated pixels hold
        ``config.invalid_value`` and ``valid`` is a ``(B, 1, H, W)`` float mask.
    """
    if not config.enabled:
        return disparity, torch.ones_like(disparity)

    confidence = upsample_confidence(confidence, disparity.shape[-2:])
    valid = (confidence >= config.confidence_threshold)

    if config.min_region_pixels > 1:
        disparity_np = disparity.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        for batch_index in range(disparity_np.shape[0]):
            plane = disparity_np[batch_index, 0]
            mask = valid_np[batch_index, 0]
            if not mask.any():
                continue
            labels = _region_labels(plane, mask, config.region_disparity_tolerance)
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            small = set(unique[counts < config.min_region_pixels].tolist())
            if small:
                drop = np.isin(labels, list(small))
                valid_np[batch_index, 0] = mask & ~drop
        valid = torch.from_numpy(valid_np).to(disparity.device)

    valid = valid.to(disparity.dtype)
    filtered = torch.where(valid > 0.5, disparity, torch.full_like(disparity, config.invalid_value))
    return filtered, valid
