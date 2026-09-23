"""Block matching: photometric cost volumes built from the two images alone.

These are *measurement* utilities, not a loss. Nothing here is trained and
nothing here is part of either paper's objective -- they exist so that the
disparity search range can be calibrated from images without ground truth
(:mod:`stereo.data.calibration`) and so that block matching is available as a
baseline to measure a trained model against.

They read the rectified left and right images only. No ground truth.

History: these grew out of a cost-volume loss that has since been removed. That
loss was a label-free stand-in for the paper's NSCE term, which cannot be used
here because NSCE is anchored on ground-truth disparity. Substituting an invented
loss for it was a deviation from "reimplement the paper", so the loss is gone;
the block-matching utilities it was built on are kept because they are useful on
their own and carry no such baggage.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


#: Side of the square window the matching cost is aggregated over. A single
#: pixel's absolute difference matches equally well at dozens of disparities, so
#: a per-pixel cost carries almost no information: measured on a real Middlebury
#: pair, a 1x1 cost scored MAE 13.28 px against a true disparity of 2.2-18.1 --
#: worse than predicting a constant (4.52 px). A 9x9 window scores 1.78 px.
#: Every classical stereo matcher aggregates over a window; this is that.
MATCH_WINDOW = 9


def box_filter(values: torch.Tensor, window: int) -> torch.Tensor:
    """Mean over a ``window x window`` neighbourhood, keeping the resolution."""
    if window <= 1:
        return values
    pad = window // 2
    return F.avg_pool2d(F.pad(values, (pad, pad, pad, pad), mode="replicate"),
                        window, stride=1)


def photometric_cost_volume(reference: torch.Tensor, source: torch.Tensor,
                            num_disparities: int, direction: str = "left",
                            window: int = MATCH_WINDOW) -> torch.Tensor:
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
        residual = torch.abs(reference - shifted).mean(dim=1, keepdim=True)
        costs.append(box_filter(residual, window)[:, 0])
        valid.append(inside[:, 0])
    return torch.stack(costs, dim=1), torch.stack(valid, dim=1)


def pooled_photometric_cost_volume(reference: torch.Tensor, source: torch.Tensor,
                                   num_disparities: int, scale: int,
                                   direction: str = "left",
                                   window: int = MATCH_WINDOW):
    """Photometric cost at the cost volume's resolution, built from FULL-resolution
    evidence.

    Matching on downsampled images does not work: downsampling destroys the very
    texture that makes correspondence possible. Measured on a test scene, the
    argmin of a cost volume computed on 1/4-resolution images was off by 4.56 px
    against a true disparity of 1.5-4.5 px -- an error as large as the signal --
    while the same construction at full resolution was off by 1.24 px. Training
    the network toward the former plateaued immediately at a wrong answer.

    So the cost is evaluated at full resolution for every full-resolution
    disparity, then pooled: cell ``d`` of the output averages the full-resolution
    costs for disparities ``[d * scale, (d + 1) * scale)`` and averages spatially
    over ``scale x scale`` blocks. Only one full-resolution slab exists at a time,
    so the memory cost is that of a single image, not of the whole volume.

    Args:
        reference / source: images at FULL resolution.
        num_disparities: candidates at the cost volume's resolution.
        scale: the cost volume's downsample factor.

    Returns:
        ``(cost, valid)``, both ``(B, num_disparities, H // scale, W // scale)``.
    """
    width = reference.shape[-1]
    costs, valids = [], []
    for coarse in range(num_disparities):
        accumulated, accumulated_valid = None, None
        for offset in range(scale):
            disparity = coarse * scale + offset
            # Clamp: a disparity at or beyond the image width would make
            # ``width - disparity`` negative, and a negative slice bound wraps
            # around instead of giving an empty slice -- silently producing a
            # wider tensor than the reference. Nothing is visible at such a
            # disparity anyway, so the whole column is invalid.
            offset = min(disparity, width)
            if offset == 0:
                shifted = source
                inside = torch.ones_like(source[:, :1])
            elif direction == "left":
                shifted = F.pad(source[..., : width - offset], (offset, 0))
                inside = F.pad(torch.ones_like(source[:, :1, :, : width - offset]), (offset, 0))
            else:
                shifted = F.pad(source[..., offset:], (0, offset))
                inside = F.pad(torch.ones_like(source[:, :1, :, offset:]), (0, offset))
            # Aggregate over a window BEFORE pooling: the window is what makes
            # the cost discriminative, and pooling alone gives only a 4x4
            # non-overlapping support, which is not enough.
            residual = box_filter(torch.abs(reference - shifted).mean(dim=1, keepdim=True), window)
            accumulated = residual if accumulated is None else accumulated + residual
            accumulated_valid = inside if accumulated_valid is None else accumulated_valid + inside
        costs.append(F.avg_pool2d(accumulated / scale, scale)[:, 0])
        valids.append((F.avg_pool2d(accumulated_valid / scale, scale)[:, 0] > 0.5).to(reference.dtype))
    return torch.stack(costs, dim=1), torch.stack(valids, dim=1)


def _fit_to_pooled_size(reference: torch.Tensor, source: torch.Tensor,
                        cost_size, scale: int):
    """Pad or crop both views so that pooling by ``scale`` lands exactly on ``cost_size``.

    Padding replicates the edge and is applied on the right and bottom only, so
    no pixel's x coordinate moves and disparities stay valid.
    """
    target_height, target_width = cost_size[0] * scale, cost_size[1] * scale
    height, width = reference.shape[-2:]
    if (height, width) == (target_height, target_width):
        return reference, source
    if height > target_height or width > target_width:
        reference = reference[..., :target_height, :target_width]
        source = source[..., :target_height, :target_width]
        height, width = reference.shape[-2:]
    pad = (0, max(0, target_width - width), 0, max(0, target_height - height))
    if any(pad):
        reference = F.pad(reference, pad, mode="replicate")
        source = F.pad(source, pad, mode="replicate")
    return reference, source
