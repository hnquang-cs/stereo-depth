"""Label-free shaping of the cost volume.

The paper trains its cost volume with a Noise-Sampling Cross-Entropy loss whose
target is a peak at the **ground-truth** disparity. Removing that -- as this
project must -- leaves the cost volume with only the indirect gradient that
reaches it through soft-argmin, which says "move the expected disparity" rather
than "the match is at index k". Measured on a synthetic scene, that is not
enough: the coarse disparity sat at the midpoint of its search range and moved
1.4 px in 600 steps while the refinement network reduced the photometric loss by
other means. The network reconstructed well and matched nothing.

The fix is that **the true matching cost needs no labels**. For every candidate
disparity you can warp the source view and measure the photometric residual
directly; the resulting volume is a genuine, dense, per-pixel cost. Distilling
the learned cost volume toward it is the label-free analogue of the paper's NSCE
loss -- same cross-entropy form, with the target anchored on photometric
evidence instead of on ground truth.

Nothing here reads ground truth: the target is computed from the two input
images alone.
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


class CostVolumeLoss(nn.Module):
    """Cross-entropy from the network's cost volume to a photometric target.

    Mirrors the paper's NSCE formulation -- a soft target over disparities,
    cross-entropy against ``log_softmax(-cost)`` -- but builds the target from
    photometric evidence rather than ground truth.

    The target must be judged by its **expectation**, not its argmin, because
    soft-argmin takes the expectation. A target can have a near-perfect argmin
    and still be useless: measured on a test scene, a target whose argmin was
    0.78 px from the truth had an expectation of 9.14 against a true disparity of
    2.49, because a long tail across the other candidates dragged the mean up.
    Training fitted that target faithfully and produced exactly that wrong
    disparity.

    The cost is therefore standardised per pixel before the softmin -- shifted by
    the best candidate's cost and divided by the spread across candidates -- so
    ``temperature`` means the same thing regardless of image contrast, exposure
    or noise, none of which a fixed absolute temperature survives.

    Args:
        temperature: sharpness of the target, in units of the per-pixel cost
            spread. Lower is more peaked; too low and photometric noise becomes
            a hard, wrong label.
        min_confidence: skip pixels whose target is nearly uniform. In textureless
            regions every disparity matches equally well and the target carries no
            information, so imitating it would inject noise.
    """

    def __init__(self, temperature: float = 0.08, min_confidence: float = 0.05,
                 window: int = MATCH_WINDOW):
        super().__init__()
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.window = window

    def forward(self, cost: torch.Tensor, reference: torch.Tensor, source: torch.Tensor,
                direction: str = "left",
                valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Args:
            cost: ``(B, D, h, w)`` the network's aggregated cost volume.
            reference / source: images at **full** resolution. Matching is done
                there and pooled down; see :func:`pooled_photometric_cost_volume`
                for why matching on downsampled images fails.
        """
        num_disparities = cost.shape[1]
        scale = max(1, round(reference.shape[-1] / cost.shape[-1]))
        with torch.no_grad():
            if scale > 1:
                # The model pads its input to a multiple of its stride and crops
                # the output, so the cost volume's size is not always
                # floor(H / scale). Fit the images to exactly cost_size * scale
                # first, or the pooled target comes out a row or column short.
                reference, source = _fit_to_pooled_size(reference, source,
                                                        cost.shape[-2:], scale)
                photometric, valid = pooled_photometric_cost_volume(
                    reference, source, num_disparities, scale, direction, self.window)
            else:
                photometric, valid = photometric_cost_volume(
                    reference, source, num_disparities, direction, self.window)
            # Standardise per pixel so the temperature is scale-free: subtract the
            # best candidate's cost and divide by the spread across candidates.
            # Invalid candidates are pushed far up so they cannot win.
            masked = photometric + (1.0 - valid) * 1e3
            best = masked.min(dim=1, keepdim=True).values
            spread = masked.masked_fill(valid < 0.5, float("nan"))
            spread = (torch.nanmean((spread - best).abs(), dim=1, keepdim=True)
                      .nan_to_num(1.0).clamp(min=1e-6))
            standardised = (masked - best) / spread

            # Softmin over candidates: the distribution photometric evidence implies.
            target = F.softmin(standardised / self.temperature, dim=1)
            target = target * valid
            target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-6)

            # How peaked that target is, as 1 - normalised entropy. Flat means the
            # region is ambiguous (textureless or repetitive) and teaches nothing.
            #
            # Normalised by the number of *valid* candidates, not by D. Near the
            # image border only a few disparities are valid at all, so a target
            # that is uniform over those few still has low absolute entropy and
            # would look confident -- measured at 13.5% of a blank image being
            # "supervised" before this correction.
            entropy = -(target * torch.log(target.clamp(min=1e-8))).sum(dim=1, keepdim=True)
            candidates = valid.sum(dim=1, keepdim=True).clamp(min=2.0)
            confidence = 1.0 - entropy / torch.log(candidates)
            weight = (confidence > self.min_confidence).to(cost.dtype)

            # Pixels near the left border cannot be evaluated at every disparity,
            # so their target is computed over a truncated candidate set and is
            # biased low regardless of the evidence. Supervising them teaches the
            # cost volume a ramp. Require the full search to have been available.
            weight = weight * (candidates >= num_disparities).to(cost.dtype)

            # Rows the collate padded onto a ragged batch are replicated pixels:
            # they match each other perfectly at every disparity, so they would
            # otherwise contribute a confident, meaningless target.
            if valid_mask is not None:
                weight = weight * valid_mask.to(cost.dtype)

        log_probability = F.log_softmax(-cost, dim=1)
        per_pixel = -(target * log_probability).sum(dim=1, keepdim=True)
        total = weight.sum()
        loss = (per_pixel * weight).sum() / total.clamp(min=1.0)
        return {
            "loss": loss,
            "target_confidence": confidence.mean().detach(),
            "supervised_ratio": (total / weight.numel()).detach(),
        }
