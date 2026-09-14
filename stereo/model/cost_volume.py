"""Cross-correlation cost volume, soft-argmin and matchability (paper components 2 and 4).

Indexing convention (derived in :mod:`stereo.geometry`)
------------------------------------------------------
The volume is always built in the **reference-view / left-referenced** sense: the
reference pixel ``x`` is matched against the source pixel ``x - d``.  Concretely

    volume[b, c, d, y, x] = ref[b, c, y, x] * src[b, c, y, x - d]      (x >= d)
                          = 0                                          (x <  d)

which is exactly the reference implementation's ``is_right=False`` branch.

The *opposite* direction does not need a second indexing scheme.  Writing
``flip(a)[x] = a[W-1-x]`` and substituting ``x' = W-1-x``::

    corr(flip(src), flip(ref))[d, x] = src[W-1-x] * ref[W-1-x+d]
                                     = src[x'] * ref[x' + d]

i.e. the source pixel ``x'`` matched against the reference pixel ``x' + d`` --
the right-referenced volume, mirrored.  So feeding the mirrored feature maps
through the *same* weights yields the reverse-direction cost volume with
identical border statistics, and a final horizontal flip puts it back in place.
:class:`stereo.model.stereo_net.StereoNet` uses this, which is why only one
cost-volume function exists here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def correlation_volume(reference: torch.Tensor, source: torch.Tensor, num_disparities: int) -> torch.Tensor:
    """Element-wise cross-correlation cost volume.

    The comparison is vector valued (the feature channel is kept, as in the
    paper's Table I "vector-valued comparison function"), so the result is
    ``(B, C, D, H, W)`` and the following 3D convolutions reduce the channel
    dimension.

    Args:
        reference: ``(B, C, H, W)`` features of the view whose disparity is predicted.
        source: ``(B, C, H, W)`` features of the other view.
        num_disparities: number of disparity levels ``D`` at this resolution.

    Returns:
        ``(B, C, D, H, W)`` volume, zero where the shift falls off the image.
    """
    if reference.shape != source.shape:
        raise ValueError(f"feature shape mismatch: {tuple(reference.shape)} vs {tuple(source.shape)}")
    width = reference.shape[-1]
    if num_disparities < 1:
        raise ValueError("num_disparities must be >= 1")

    levels = []
    for disparity in range(num_disparities):
        if disparity == 0:
            product = reference * source
        else:
            shift = min(disparity, width)
            product = reference[..., shift:] * source[..., : width - shift]
            # Pad on the left: reference pixels with x < d have no partner.
            product = F.pad(product, (shift, 0))
        levels.append(product)
    return torch.stack(levels, dim=2)


class CorrelationCostVolume(nn.Module):
    """:func:`correlation_volume` as a module, forced to float32.

    The products can overflow in float16, so the volume is computed outside
    autocast and clamped -- matching the reference implementation's behaviour
    under mixed precision.
    """

    def __init__(self, num_disparities: int, clamp: float = 1e3):
        super().__init__()
        self.num_disparities = num_disparities
        self.clamp = clamp

    def forward(self, reference: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=reference.device.type, enabled=False):
            volume = correlation_volume(reference.float(), source.float(), self.num_disparities)
            return torch.clamp(volume, -self.clamp, self.clamp)


def soft_argmin(cost: torch.Tensor) -> torch.Tensor:
    """Differentiable disparity regression from a cost volume.

    ``cost`` is ``(B, D, H, W)`` and is a *cost* (lower is better), so the
    distribution over disparities is ``softmin`` and the estimate is

        d(x) = sum_k k * softmin(cost)_k(x)

    Returns ``(B, 1, H, W)`` disparity in pixels **of this resolution**.
    """
    num_disparities = cost.shape[1]
    probability = F.softmin(cost, dim=1)
    indices = torch.arange(num_disparities, dtype=probability.dtype,
                           device=probability.device).view(1, num_disparities, 1, 1)
    return torch.sum(probability * indices, dim=1, keepdim=True)


def matchability(cost: torch.Tensor) -> torch.Tensor:
    """Matchability = negative entropy of the disparity distribution.

    From https://arxiv.org/abs/2008.04800, as used by the paper::

        m(x) = sum_k p_k(x) * log p_k(x),   p = softmin(cost)

    so ``m`` lies in ``[-log D, 0]`` and the paper's confidence is
    ``exp(m) in [1/D, 1]``: 1 for a perfectly peaked cost curve, 1/D for a flat
    one.  Returns ``(B, 1, H, W)``.
    """
    probability = F.softmin(cost, dim=1)
    log_probability = F.log_softmax(-cost, dim=1)  # == log(softmin(cost)), computed stably
    return torch.sum(probability * log_probability, dim=1, keepdim=True)


def confidence_from_matchability(match: torch.Tensor) -> torch.Tensor:
    """``confidence = exp(matchability)``, the paper's post-processing quantity."""
    return torch.exp(match)


class SoftArgmin(nn.Module):
    """:func:`soft_argmin` forced to float32 (unstable in float16)."""

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=cost.device.type, enabled=False):
            return soft_argmin(cost.float())


class Matchability(nn.Module):
    """:func:`matchability` forced to float32 (unstable in float16)."""

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=cost.device.type, enabled=False):
            return matchability(cost.float())
