"""Pre-activation residual blocks and transitions used by both dilated ResNets.

These are the building blocks of the paper's feature extractor and refinement
network.  Compared with the reference implementation the class hierarchy is
flattened: there is one residual block with a ``leaky`` switch instead of two
near-duplicate classes, and the group builders are plain functions.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _activation(leaky: bool) -> nn.Module:
    return nn.LeakyReLU(inplace=True, negative_slope=0.1) if leaky else nn.ReLU(inplace=True)


class PreactResidualBlock(nn.Module):
    """Pre-activation basic residual block with optional dilation.

    ``out = shortcut(x) + conv3x3(act(bn(conv3x3(act(bn(x))))))``

    Args:
        in_channels: input channel count.
        out_channels: output channel count.
        stride: stride of the first convolution (1 or 2).
        dilation: dilation rate of both convolutions.
        preact: apply BN+activation to the input before the first convolution.
        last_norm: apply BN+activation to the block output.
        leaky: use LeakyReLU(0.1) instead of ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1,
                 preact: bool = True, last_norm: bool = False, leaky: bool = False):
        super().__init__()
        self.preact_bn = nn.BatchNorm2d(in_channels) if preact else None
        # Pre-activation shortcut: a bare 1x1 convolution, no normalisation.
        self.shortcut = (None if (stride == 1 and in_channels == out_channels)
                         else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn_last = nn.BatchNorm2d(out_channels) if last_norm else None
        self.act = _activation(leaky)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = inputs if self.shortcut is None else self.shortcut(inputs)
        if self.preact_bn is not None:
            inputs = self.act(self.preact_bn(inputs))
        out = self.act(self.bn1(self.conv1(inputs)))
        out = self.conv2(out)
        out = out + shortcut
        if self.bn_last is not None:
            out = self.act(self.bn_last(out))
        return out


class TransitionBlock(nn.Module):
    """Change resolution and/or channel count: conv -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        if stride == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        elif stride == 2:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        self.block = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


def residual_group(in_channels: int, out_channels: int, num_blocks: int,
                   dilation_rates: Sequence[int] = (1,), leaky: bool = False) -> nn.Sequential:
    """A group of pre-activation residual blocks.

    ``dilation_rates`` is cycled over the blocks, which is the Hybrid Dilated
    Convolution scheme (https://arxiv.org/abs/1702.08502) used by the paper's
    deepest group.  Passing ``(1,)`` gives a plain undilated group.

    The first block has no pre-activation (the preceding transition block
    already ends in BN+ReLU) and the last block ends with BN+activation.
    """
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1")
    rates = list(dilation_rates)
    blocks = [PreactResidualBlock(in_channels, out_channels, dilation=rates[0],
                                  preact=False, last_norm=(num_blocks == 1), leaky=leaky)]
    for idx in range(1, num_blocks):
        blocks.append(PreactResidualBlock(out_channels, out_channels, dilation=rates[idx % len(rates)],
                                          preact=True, last_norm=(idx == num_blocks - 1), leaky=leaky))
    return nn.Sequential(*blocks)
