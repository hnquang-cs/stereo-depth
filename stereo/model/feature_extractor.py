"""Dilated-ResNet feature extractor (paper component 1).

Produces the 16-dimensional feature map that the cost volume is built from,
downsampled from the input by 4 (low-resolution variant) or 8 (high-resolution
variant).

Structure, following the paper's "dilated ResNet":

    image ---> /2 ---> /2 ---> group1 (4x)  ------------.
                               |                        |
                               /2 ---> group2 (8x)  ----+--> score head --> 16-d
                                        |               |
                                        /2 ---> group3 (16x, hybrid dilated)

The score head merges the three scales top-down with 1x1 convolutions and
bilinear x2 upsampling, stopping at the requested output scale.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .blocks import TransitionBlock, residual_group

HDC_DILATION_RATES = (1, 2, 5, 9)
GROUP_BLOCKS = (3, 4, 8)


class DilatedResNetBackbone(nn.Module):
    """Three-stage dilated ResNet returning features at 1/4, 1/8 and 1/16."""

    def __init__(self, in_channels: int = 3, width: int = 16):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        self.stem = nn.Sequential(TransitionBlock(in_channels, c1, stride=2),
                                  TransitionBlock(c1, c1, stride=2))
        self.group1 = residual_group(c1, c1, GROUP_BLOCKS[0])
        self.down2 = TransitionBlock(c1, c2, stride=2)
        self.group2 = residual_group(c2, c2, GROUP_BLOCKS[1])
        self.down3 = TransitionBlock(c2, c3, stride=2)
        self.group3 = residual_group(c3, c3, GROUP_BLOCKS[2], dilation_rates=HDC_DILATION_RATES)
        self.out_channels = (c1, c2, c3)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_4x = self.group1(self.stem(images))
        feat_8x = self.group2(self.down2(feat_4x))
        feat_16x = self.group3(self.down3(feat_8x))
        return feat_4x, feat_8x, feat_16x


class ScoreHead(nn.Module):
    """Merge the backbone scales top-down into a single feature map.

    Args:
        backbone_channels: ``(c_4x, c_8x, c_16x)``.
        out_channels: matching-feature dimension (16 in the paper).
        out_scale: 4, 8 or 16 -- the resolution the cost volume is built at.
    """

    def __init__(self, backbone_channels: Tuple[int, int, int], out_channels: int, out_scale: int):
        super().__init__()
        if out_scale not in (4, 8, 16):
            raise ValueError(f"out_scale must be 4, 8 or 16, got {out_scale}")
        c1, c2, c3 = backbone_channels
        self.out_scale = out_scale
        self.score_16x = nn.Conv2d(c3, out_channels, kernel_size=1, bias=True)
        self.score_8x = nn.Conv2d(c2, out_channels, kernel_size=1, bias=True) if out_scale <= 8 else None
        self.score_4x = nn.Conv2d(c1, out_channels, kernel_size=1, bias=True) if out_scale <= 4 else None
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, feat_4x: torch.Tensor, feat_8x: torch.Tensor, feat_16x: torch.Tensor) -> torch.Tensor:
        out = self.score_16x(feat_16x)
        if self.score_8x is not None:
            out = self.score_8x(feat_8x) + self.upsample(out)
        if self.score_4x is not None:
            out = self.score_4x(feat_4x) + self.upsample(out)
        return out


class FeatureExtractor(nn.Module):
    """Backbone + score head: image -> ``feature_channels``-d map at ``1/out_scale``."""

    def __init__(self, feature_channels: int = 16, backbone_width: int = 16, out_scale: int = 4):
        super().__init__()
        self.backbone = DilatedResNetBackbone(in_channels=3, width=backbone_width)
        self.score = ScoreHead(self.backbone.out_channels, feature_channels, out_scale)
        self.out_scale = out_scale
        self.feature_channels = feature_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.score(*self.backbone(images))
