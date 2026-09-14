"""Cost-volume aggregation: 3D convolutions then 2D convolutions (paper component 3).

The paper's design point is that most of the filtering happens in 2D: a short
3D stack squeezes the feature channel down to a handful, the volume is then
flattened into ``channels_3d * D`` 2D channels, and a dilated 2D residual stack
does the heavy aggregation before emitting one cost map per disparity.

Because the flattened representation makes the channel count proportional to
``D``, the number of disparity levels is baked into the weights: a trained model
has one fixed disparity search range (the original paper likewise picks the
range per dataset).  Spatial resolution stays completely free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import PreactResidualBlock

AGGREGATION_DILATION_RATES = (1, 2, 5, 9)


class Conv3dBlock(nn.Module):
    """Two 3x3x3 convolutions over the (disparity, height, width) volume."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.block(volume)


class CostAggregation(nn.Module):
    """``(B, C, D, H, W)`` correlation volume -> ``(B, D, H, W)`` cost map.

    Args:
        feature_channels: ``C``, the matching-feature dimension (16).
        num_disparities: ``D`` at cost-volume resolution.
        channels_3d: channels kept after the 3D stack (4 in the reference config).
    """

    def __init__(self, feature_channels: int, num_disparities: int, channels_3d: int = 4):
        super().__init__()
        self.num_disparities = num_disparities
        self.conv3d = Conv3dBlock(feature_channels, max(feature_channels // 2, 1), channels_3d)

        channels = channels_3d * num_disparities
        # Channel schedule 4D -> 4D -> 2D -> D -> D with the hybrid dilation rates.
        widths = [channels, channels, channels // 2, channels // 4, num_disparities]
        if widths[3] < num_disparities:
            raise ValueError(
                f"channels_3d={channels_3d} is too small: the 2D stack narrows to "
                f"{widths[3]} channels but needs at least num_disparities={num_disparities}")
        blocks = []
        for idx in range(4):
            blocks.append(PreactResidualBlock(
                widths[idx], widths[idx + 1],
                dilation=AGGREGATION_DILATION_RATES[idx],
                preact=(idx > 0),
                last_norm=(idx == 3),
                leaky=True))
        self.conv2d = nn.Sequential(*blocks)
        self.out = nn.Conv2d(num_disparities, num_disparities, kernel_size=1, bias=True)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        out = self.conv3d(volume)
        out = out.flatten(1, 2)  # (B, channels_3d * D, H, W)
        out = self.conv2d(out)
        return self.out(out)
