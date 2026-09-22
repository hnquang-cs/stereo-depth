"""Second dilated ResNet: full-resolution disparity refinement (paper component 5).

Per the paper the refinement network "calculates a disparity residual given the
original input image, low resolution disparity, and matchability".  It is an
encoder/decoder dilated ResNet over the reference image, into which the
low-resolution disparity and matchability are injected at 1/4 resolution, with
the bilinearly upsampled low-resolution disparity carried through as the base
prediction so the network only has to learn the residual.

Simplification versus the reference implementation: the reference also threads
the raw cost volume into the refinement input block but then never uses it
(the argument is ignored inside the block).  It is dropped here, which also
matches the paper's own description of the inputs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .blocks import PreactResidualBlock, TransitionBlock, residual_group
from .feature_extractor import GROUP_BLOCKS, HDC_DILATION_RATES

OFFSET_CHANNELS = 8


class DisparityInputBlock(nn.Module):
    """Encode (disparity, matchability) at cost-volume scale and resample to 1/4."""

    def __init__(self, out_channels: int, in_scale: int):
        super().__init__()
        self.resize = nn.Upsample(scale_factor=in_scale / 4.0, mode="bilinear", align_corners=True)
        self.encode = nn.Sequential(
            PreactResidualBlock(2, out_channels // 2, preact=False),
            PreactResidualBlock(out_channels // 2, out_channels, preact=True, last_norm=True),
        )
        self.refine = nn.Sequential(
            PreactResidualBlock(out_channels, out_channels, preact=False),
            PreactResidualBlock(out_channels, out_channels, preact=True, last_norm=True),
        )

    def forward(self, disparity: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
        out = self.encode(torch.cat([disparity, match], dim=1))
        return self.refine(self.resize(out))


class UpsampleBlock(nn.Module):
    """Bilinear x2 upsample, 3x3 convolution, optional skip addition, BN + LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, features: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.conv(self.upsample(features))
        if skip is not None:
            out = out + skip
        return self.act(self.norm(out))


class DisparityRefinement(nn.Module):
    """Low-resolution disparity + reference image -> full-resolution disparity.

    Args:
        in_scale: resolution of the incoming disparity (4, 8 or 16).
        channels: ``(c1, c2, c3)`` widths of the three encoder stages.
    """

    def __init__(self, in_scale: int, channels=(32, 64, 128),
                 residual_limit: Optional[float] = None):
        """Args:
            residual_limit: if set, the refinement residual is squashed into
                ``[-limit, +limit]`` with a tanh. The base disparity is already
                bounded by the cost volume -- only the residual is free, and an
                unbounded residual lets the head emit disparities the cost volume
                cannot support, which are pure extrapolation. ``None`` keeps the
                reference implementation's unbounded ``relu(base + residual)``.
        """
        super().__init__()
        if in_scale not in (4, 8, 16):
            raise ValueError(f"in_scale must be 4, 8 or 16, got {in_scale}")
        c1, c2, c3 = channels
        self.in_scale = in_scale
        self.residual_limit = residual_limit

        self.disparity_input = DisparityInputBlock(c1, in_scale)
        self.merge = nn.Sequential(
            PreactResidualBlock(c1 * 2, c1 * 2, preact=False),
            PreactResidualBlock(c1 * 2, c1, preact=True, last_norm=True),
        )

        self.stem = nn.Sequential(TransitionBlock(3, c1, stride=2), TransitionBlock(c1, c1, stride=2))
        self.group1 = residual_group(c1, c1, GROUP_BLOCKS[0])
        self.down2 = TransitionBlock(c1, c2, stride=2)
        self.group2 = residual_group(c2, c2, GROUP_BLOCKS[1])
        self.down3 = TransitionBlock(c2, c3, stride=2)
        self.group3 = residual_group(c3, c3, GROUP_BLOCKS[2], dilation_rates=HDC_DILATION_RATES)

        # Decoder skip projections. The 1/16 branch leaves one channel free for the
        # downsampled base disparity, so the decoder always "sees" the coarse geometry.
        self.skip1 = nn.Conv2d(c1, OFFSET_CHANNELS * 2, kernel_size=3, padding=1)
        self.skip2 = nn.Conv2d(c2, OFFSET_CHANNELS * 4, kernel_size=3, padding=1)
        self.skip3 = nn.Conv2d(c3, OFFSET_CHANNELS * 8 - 1, kernel_size=3, padding=1)

        self.up_8x = UpsampleBlock(OFFSET_CHANNELS * 8, OFFSET_CHANNELS * 4)
        self.up_4x = UpsampleBlock(OFFSET_CHANNELS * 4, OFFSET_CHANNELS * 2)
        self.up_2x = UpsampleBlock(OFFSET_CHANNELS * 2, OFFSET_CHANNELS - 1)
        self.up_1x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.base_upsample = nn.Upsample(scale_factor=in_scale, mode="bilinear", align_corners=False)
        # Pool the low-resolution disparity down to 1/16 and convert to 1/16-pixel units.
        self.base_pool_factor = 16 // in_scale
        self.base_pool = nn.MaxPool2d(kernel_size=self.base_pool_factor) if self.base_pool_factor > 1 else nn.Identity()

        self.out = nn.Conv2d(OFFSET_CHANNELS, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def zero_init_residual(self) -> None:
        """Start the refinement as an exact identity on the coarse disparity.

        ``self.out`` takes ``base_disparity`` itself as one of its input channels
        and its result is added back to ``base_disparity``, so under a generic
        (Kaiming) init the head computes ``(1 + w) * base_disparity`` for a
        random ``w``: measured over five seeds, the refined output came out
        between 0.73x and 1.34x the coarse disparity *before any training*.

        The reference implementation tolerates this because it trains the
        refined output against ground-truth disparity, which pins the scale down
        immediately. Label-free, the only full-resolution signal is the
        photometric residual -- too weak and too non-convex to undo a global
        rescaling, so the error persists (measured: refined MAE 71.5 px against
        a coarse MAE of 17.4 px after 200 steps).

        Zeroing this one layer makes the initial residual exactly zero, so
        training starts from ``refined == coarse`` and the head can only earn
        its way away from that. Parameter count and architecture are unchanged.
        """
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, image: torch.Tensor, disparity: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
        """Args:
            image: ``(B, 3, H, W)`` reference image (the view the disparity belongs to).
            disparity: ``(B, 1, H/s, W/s)`` disparity in ``1/s``-resolution pixels.
            match: ``(B, 1, H/s, W/s)`` matchability.

        Returns:
            ``(B, 1, H, W)`` non-negative disparity in full-resolution pixels.
        """
        # Upsampling a 1/s-pixel disparity to full resolution also scales its value by s.
        base_disparity = self.base_upsample(disparity) * self.in_scale

        disparity_features = self.disparity_input(disparity, match)

        out = self.stem(image)
        out = torch.cat([out, disparity_features], dim=1)
        out = self.merge(out)

        out = self.group1(out)
        skip1 = self.skip1(out)
        out = self.group2(self.down2(out))
        skip2 = self.skip2(out)
        out = self.group3(self.down3(out))
        skip3 = self.skip3(out)

        coarse_disparity = self.base_pool(disparity) / self.base_pool_factor
        out = torch.cat([coarse_disparity, skip3], dim=1)
        out = self.up_8x(out, skip2)
        out = self.up_4x(out, skip1)
        out = self.up_2x(out, None)

        residual = self.out(torch.cat([base_disparity, self.up_1x(out)], dim=1))
        if self.residual_limit is not None:
            # Saturating rather than clipped, so gradients survive at the bound.
            residual = self.residual_limit * torch.tanh(residual / self.residual_limit)
        return self.relu(base_disparity + residual)
