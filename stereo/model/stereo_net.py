"""The stereo network of "A Learned Stereo Depth System for Robotic Manipulation in Homes".

    left  --> feature extractor --.
                                   >-- cross-correlation cost volume
    right --> feature extractor --'              |
                                                 v
                                     3D convs -> flatten -> 2D convs
                                                 |
                                      .----------+----------.
                                      v                     v
                                 soft argmin           matchability
                                      |                     |
                                      '-----> refinement <--'  (+ reference image)
                                                 |
                                        full-resolution disparity

Arbitrary input sizes are supported: the inputs are padded on the right/bottom
to a multiple of :attr:`StereoNet.size_divisor` and the outputs are cropped
back.  Padding right/bottom keeps every pixel's x coordinate, so disparity
values are untouched by it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from ..geometry import compute_num_disparities, flip_lr, pad_to_multiple, unpad
from .aggregation import CostAggregation
from .cost_volume import CorrelationCostVolume, Matchability, SoftArgmin, confidence_from_matchability
from .feature_extractor import FeatureExtractor
from .refinement import DisparityRefinement


@dataclass
class StereoNetConfig:
    """Architecture hyper-parameters.

    ``num_disparities`` is a *construction-time* parameter: the aggregation
    stack flattens the disparity axis into channels, so the search range is part
    of the weights.  Use :meth:`StereoNetConfig.for_width` to derive it from an
    image width with the ``min(width // 2, 384)`` policy.
    """
    num_disparities: int = 256
    downsample: int = 4
    feature_channels: int = 16
    backbone_width: int = 16
    cost_volume_channels: int = 4
    max_disparities_cap: int = 384
    #: Bound on the refinement residual, as a fraction of the search range.
    #: ``None`` reproduces the reference implementation's unbounded head.
    residual_limit_fraction: Optional[float] = None
    #: Width at which ``num_disparities`` is expressed. Disparity scales linearly
    #: with horizontal resize, so ``num_disparities / canonical_width`` -- not
    #: ``num_disparities`` -- is the resize-invariant description of the search
    #: range. Inference on an image of any other width resizes to this width and
    #: scales the result back (:func:`stereo.model.predict_disparity`).
    canonical_width: int = 640
    #: Restrict the soft-argmin expectation to ``+/- window`` bins around the cost
    #: minimum. ``None``, the reference implementation's full expectation, is the
    #: default **because the restriction did not replicate end to end**: it is a
    #: large win on the photometric cost volume (9.14 -> 6.84 px) but on a trained
    #: network's own volume every read-out scores the same, and the full
    #: expectation scored best of all (19.18 vs 20.26-20.42). See
    #: :func:`stereo.model.cost_volume.soft_argmin` and docs/REPORT.md 16f.
    soft_argmin_window: Optional[int] = None

    @classmethod
    def for_width(cls, width: int, downsample: int = 4, max_disparities_cap: int = 384, **kwargs) -> "StereoNetConfig":
        """Build a config whose search range follows ``min(width // 2, 384)``."""
        return cls(num_disparities=compute_num_disparities(width, downsample, max_disparities_cap),
                   downsample=downsample, max_disparities_cap=max_disparities_cap, **kwargs)

    def __post_init__(self):
        if self.downsample not in (4, 8, 16):
            raise ValueError(f"downsample must be 4, 8 or 16, got {self.downsample}")
        if self.num_disparities % self.downsample != 0:
            raise ValueError(f"num_disparities={self.num_disparities} must be a multiple of "
                             f"downsample={self.downsample}")

    def to_dict(self) -> dict:
        return asdict(self)


class StereoNet(nn.Module):
    """Full stereo matching network.

    Output keys (per direction):
        ``disparity``        ``(B, 1, H, W)`` full-resolution disparity, in pixels.
        ``disparity_small``  ``(B, 1, H/s, W/s)`` disparity in 1/s-resolution pixels.
        ``matchability``     ``(B, 1, H/s, W/s)`` negative entropy of the cost curve.
        ``confidence``       ``exp(matchability)``, in ``[1/D, 1]``.
        ``cost``             ``(B, D, H/s, W/s)`` aggregated cost volume.
    """

    size_divisor = 16  # deepest stride in both the feature extractor and the refinement net

    def __init__(self, config: Optional[StereoNetConfig] = None):
        super().__init__()
        self.config = config or StereoNetConfig()
        scale = self.config.downsample
        self.scale = scale
        self.num_disparities = self.config.num_disparities
        self.num_disparities_small = self.config.num_disparities // scale

        # Largest disparity the cost volume can express, in the units of each map.
        # Matches the reference implementation's definition, which the evaluation
        # protocol uses to build its valid-pixel mask.
        self.max_disparity_small = self.num_disparities_small - 1
        self.max_disparity = self.max_disparity_small * scale - 1

        self.feature_extractor = FeatureExtractor(
            feature_channels=self.config.feature_channels,
            backbone_width=self.config.backbone_width,
            out_scale=scale)
        self.cost_volume = CorrelationCostVolume(self.num_disparities_small)
        self.aggregation = CostAggregation(self.config.feature_channels, self.num_disparities_small,
                                           self.config.cost_volume_channels)
        self.soft_argmin = SoftArgmin(window=self.config.soft_argmin_window)
        self.matchability = Matchability()
        limit = (self.config.residual_limit_fraction * self.max_disparity
                 if self.config.residual_limit_fraction else None)
        self.refinement = DisparityRefinement(in_scale=scale, residual_limit=limit)

        self.canonical_width = self.config.canonical_width
        #: Search range as a fraction of image width -- the quantity that is
        #: invariant under resizing, and the one to keep fixed across datasets.
        self.max_disparity_fraction = self.num_disparities / self.canonical_width

        self.apply(_init_weights)
        # After the generic init, so it is not overwritten by it.
        self.refinement.zero_init_residual()

    # -- internals ---------------------------------------------------------- #

    def _match(self, reference_features: torch.Tensor, source_features: torch.Tensor,
               reference_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One matching pass in the left-referenced convention."""
        volume = self.cost_volume(reference_features, source_features)
        cost = self.aggregation(volume)
        disparity_small = self.soft_argmin(cost)
        match = self.matchability(cost)
        disparity = self.refinement(reference_image, disparity_small, match)
        return {
            "disparity": disparity,
            "disparity_small": disparity_small,
            "matchability": match,
            "confidence": confidence_from_matchability(match),
            "cost": cost,
        }

    # -- public API --------------------------------------------------------- #

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                directions: Iterable[str] = ("left",)) -> Dict[str, Dict[str, torch.Tensor]]:
        """Predict disparity for the requested reference views.

        Args:
            left: ``(B, 3, H, W)`` rectified left image, values in ``[0, 1]``.
            right: ``(B, 3, H, W)`` rectified right image.
            directions: any of ``"left"`` and ``"right"``.  ``"left"`` gives
                ``d_L`` (left pixel ``x`` matches right pixel ``x - d_L``);
                ``"right"`` gives ``d_R`` (right pixel ``x`` matches left pixel
                ``x + d_R``).

        Returns:
            ``{direction: output_dict}``; see the class docstring for keys.  All
            spatial tensors are cropped back to the input resolution.
        """
        directions = tuple(directions)
        for direction in directions:
            if direction not in ("left", "right"):
                raise ValueError(f"unknown direction {direction!r}")
        if left.shape != right.shape:
            raise ValueError(f"left {tuple(left.shape)} and right {tuple(right.shape)} must match")

        padded_left, padding = pad_to_multiple(left, self.size_divisor)
        padded_right, _ = pad_to_multiple(right, self.size_divisor)

        # One feature-extractor call over the concatenated batch so that batch
        # normalisation sees left and right statistics together.
        batch = padded_left.shape[0]
        features = self.feature_extractor(torch.cat([padded_left, padded_right], dim=0))
        left_features, right_features = features[:batch], features[batch:]

        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        if "left" in directions:
            outputs["left"] = self._match(left_features, right_features, padded_left)
        if "right" in directions:
            # Mirror trick -- see stereo/model/cost_volume.py for the derivation.
            # Running the mirrored problem through the same weights yields the
            # right-referenced result, mirrored; flip it back.
            mirrored = self._match(flip_lr(right_features), flip_lr(left_features), flip_lr(padded_right))
            outputs["right"] = {key: flip_lr(value) for key, value in mirrored.items()}

        if padding != (0, 0):
            outputs = {direction: self._unpad_output(out, padding) for direction, out in outputs.items()}
        return outputs

    def _unpad_output(self, output: Dict[str, torch.Tensor], padding) -> Dict[str, torch.Tensor]:
        pad_right, pad_bottom = padding
        small_padding = (pad_right // self.scale, pad_bottom // self.scale)
        cropped = {"disparity": unpad(output["disparity"], padding)}
        for key in ("disparity_small", "matchability", "confidence", "cost"):
            cropped[key] = unpad(output[key], small_padding)
        return cropped

    def forward_left(self, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience wrapper returning only the left-referenced output."""
        return self(left, right, directions=("left",))["left"]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _init_weights(module: nn.Module) -> None:
    """Kaiming init for convolutions, unit init for normalisation: random start, no pretraining."""
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(config: Optional[StereoNetConfig] = None) -> StereoNet:
    return StereoNet(config)
