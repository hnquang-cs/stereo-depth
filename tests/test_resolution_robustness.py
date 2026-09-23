"""One model, any input shape: the search range, the estimator, the calibrator.

``num_disparities`` is part of the weights, so deriving it from the input width
means a different, incompatible network per resolution.  These tests pin the
alternative: a range fixed at a declared ``canonical_width``, with any other
resolution handled by resizing and scaling the answer back.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from stereo.model import StereoNet, StereoNetConfig, canonical_size, predict_disparity
from stereo.model.cost_volume import soft_argmin


# -- the reason the range must be fixed ------------------------------------- #

def test_width_derived_range_produces_incompatible_checkpoints():
    """Why ``min(width // 2, 384)`` cannot give one resolution-robust model."""
    big = StereoNet(StereoNetConfig.for_width(640))
    small = StereoNet(StereoNetConfig.for_width(224))
    assert big.num_disparities != small.num_disparities
    with pytest.raises(RuntimeError, match="size mismatch"):
        small.load_state_dict(big.state_dict())


def test_a_fixed_range_runs_at_any_shape_and_ratio():
    net = StereoNet(StereoNetConfig(num_disparities=96)).eval()
    for height, width in ((384, 640), (375, 1242), (224, 224), (272, 993)):
        with torch.no_grad():
            out = net(torch.rand(1, 3, height, width), torch.rand(1, 3, height, width))
        assert out["left"]["disparity"].shape[-2:] == (height, width)


# -- the rescaling ---------------------------------------------------------- #

class _ConstantModel(torch.nn.Module):
    """Always predicts ``value`` px, so the rescaling is the only thing measured."""

    canonical_width = 640

    def __init__(self, value=40.0):
        super().__init__()
        self.value = value

    def forward(self, left, right):
        return {"left": {"disparity": torch.full_like(left[:, :1], self.value)}}


@pytest.mark.parametrize("width", [320, 640, 1280, 1242])
def test_predicted_disparity_is_scaled_into_the_input_s_own_pixels(width):
    """Disparity scales linearly with horizontal resize, and must be undone."""
    out = predict_disparity(_ConstantModel(40.0), torch.rand(1, 3, 200, width),
                            torch.rand(1, 3, 200, width))["left"]["disparity"]
    assert float(out.mean()) == pytest.approx(40.0 * width / 640, rel=1e-4)


def test_prediction_keeps_the_input_resolution():
    net = StereoNet(StereoNetConfig(num_disparities=96, canonical_width=640)).eval()
    for height, width in ((375, 1242), (224, 224), (540, 960)):
        out = predict_disparity(net, torch.rand(1, 3, height, width),
                                torch.rand(1, 3, height, width))["left"]
        assert out["disparity"].shape[-2:] == (height, width)
        assert out["confidence"].shape[-2:] == (height, width)


def test_canonical_size_preserves_aspect_ratio():
    assert canonical_size(375, 1242, 640) == (193, 640)      # 375 * 640/1242
    assert canonical_size(384, 640, 640) == (384, 640)       # already canonical
    assert canonical_size(100, 10000, 640)[0] >= 64          # clamped, not degenerate


def test_max_disparity_fraction_is_the_resize_invariant_description():
    net = StereoNet(StereoNetConfig(num_disparities=96, canonical_width=640))
    assert net.max_disparity_fraction == pytest.approx(0.15)


# -- the estimator ---------------------------------------------------------- #

def test_peak_restricted_soft_argmin_beats_the_full_expectation_on_a_bimodal_cost():
    """The failure the window exists for: two modes, and the mean lands on neither."""
    cost = torch.full((1, 21, 1, 1), 1.0)
    cost[0, 4] = 0.0    # true match
    cost[0, 16] = 0.05  # a repeated-texture decoy, nearly as good

    full = float(soft_argmin(cost * 20))            # sharpened, still averages both
    restricted = float(soft_argmin(cost * 20, window=2))
    assert abs(restricted - 4.0) < 0.1, restricted
    assert abs(full - 4.0) > 2.0, f"expected the full expectation to be pulled away, got {full}"


def test_soft_argmin_window_is_still_differentiable():
    cost = torch.rand(1, 12, 4, 4, requires_grad=True)
    soft_argmin(cost, window=2).sum().backward()
    assert cost.grad is not None and cost.grad.abs().sum() > 0


def test_soft_argmin_window_recovers_subpixel_precision():
    """A window must not collapse to a hard argmin: the answer lies between bins."""
    cost = torch.full((1, 11, 1, 1), 5.0)
    cost[0, 5] = 0.0
    cost[0, 6] = 0.0  # exactly between bins 5 and 6
    assert float(soft_argmin(cost, window=2)) == pytest.approx(5.5, abs=0.05)


def test_full_expectation_is_still_available():
    """``window=None`` must reproduce the reference implementation exactly."""
    cost = torch.rand(1, 16, 2, 2)
    expected = (F.softmin(cost, dim=1)
                * torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1)).sum(1, keepdim=True)
    assert torch.allclose(soft_argmin(cost, window=None), expected, atol=1e-6)


# -- the calibrator --------------------------------------------------------- #

def _shifted_pair(width=640, height=256, shift=40, seed=0):
    """A textured pair with a known, exact horizontal shift."""
    generator = torch.Generator().manual_seed(seed)
    texture = torch.rand((1, 3, height, width + shift), generator=generator)
    texture = F.avg_pool2d(texture, 3, stride=1, padding=1)  # give it local structure
    return texture[..., :width], texture[..., shift:shift + width]


def test_calibration_recovers_a_known_disparity_without_ground_truth():
    from stereo.data import calibrate_disparity_range

    shift = 40
    estimate = calibrate_disparity_range([_shifted_pair(shift=shift)], canonical_width=640,
                                         search_fraction=0.4)
    assert estimate.disparity_at_percentile == pytest.approx(shift, abs=8), str(estimate)
    assert estimate.recommended >= shift, str(estimate)


def test_calibration_recommends_a_usable_num_disparities():
    from stereo.data import calibrate_disparity_range

    estimate = calibrate_disparity_range([_shifted_pair(shift=24)], canonical_width=640,
                                         search_fraction=0.4)
    assert estimate.recommended % 4 == 0, "must be a multiple of the downsample factor"
    assert 0 < estimate.recommended <= 640
    assert 0.0 <= estimate.reliable_fraction <= 1.0


def test_calibration_reads_only_the_two_views():
    """Label-free: a sample carrying ground truth must give the same answer."""
    from stereo.data import calibrate_disparity_range

    left, right = _shifted_pair(shift=32)
    plain = calibrate_disparity_range([{"left": left, "right": right}], canonical_width=640,
                                      search_fraction=0.4)
    poisoned = calibrate_disparity_range(
        [{"left": left, "right": right,
          "disparity": torch.full_like(left[:, :1], 999.0)}], canonical_width=640,
        search_fraction=0.4)
    assert plain.recommended == poisoned.recommended


# -- evaluation ------------------------------------------------------------- #

def test_evaluation_runs_at_the_canonical_width_not_the_benchmark_s_own():
    """A benchmark image is scored at its native size, but *run* at the canonical
    width, or the search range covers a different fraction of the image than the
    model was trained on."""
    from stereo.model import predict_left_disparity

    net = StereoNet(StereoNetConfig(num_disparities=96, canonical_width=640)).eval()
    # Middlebury-like native resolution, far from the training width.
    out = predict_left_disparity(net, torch.rand(1, 3, 994, 1500), torch.rand(1, 3, 994, 1500))
    assert out["disparity"].shape[-2:] == (994, 1500)
    assert out["confidence"].shape[-2:] == (994, 1500)


def test_benchmark_uses_the_canonical_width_path():
    """Guard against a future edit reverting evaluation to a raw forward pass."""
    import inspect

    from stereo.evaluation import benchmark

    source = inspect.getsource(benchmark)
    assert "predict_left_disparity(model" in source
    assert "model.forward_left(" not in source, (
        "evaluation must not call the model directly: that runs it at the "
        "benchmark's native resolution, not the width its range was declared at")


# -- aspect-preserving batches ---------------------------------------------- #

def test_resize_fixes_the_width_and_follows_the_source_aspect_ratio():
    from stereo.data.augmentation import ResizeConfig, ResizeSample

    resize = ResizeSample(ResizeConfig(width=640, preserve_aspect=True))
    assert resize.target_height(375, 1242) == 192    # KITTI, ratio 3.31
    assert resize.target_height(540, 960) == 352     # FlyingThings3D, ratio 1.78
    assert resize.target_height(224, 224) == 640     # square stays square
    for height, width in ((375, 1242), (540, 960), (500, 741)):
        assert resize.target_height(height, width) % 16 == 0


def test_ragged_batches_are_padded_at_the_bottom_with_a_validity_mask():
    """Padding must go below the image, so no pixel's x coordinate moves."""
    from stereo.data.base import collate_samples

    samples = [{"left": torch.rand(3, 192, 640), "right": torch.rand(3, 192, 640), "metadata": {}},
               {"left": torch.rand(3, 432, 640), "right": torch.rand(3, 432, 640), "metadata": {}}]
    batch = collate_samples(samples)
    assert batch["left"].shape == (2, 3, 432, 640)
    assert batch["valid_mask"].shape == (2, 1, 432, 640)
    assert float(batch["valid_mask"][0, 0, :192].min()) == 1.0
    assert float(batch["valid_mask"][0, 0, 192:].max()) == 0.0
    assert float(batch["valid_mask"][1].min()) == 1.0
    # The real rows are untouched by the padding.
    assert torch.equal(batch["left"][0, :, :192], samples[0]["left"])


def test_uniform_batches_carry_no_mask():
    """The common case must not pay for the ragged one."""
    from stereo.data.base import collate_samples

    samples = [{"left": torch.rand(3, 384, 640), "right": torch.rand(3, 384, 640), "metadata": {}}
               for _ in range(2)]
    assert "valid_mask" not in collate_samples(samples)


def test_padded_rows_do_not_contribute_to_the_objective():
    """Replicated padding matches itself perfectly, so it must be excluded."""
    from stereo.config import LossWeights
    from stereo.training import LabelFreeObjective, ObjectiveState

    torch.manual_seed(0)
    net = StereoNet(StereoNetConfig(num_disparities=32)).eval()
    left, right = torch.rand(1, 3, 128, 256), torch.rand(1, 3, 128, 256)
    outputs = net(left, right, directions=("left", "right"))

    mask = torch.ones(1, 1, 128, 256)
    mask[..., 96:, :] = 0.0
    objective = LabelFreeObjective(LossWeights())
    common = dict(state=ObjectiveState(warmup_scale=1.0), max_disparity=100.0)
    unmasked = objective(outputs, {"left": left, "right": right}, **common)
    masked = objective(outputs, {"left": left, "right": right}, valid_mask=mask, **common)
    assert float(masked["loss"]) != float(unmasked["loss"]), "the mask had no effect"
    assert torch.isfinite(masked["loss"])
