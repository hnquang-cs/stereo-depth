"""Losses: photometric, smoothness, consistency, pseudo-labels, confidence."""

import numpy as np
import pytest
import torch

from stereo.losses import (ConfidenceLoss, LeftRightConsistencyLoss, PhotometricLoss, SmoothnessLoss, occlusion_mask, ssim)
from tests.helpers import make_shifted_pair


def test_ssim_of_identical_images_is_one():
    image = torch.rand(1, 3, 32, 32)
    assert float(ssim(image, image).mean()) == pytest.approx(1.0, abs=1e-3)


def test_photometric_loss_is_minimal_at_the_true_disparity():
    shift = 6
    left, right = make_shifted_pair(height=32, width=64, shift=shift)
    loss_fn = PhotometricLoss()

    losses = {}
    for candidate in (0, 3, shift, 9, 12):
        disparity = torch.full((1, 1, 32, 64), float(candidate))
        losses[candidate] = float(loss_fn(left, right, disparity, "left")["loss"])
    assert min(losses, key=losses.get) == shift, losses
    # Not exactly zero: the 3x3 SSIM window at x = d straddles the boundary of the
    # valid region, where the source sample is border-clamped. It must still be
    # a small fraction of the loss at any wrong disparity.
    assert losses[shift] < 0.02 * losses[0], losses


def test_photometric_loss_masks_invalid_warps():
    left, right = make_shifted_pair(height=16, width=32, shift=4)
    disparity = torch.full((1, 1, 16, 32), 100.0)  # far outside the image
    terms = PhotometricLoss(border=0)(left, right, disparity, "left")
    assert float(terms["valid_warp"].sum()) == 0.0


def test_smoothness_penalises_gradients_and_respects_edges():
    loss_fn = SmoothnessLoss(normalize=False)
    flat_image = torch.full((1, 3, 16, 16), 0.5)
    constant = torch.full((1, 1, 16, 16), 5.0)
    assert float(loss_fn(constant, flat_image)) == pytest.approx(0.0, abs=1e-6)

    ramp = torch.arange(16, dtype=torch.float32).view(1, 1, 1, 16).expand(1, 1, 16, 16).contiguous()
    assert float(loss_fn(ramp, flat_image)) > 0.5

    # The same disparity step costs less where the image has a matching edge.
    edge_image = torch.zeros(1, 3, 16, 16)
    edge_image[..., 8:] = 1.0
    step = torch.zeros(1, 1, 16, 16)
    step[..., 8:] = 4.0
    assert float(loss_fn(step, edge_image)) < float(loss_fn(step, flat_image))


def test_smoothness_normalisation_blocks_the_shrink_to_zero_escape():
    """With normalisation, uniformly scaling the disparity down must not help."""
    image = torch.rand(1, 3, 16, 16)
    disparity = torch.rand(1, 1, 16, 16) + 1.0

    normalised = SmoothnessLoss(normalize=True)
    raw = SmoothnessLoss(normalize=False)
    assert float(normalised(disparity, image)) == pytest.approx(
        float(normalised(disparity * 0.01, image)), rel=1e-4)
    assert float(raw(disparity * 0.01, image)) < float(raw(disparity, image)) * 0.5


def test_left_right_consistency_loss_on_consistent_and_inconsistent_fields():
    loss_fn = LeftRightConsistencyLoss()
    shape = (1, 1, 8, 32)
    consistent = loss_fn(torch.full(shape, 4.0), torch.full(shape, 4.0))
    assert float(consistent["loss"]) == pytest.approx(0.0, abs=1e-5)
    inconsistent = loss_fn(torch.full(shape, 4.0), torch.full(shape, 10.0))
    assert float(inconsistent["loss"]) == pytest.approx(6.0, abs=1e-3)


def test_occlusion_mask_flags_disagreement():
    shape = (1, 1, 8, 32)
    agree = occlusion_mask(torch.full(shape, 4.0), torch.full(shape, 4.0), "left", 1.0)
    assert float(agree[..., 5:].mean()) == pytest.approx(1.0)
    disagree = occlusion_mask(torch.full(shape, 4.0), torch.full(shape, 12.0), "left", 1.0)
    assert float(disagree.mean()) == pytest.approx(0.0)


def test_confidence_loss_prefers_matching_the_reliability_target():
    loss_fn = ConfidenceLoss()
    confident = torch.log(torch.full((1, 1, 8, 8), 0.95))
    unconfident = torch.log(torch.full((1, 1, 8, 8), 0.10))
    reliable = torch.ones((1, 1, 8, 8))

    assert float(loss_fn(confident, reliable)["loss"]) < float(loss_fn(unconfident, reliable)["loss"])
    unreliable = torch.zeros((1, 1, 8, 8))
    assert float(loss_fn(unconfident, unreliable)["loss"]) < float(loss_fn(confident, unreliable)["loss"])


def test_confidence_target_is_detached():
    matchability = torch.log(torch.full((1, 1, 4, 4), 0.5)).requires_grad_(True)
    reliability = torch.rand((1, 1, 4, 4), requires_grad=True)
    ConfidenceLoss()(matchability, reliability)["loss"].backward()
    assert matchability.grad is not None
    assert reliability.grad is None


def test_confidence_loss_is_autocast_safe():
    """torch bans binary_cross_entropy under autocast, and the confidence loss is
    the only site in this package that uses it.

    The ban is raised by CUDA autocast, which cannot run here, so this test
    pins the property that actually prevents it: the term is computed in
    float32 regardless of the input dtype, which is what taking it outside
    autocast achieves.
    """
    matchability = torch.log(torch.full((2, 1, 8, 8), 0.5)).half().requires_grad_(True)
    reliability = torch.ones((2, 1, 8, 8), dtype=torch.half)

    terms = ConfidenceLoss()(matchability, reliability)
    assert terms["loss"].dtype == torch.float32, "must be computed in float32"
    assert torch.isfinite(terms["loss"])

    # And it must still be differentiable back into the half-precision input.
    terms["loss"].backward()
    assert matchability.grad is not None and torch.isfinite(matchability.grad).all()


def test_confidence_loss_value_matches_bce_by_hand():
    """Guard the float32 conversion against silently changing the number."""
    probability = 0.25
    matchability = torch.log(torch.full((1, 1, 4, 4), probability))
    target = torch.ones((1, 1, 4, 4))
    expected = -np.log(probability)            # BCE with target 1 is -log(p)
    assert float(ConfidenceLoss()(matchability, target)["loss"]) == pytest.approx(expected, rel=1e-5)


def test_a_constant_disparity_field_is_free_under_left_right_consistency():
    """The reason an oversized left-right weight collapses training.

    A constant field is *exactly* left-right consistent, while any real
    structured field is not. So the term does not merely dominate when
    over-weighted -- it rewards flatness.  This pins the fact, so the weighting
    that depends on it is never changed by accident.
    """
    shape = (1, 1, 32, 128)
    constant = torch.full(shape, 30.0)
    assert float(LeftRightConsistencyLoss()(constant, constant)["loss"]) == pytest.approx(0.0, abs=1e-5)

    torch.manual_seed(0)
    structured_left = constant + torch.randn(shape) * 3.0
    structured_right = constant + torch.randn(shape) * 3.0
    assert float(LeftRightConsistencyLoss()(structured_left, structured_right)["loss"]) > 1.0


def test_disparity_space_losses_are_normalised_by_the_search_range():
    """left_right, pseudo and range_penalty are pixel-scale quantities weighed
    against a photometric residual in [0, 1]. They must be divided by the search
    range before weighting, or the weights mean different things at different
    resolutions -- and a plausible-looking 0.5 becomes eight times the whole
    photometric signal."""
    from stereo.config import LossWeights
    from stereo.training import LabelFreeObjective, ObjectiveState

    torch.manual_seed(0)
    left, right = torch.rand(1, 3, 32, 128), torch.rand(1, 3, 32, 128)
    images = {"left": left, "right": right}

    def total_for(max_disparity):
        outputs = {}
        for direction, value in (("left", 20.0), ("right", 26.0)):
            outputs[direction] = {
                "disparity": torch.full((1, 1, 32, 128), value),
                "disparity_small": torch.full((1, 1, 8, 32), value / 4),
                "matchability": torch.full((1, 1, 8, 32), -0.1),
            }
        objective = LabelFreeObjective(LossWeights(photometric=0.0, smoothness=0.0, low_resolution=0.0,
                        confidence=0.0, range_penalty=0.0, left_right=1.0))
        result = objective(outputs, images, ObjectiveState(warmup_scale=1.0),
                           max_disparity=max_disparity)
        return float(result["loss"]), result["logs"]["left_right"]

    small_total, small_raw = total_for(100.0)
    large_total, large_raw = total_for(400.0)

    # The raw, reported error is the same -- it is a physical pixel quantity.
    assert small_raw == pytest.approx(large_raw, rel=1e-5)
    # But the contribution to the objective scales inversely with the range, so
    # the same weight means the same thing regardless of the disparity range.
    assert small_total == pytest.approx(4.0 * large_total, rel=1e-4)
