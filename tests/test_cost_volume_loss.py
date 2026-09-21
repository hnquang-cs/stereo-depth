"""Label-free shaping of the cost volume.

The measured failure this addresses: with only the indirect gradient that
reaches the cost volume through soft-argmin, the coarse disparity sat at the
midpoint of its search range and moved 1.4 px in 600 steps, while the
refinement network drove the photometric loss to 0.0047 by other means. The
network reconstructed the image almost perfectly and matched nothing.
"""

import numpy as np
import pytest
import torch

from stereo.losses import CostVolumeLoss, photometric_cost_volume
from tests.helpers import make_shifted_pair


def test_photometric_cost_volume_finds_a_known_shift():
    """The whole idea rests on this: the true matching cost needs no labels."""
    shift = 9
    left, right = make_shifted_pair(height=32, width=128, shift=shift)
    volume, valid = photometric_cost_volume(left, right, 24, "left")

    # Ignore the left border, where no correspondence exists.
    interior = volume[:, :, :, shift + 12:]
    assert float(interior[:, shift].mean()) == pytest.approx(0.0, abs=1e-3)
    assert int(interior.mean(dim=(0, 2, 3)).argmin()) == shift
    # Every other candidate must be clearly worse.
    others = [float(interior[:, d].mean()) for d in range(24) if d != shift]
    assert min(others) > 0.05


def test_photometric_cost_volume_right_direction():
    shift = 7
    left, right = make_shifted_pair(height=32, width=128, shift=shift)
    volume, _ = photometric_cost_volume(right, left, 20, "right")
    interior = volume[:, :, :, 10:-shift - 12]
    assert int(interior.mean(dim=(0, 2, 3)).argmin()) == shift


def test_the_validity_mask_marks_the_border():
    left, right = make_shifted_pair(height=8, width=32, shift=4)
    _, valid = photometric_cost_volume(left, right, 6, "left")
    assert float(valid[:, 0].mean()) == 1.0
    assert float(valid[0, 5, :, :5].sum()) == 0.0    # d=5 invalidates the first 5 columns


def test_loss_is_lower_when_the_cost_volume_agrees_with_photometry():
    """A cost volume peaked at the true disparity must beat a flat one."""
    shift = 6
    left, right = make_shifted_pair(height=32, width=96, shift=shift)
    loss_fn = CostVolumeLoss()
    shape = (1, 16, 32, 96)

    flat = torch.zeros(shape)
    correct = torch.full(shape, 5.0)
    correct[:, shift] = 0.0
    wrong = torch.full(shape, 5.0)
    wrong[:, 13] = 0.0

    correct_loss = float(loss_fn(correct, left, right)["loss"])
    flat_loss = float(loss_fn(flat, left, right)["loss"])
    wrong_loss = float(loss_fn(wrong, left, right)["loss"])
    assert correct_loss < flat_loss < wrong_loss


def test_ambiguous_regions_are_excluded():
    """Textureless areas match every disparity equally and teach nothing; forcing
    the cost volume to imitate a flat target would inject noise."""
    uniform = torch.full((1, 3, 32, 96), 0.5)      # no texture at all
    cost = torch.zeros((1, 16, 32, 96))
    terms = CostVolumeLoss(min_confidence=0.05)(cost, uniform, uniform)
    assert float(terms["supervised_ratio"]) < 0.05, "a blank image must supervise almost nothing"

    left, right = make_shifted_pair(height=32, width=96, shift=6)
    textured = CostVolumeLoss(min_confidence=0.05)(cost, left, right)
    assert float(textured["supervised_ratio"]) > 0.5, "a textured pair must supervise most pixels"


def test_the_target_carries_no_gradient():
    left, right = make_shifted_pair(height=16, width=64, shift=5)
    cost = torch.zeros((1, 12, 16, 64), requires_grad=True)
    CostVolumeLoss()(cost, left, right)["loss"].backward()
    assert cost.grad is not None and torch.isfinite(cost.grad).all()


def test_the_loss_reads_no_ground_truth():
    """It is built from the two input images alone -- that is the point."""
    import inspect
    from stereo.losses import cost_volume_loss
    source = inspect.getsource(cost_volume_loss)
    for token in ("disparity_gt", "depth_gt", "valid_gt_mask"):
        assert token not in source
    parameters = set(inspect.signature(CostVolumeLoss.forward).parameters)
    assert parameters == {"self", "cost", "reference", "source", "direction"}
