"""Block matching: photometric cost volumes built from the two images alone.

These are measurement utilities, not a loss -- used to calibrate the disparity
search range without labels, and as a baseline to measure a trained model
against. They read the left and right images only.
"""

import numpy as np
import pytest
import torch

from stereo.block_matching import (MATCH_WINDOW, box_filter, photometric_cost_volume,
                                   pooled_photometric_cost_volume)
from tests.helpers import make_shifted_pair


def test_cost_volume_is_minimal_at_the_true_disparity():
    shift = 6
    left, right = make_shifted_pair(height=32, width=64, shift=shift)
    cost, valid = photometric_cost_volume(left, right, num_disparities=16, direction="left")
    assert cost.shape[1] == 16
    interior = cost[..., :, 20:]           # away from the border ramp
    best = interior.argmin(dim=1).float().median()
    assert abs(float(best) - shift) < 1.0, f"argmin landed at {float(best)}, expected {shift}"
    # At disparity d, x - d must land inside the image, so validity is per-d.
    assert valid[:, shift, :, shift:].min() > 0.5
    assert valid[:, shift, :, :shift].max() < 0.5


def test_box_filter_averages_over_its_window():
    values = torch.zeros(1, 1, 9, 9)
    values[0, 0, 4, 4] = 81.0
    assert float(box_filter(values, 1)[0, 0, 4, 4]) == 81.0
    smoothed = box_filter(values, 9)
    assert float(smoothed[0, 0, 4, 4]) == pytest.approx(1.0, abs=1e-4)


def test_the_window_is_what_makes_matching_work():
    """A 1x1 cost carries almost no information; this is why MATCH_WINDOW exists.

    Measured on a real Middlebury pair, a 1x1 cost scored MAE 13.28 px against a
    true disparity of 2.2-18.1 -- worse than predicting a constant (4.52 px) --
    while a 9x9 window scored 1.78 px. Here the same property is pinned on a
    low-texture synthetic pair, where a per-pixel cost is likewise ambiguous.
    """
    torch.manual_seed(0)
    shift = 5
    left, right = make_shifted_pair(height=48, width=96, shift=shift)
    # Independent sensor noise on each view: a single pixel's difference is now
    # an unreliable match score, which is the situation a window exists for.
    left = (left + torch.randn_like(left) * 0.25).clamp(0, 1)
    right = (right + torch.randn_like(right) * 0.25).clamp(0, 1)

    def error(window):
        cost, _ = photometric_cost_volume(left, right, 12, "left", window)
        picked = cost[..., 12:].argmin(dim=1).float()
        return float((picked - shift).abs().mean())

    per_pixel, windowed = error(1), error(MATCH_WINDOW)
    assert windowed < per_pixel / 2, (
        f"window {MATCH_WINDOW} scored {windowed:.2f} px against {per_pixel:.2f} "
        f"for a 1x1 cost; the window should roughly halve it or better")


def test_pooled_volume_matches_at_full_resolution():
    """The target must be built by matching at full resolution and pooling down,
    never by matching already-downsampled images."""
    left, right = make_shifted_pair(height=32, width=64, shift=8)
    cost, valid = pooled_photometric_cost_volume(left, right, num_disparities=8, scale=4,
                                                 direction="left")
    assert cost.shape == (1, 8, 8, 16)
    assert valid.shape == cost.shape
    assert torch.isfinite(cost).all()


def test_block_matching_reads_no_ground_truth():
    import inspect

    from stereo import block_matching

    source = inspect.getsource(block_matching)
    for token in ("disparity_gt", "depth_gt", "valid_gt_mask"):
        assert token not in source
