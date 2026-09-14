"""Metrics checked against hand-computable examples.

Ground truth is used here, which is allowed: these tests verify the evaluation
code, they do not train anything.
"""

import numpy as np
import pytest
import torch

from stereo.evaluation.depth_metrics import depth_metrics, depth_valid_mask, median_scale_factor
from stereo.evaluation.disparity_metrics import (DisparityAccumulator, disparity_valid_mask,
                                                 per_image_disparity_metrics)


def make_case():
    """Errors by hand: [0.0, 0.5, 1.0, 3.0, 6.0] against ground truth [10..50]."""
    disparity_gt = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).view(1, 1, 1, 5)
    disparity = torch.tensor([10.0, 20.5, 31.0, 43.0, 56.0]).view(1, 1, 1, 5)
    valid = torch.ones_like(disparity_gt, dtype=torch.bool)
    return disparity, disparity_gt, valid


def test_epe_and_rms_match_hand_calculation():
    disparity, disparity_gt, valid = make_case()
    metrics = per_image_disparity_metrics(disparity, disparity_gt, valid)
    errors = np.array([0.0, 0.5, 1.0, 3.0, 6.0])
    assert metrics["epe"] == pytest.approx(errors.mean())          # 2.1
    assert metrics["avgerr"] == pytest.approx(errors.mean())
    assert metrics["rms"] == pytest.approx(np.sqrt((errors ** 2).mean()))


def test_bad_thresholds_match_hand_calculation():
    """bad_t counts |err| > t, strictly, over 5 pixels."""
    disparity, disparity_gt, valid = make_case()
    metrics = per_image_disparity_metrics(disparity, disparity_gt, valid,
                                          bad_thresholds=(0.25, 0.5, 1.0, 2.0, 4.0))
    assert metrics["bad_0.25"] == pytest.approx(80.0)   # 0.5, 1, 3, 6 exceed 0.25
    assert metrics["bad_0.5"] == pytest.approx(60.0)    # 1, 3, 6   (0.5 is not > 0.5)
    assert metrics["bad_1"] == pytest.approx(40.0)      # 3, 6      (1.0 is not > 1.0)
    assert metrics["bad_2"] == pytest.approx(40.0)
    assert metrics["bad_4"] == pytest.approx(20.0)      # 6 only


def test_quantile_metrics():
    disparity, disparity_gt, valid = make_case()
    metrics = per_image_disparity_metrics(disparity, disparity_gt, valid)
    errors = torch.tensor([0.0, 0.5, 1.0, 3.0, 6.0], dtype=torch.float64)
    assert metrics["A90"] == pytest.approx(float(torch.quantile(errors, 0.90)))
    assert metrics["A95"] == pytest.approx(float(torch.quantile(errors, 0.95)))


def test_d1_requires_both_absolute_and_relative_failure():
    """KITTI D1: |err| > 3 AND |err|/d_gt > 0.05."""
    disparity_gt = torch.tensor([100.0, 10.0, 100.0, 10.0]).view(1, 1, 1, 4)
    #                            err=4     err=1    err=10    err=4
    # pixel 0: 4 > 3 but 4/100 = 0.04 < 0.05  -> NOT an outlier
    # pixel 1: 1 < 3                          -> NOT an outlier
    # pixel 2: 10 > 3 and 0.10 > 0.05         -> outlier
    # pixel 3: 4 > 3 and 0.40 > 0.05          -> outlier
    disparity = torch.tensor([104.0, 11.0, 110.0, 14.0]).view(1, 1, 1, 4)
    metrics = per_image_disparity_metrics(disparity, disparity_gt, torch.ones_like(disparity_gt, dtype=torch.bool))
    assert metrics["d1"] == pytest.approx(50.0)


def test_valid_mask_follows_the_reference_implementation():
    disparity_gt = torch.tensor([0.0, 1e-4, 5.0, 100.0, 300.0]).view(1, 1, 1, 5)
    valid = disparity_valid_mask(disparity_gt, max_disparity=251.0)
    assert valid.flatten().tolist() == [False, False, True, True, False]


def test_valid_mask_ignore_edge():
    """ignore_edge drops pixels whose match falls outside the image: d_gt < x - 1."""
    disparity_gt = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]).view(1, 1, 1, 8)
    valid = disparity_valid_mask(disparity_gt, max_disparity=100.0, ignore_edge=True)
    # x - 1 > 5  =>  x > 6  =>  only x = 7
    assert valid.flatten().tolist() == [False] * 7 + [True]


def test_global_and_per_image_aggregation_differ_and_are_both_reported():
    """One big image and one small image: the two conventions must disagree."""
    accumulator = DisparityAccumulator(bad_thresholds=(1.0,))
    big_gt = torch.full((1, 1, 1, 100), 10.0)
    big_pred = torch.full((1, 1, 1, 100), 10.0)      # 100 pixels, error 0
    small_gt = torch.full((1, 1, 1, 10), 10.0)
    small_pred = torch.full((1, 1, 1, 10), 20.0)     # 10 pixels, error 10
    accumulator.update(big_pred, big_gt, torch.ones_like(big_gt, dtype=torch.bool), "big")
    accumulator.update(small_pred, small_gt, torch.ones_like(small_gt, dtype=torch.bool), "small")

    results = accumulator.compute()
    assert results["global_epe"] == pytest.approx(10 * 10 / 110)   # pixel-weighted
    assert results["image_epe"] == pytest.approx((0.0 + 10.0) / 2)  # image-weighted
    assert results["global_epe"] != results["image_epe"]


def test_bad_and_correct_are_complementary():
    """The paper's %Bad(1.0) is 100 - the reference implementation's correct_1.0."""
    disparity, disparity_gt, valid = make_case()
    accumulator = DisparityAccumulator(bad_thresholds=(1.0,))
    accumulator.update(disparity, disparity_gt, valid)
    results = accumulator.compute()
    assert results["global_bad_1"] + results["global_correct_1"] == pytest.approx(100.0)


def test_depth_metrics_match_hand_calculation():
    depth_gt = torch.tensor([10.0, 20.0]).view(1, 1, 1, 2)
    depth = torch.tensor([11.0, 18.0]).view(1, 1, 1, 2)
    valid = torch.ones_like(depth_gt, dtype=torch.bool)
    metrics = depth_metrics(depth, depth_gt, valid)

    assert metrics["abs_rel"] == pytest.approx((1 / 10 + 2 / 20) / 2)
    assert metrics["sq_rel"] == pytest.approx((1 / 10 + 4 / 20) / 2)
    assert metrics["rmse"] == pytest.approx(np.sqrt((1 + 4) / 2))
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["delta_1"] == pytest.approx(1.0)   # 1.10 and 1.11 are both < 1.25
    assert metrics["scale_factor"] == 1.0, "no scale alignment by default"


def test_depth_metrics_apply_no_scaling_by_default():
    depth_gt = torch.full((1, 1, 4, 4), 10.0)
    depth = torch.full((1, 1, 4, 4), 5.0)   # a uniform factor-2 underestimate
    valid = torch.ones_like(depth_gt, dtype=torch.bool)
    assert depth_metrics(depth, depth_gt, valid)["abs_rel"] == pytest.approx(0.5)
    # Median scaling would hide it entirely; it is opt-in and reported.
    scale = median_scale_factor(depth, depth_gt, valid)
    assert scale == pytest.approx(2.0)
    assert depth_metrics(depth, depth_gt, valid, scale)["abs_rel"] == pytest.approx(0.0, abs=1e-9)


def test_sparse_depth_mask_is_not_densified():
    depth_gt = torch.tensor([0.0, 5.0, 0.0, 30.0]).view(1, 1, 1, 4)
    gt_mask = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 1, 1, 4)
    valid = depth_valid_mask(depth_gt, 1e-3, 80.0, gt_mask)
    assert int(valid.sum()) == 2
