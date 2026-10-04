"""The full pipeline on synthetic data with an analytically known ground truth.

A stereo pair built by shifting a texture by a constant amount has disparity
exactly equal to that shift, so the benchmark path can be checked end to end
without downloading anything: train label-free -> freeze -> evaluate against
ground truth -> read back summary.json.
"""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from stereo.data import DatasetMode, DatasetSpec, build_benchmark_dataset, collate_samples
from stereo.evaluation import evaluate_checkpoint, format_summary, get_protocol, write_results
from stereo.model import StereoNet, StereoNetConfig
from stereo.postprocess import PostProcessConfig


SHIFT = 6


@pytest.fixture
def synthetic_benchmark(tmp_path):
    """left/, right/ and left_disparity/ where the true disparity is exactly SHIFT."""
    root = tmp_path / "synthetic"
    for name in ("left", "right", "left_disparity"):
        (root / name).mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index in range(4):
        texture = cv2.GaussianBlur((rng.random((64, 96, 3)) * 255).astype(np.uint8), (3, 3), 0)
        # Left-referenced disparity d means left(x) == right(x - d).  With
        # left(x) = texture(x) and right(x) = texture(x + SHIFT) we get
        # right(x - SHIFT) = texture(x) = left(x), so the true disparity is SHIFT.
        left = texture[:, :-SHIFT]
        right = texture[:, SHIFT:]
        cv2.imwrite(str(root / "left" / f"{index:04d}.png"), left)
        cv2.imwrite(str(root / "right" / f"{index:04d}.png"), right)
        np.savez(str(root / "left_disparity" / f"{index:04d}.npz"),
                 np.full(left.shape[:2], float(SHIFT), dtype=np.float32))
    return str(root)


def test_synthetic_ground_truth_is_self_consistent(synthetic_benchmark):
    """Sanity-check the fixture itself before trusting any metric computed on it."""
    from stereo.geometry import warp_right_to_left
    dataset = build_benchmark_dataset(DatasetSpec(type="folder", root=synthetic_benchmark))
    batch = collate_samples([dataset[0]])
    reconstructed, valid = warp_right_to_left(batch["right"], batch["disparity_gt"])
    error = ((reconstructed - batch["left"]).abs() * valid).sum() / valid.sum()
    assert float(error) < 0.01, "the synthetic pair does not match its own ground truth"


def test_evaluate_a_perfect_predictor_scores_zero(synthetic_benchmark):
    """A model that outputs the true disparity must give EPE 0 and 0% bad."""
    from stereo.evaluation.disparity_metrics import DisparityAccumulator, disparity_valid_mask

    dataset = build_benchmark_dataset(DatasetSpec(type="folder", root=synthetic_benchmark))
    accumulator = DisparityAccumulator(bad_thresholds=(1.0,))
    for index in range(len(dataset)):
        batch = collate_samples([dataset[index]])
        disparity_gt = batch["disparity_gt"]
        valid = disparity_valid_mask(disparity_gt, max_disparity=100.0,
                                     gt_mask=batch["valid_gt_mask"])
        accumulator.update(disparity_gt.clone(), disparity_gt, valid)
    results = accumulator.compute()
    assert results["global_epe"] == pytest.approx(0.0)
    assert results["global_bad_1"] == pytest.approx(0.0)


def test_evaluate_a_known_biased_predictor(synthetic_benchmark):
    """A constant +2 px bias must give exactly EPE 2 and 100% bad-1.0."""
    from stereo.evaluation.disparity_metrics import DisparityAccumulator, disparity_valid_mask

    dataset = build_benchmark_dataset(DatasetSpec(type="folder", root=synthetic_benchmark))
    accumulator = DisparityAccumulator(bad_thresholds=(1.0, 4.0))
    for index in range(len(dataset)):
        batch = collate_samples([dataset[index]])
        disparity_gt = batch["disparity_gt"]
        valid = disparity_valid_mask(disparity_gt, 100.0, gt_mask=batch["valid_gt_mask"])
        accumulator.update(disparity_gt + 2.0, disparity_gt, valid)
    results = accumulator.compute()
    assert results["global_epe"] == pytest.approx(2.0)
    assert results["global_bad_1"] == pytest.approx(100.0)
    assert results["global_bad_4"] == pytest.approx(0.0)


def test_full_pipeline_train_freeze_evaluate(synthetic_benchmark, tmp_path):
    """Label-free training, then a ground-truth benchmark of the frozen result.

    The assertion is deliberately weak on accuracy -- a handful of CPU iterations
    on four images cannot produce a good model -- but it proves that the whole
    chain runs, that the numbers are finite, and that the output files are
    written in the documented format.
    """
    from stereo.config import LossWeights
    from stereo.training import ObjectiveState, SupervisedObjective
    from stereo.utils.checkpoint import build_model_from_checkpoint, save_checkpoint
    from stereo.data import StereoFolderDataset

    torch.manual_seed(0)
    # --- train, supervised ------------------------------------------------- #
    # with_labels: TRAIN mode reads ground truth only when asked, so a run that
    # does not want it still cannot see it.
    train_dataset = StereoFolderDataset(synthetic_benchmark, mode=DatasetMode.TRAIN)
    train_dataset.with_labels = True
    assert "disparity_gt" in train_dataset[0]
    assert "disparity_gt" not in StereoFolderDataset(synthetic_benchmark,
                                                     mode=DatasetMode.TRAIN)[0]
    batch = collate_samples([train_dataset[i] for i in range(4)])

    model = StereoNet(StereoNetConfig.for_width(96, downsample=4, backbone_width=4,
                                                feature_channels=4))
    objective = SupervisedObjective(LossWeights())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images = {"left": batch["left"], "right": batch["right"]}
    labels = {"disparity_gt": batch["disparity_gt"],
              "valid_gt_mask": batch.get("valid_gt_mask",
                                         torch.ones_like(batch["disparity_gt"]))}
    for _ in range(5):
        outputs = model(batch["left"], batch["right"], directions=("left",))
        loss = objective(outputs, images, ObjectiveState(warmup_scale=1.0),
                         labels=labels)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    checkpoint = str(tmp_path / "model.pt")
    save_checkpoint(checkpoint, model, epoch=0, iteration=5)

    # --- freeze and evaluate ------------------------------------------------ #
    frozen = build_model_from_checkpoint(checkpoint)
    protocol = get_protocol("custom_folder")
    summary = evaluate_checkpoint(frozen, protocol, synthetic_benchmark,
                                  torch.device("cpu"), PostProcessConfig(enabled=False),
                                  progress=False)

    assert summary["num_images_evaluated"] == 4
    assert np.isfinite(summary["disparity_metrics"]["all"]["global_epe"])
    assert np.isfinite(summary["disparity_metrics"]["all"]["global_bad_1"])
    assert summary["median_scaling_applied"] is False
    assert summary["postprocess"]["enabled"] is False
    assert len(summary["per_image"]) == 4
    assert "confidence_metrics_secondary" in summary

    # --- the report must render and the files must land --------------------- #
    text = format_summary(summary)
    assert "PRIMARY METRICS" in text and "SECONDARY METRICS" in text

    output_dir = str(tmp_path / "eval")
    write_results(summary, output_dir)
    with open(os.path.join(output_dir, "summary.json")) as handle:
        written = json.load(handle)
    assert written["protocol"]["name"] == "custom_folder"
    assert written["environment"]["torch"] == torch.__version__
    assert os.path.exists(os.path.join(output_dir, "per_image.csv"))


def test_postprocessed_run_reports_its_own_settings(synthetic_benchmark, tmp_path):
    """Raw and post-processed runs must be distinguishable in the output."""
    model = StereoNet(StereoNetConfig.for_width(96, downsample=4, backbone_width=4,
                                                feature_channels=4))
    protocol = get_protocol("custom_folder")
    raw = evaluate_checkpoint(model, protocol, synthetic_benchmark, torch.device("cpu"),
                              PostProcessConfig(enabled=False), max_samples=2,
                              compute_confidence_metrics=False, progress=False)
    processed = evaluate_checkpoint(
        model, protocol, synthetic_benchmark, torch.device("cpu"),
        PostProcessConfig(enabled=True, confidence_threshold=0.25, min_region_pixels=100),
        max_samples=2, compute_confidence_metrics=False, progress=False)

    assert raw["postprocess"]["enabled"] is False
    assert processed["postprocess"]["enabled"] is True
    assert processed["postprocess"]["confidence_threshold"] == 0.25
    # Post-processing can only remove pixels, never add them.
    assert (processed["disparity_metrics"]["all"]["num_valid_pixels"]
            <= raw["disparity_metrics"]["all"]["num_valid_pixels"])


def test_published_reference_is_attached_and_labelled():
    from stereo.evaluation import PUBLISHED_RESULTS
    reference = PUBLISHED_RESULTS["sceneflow"]
    assert reference["status"] == "published"
    assert reference["metrics"]["global_epe"] == 0.936
    assert reference["metrics"]["global_bad_1"] == 10.0
    middlebury = PUBLISHED_RESULTS["middlebury2014_test"]
    assert "caveat" in middlebury and "TEST" in middlebury["caveat"]


def test_a_held_out_part_scores_only_those_images(synthetic_benchmark):
    """part="val" with training's val_fraction is exactly the part it never saw."""
    model = StereoNet(StereoNetConfig.for_width(96, downsample=4, backbone_width=4,
                                                feature_channels=4))
    protocol = get_protocol("custom_folder")

    def scored(part):
        summary = evaluate_checkpoint(model, protocol, synthetic_benchmark, torch.device("cpu"),
                                      PostProcessConfig(enabled=False), compute_confidence_metrics=False,
                                      progress=False, part=part, val_fraction=0.25)
        return summary, {row["sample_id"] for row in summary["per_image"]}

    held, held_ids = scored("val")
    kept, kept_ids = scored("train")
    assert (held["num_images_evaluated"], kept["num_images_evaluated"]) == (1, 3)
    assert not held_ids & kept_ids
    assert "part: val" in format_summary(held)
    with pytest.raises(ValueError, match="part"):
        scored("test")


class _PerfectAtItsWidth(torch.nn.Module):
    """Answers with the ground truth itself, sampled at the width it runs at."""

    num_disparities, scale, max_disparity = 64, 4, 59

    def __init__(self, truth, canonical_width):
        super().__init__()
        self.truth, self.canonical_width = truth, canonical_width

    def num_parameters(self):
        return 0

    def forward(self, left, right, directions=("left",)):
        from stereo.geometry import resize_disparity
        small = resize_disparity(self.truth, tuple(left.shape[-2:]), mode="nearest")
        return {"left": {"disparity": small, "confidence": torch.ones_like(small)}}


@pytest.mark.parametrize("upsample", ["bilinear", "nearest"])
def test_the_width_floor_is_what_a_perfect_model_at_that_width_scores(tmp_path, upsample):
    """The floor reported beside a score is the score of a model that is exactly
    right at its own width: nothing a better model could recover."""
    root = tmp_path / "step"
    for name in ("left", "right", "left_disparity"):
        (root / name).mkdir(parents=True)
    truth = np.full((64, 96), 10.0, dtype=np.float32)
    truth[:, 50:] = 30.0                                  # one depth edge
    cv2.imwrite(str(root / "left" / "0000.png"), np.zeros((64, 96, 3), np.uint8))
    cv2.imwrite(str(root / "right" / "0000.png"), np.zeros((64, 96, 3), np.uint8))
    np.savez(str(root / "left_disparity" / "0000.npz"), truth)

    model = _PerfectAtItsWidth(torch.from_numpy(truth)[None, None], canonical_width=24)
    summary = evaluate_checkpoint(model, get_protocol("custom_folder"), str(root), torch.device("cpu"),
                                  PostProcessConfig(enabled=False), compute_confidence_metrics=False,
                                  progress=False, upsample=upsample)
    floor, scored = summary["resolution_floor"], summary["disparity_metrics"]["all"]
    assert summary["upsample"] == upsample
    assert floor["global_epe"] > 0                        # the edge cannot survive 4x
    assert floor["global_epe"] == pytest.approx(scored["global_epe"])
    assert floor["global_bad_1"] == pytest.approx(scored["global_bad_1"])
    assert "Width floor" in format_summary(summary)


def test_no_floor_is_claimed_for_sparse_ground_truth(tmp_path):
    root = tmp_path / "sparse"
    for name in ("left", "right", "left_disparity"):
        (root / name).mkdir(parents=True)
    truth = np.full((64, 96), 10.0, dtype=np.float32)
    truth[::2] = 0.0                                      # half the rows unknown, as lidar
    cv2.imwrite(str(root / "left" / "0000.png"), np.zeros((64, 96, 3), np.uint8))
    cv2.imwrite(str(root / "right" / "0000.png"), np.zeros((64, 96, 3), np.uint8))
    np.savez(str(root / "left_disparity" / "0000.npz"), truth)

    model = _PerfectAtItsWidth(torch.from_numpy(truth)[None, None], canonical_width=24)
    summary = evaluate_checkpoint(model, get_protocol("custom_folder"), str(root), torch.device("cpu"),
                                  PostProcessConfig(enabled=False), compute_confidence_metrics=False,
                                  progress=False)
    assert "resolution_floor" not in summary
