"""The Middlebury protocol end to end on a synthetic scene directory.

Exercises the parts that only Middlebury/ETH3D use: the PFM ground truth with
``inf`` for invalid pixels, ``mask0nocc.png``, ``calib.txt`` parsing, and the
separate nonocc / all accumulation that the paper's Table V reports.
"""

import cv2
import numpy as np
import pytest
import torch

from stereo.data import DatasetMode, DatasetSpec, MiddleburyDataset, build_benchmark_dataset, collate_samples
from stereo.evaluation import evaluate_checkpoint, format_summary, get_protocol
from stereo.model import StereoNet, StereoNetConfig
from stereo.postprocess import PostProcessConfig

SHIFT = 5


def write_pfm(path, data):
    """Write a single-channel PFM the way Middlebury does: rows bottom-to-top."""
    with open(path, "wb") as handle:
        handle.write(b"Pf\n%d %d\n-1.0\n" % (data.shape[1], data.shape[0]))
        handle.write(np.flipud(data).astype("<f4").tobytes())


@pytest.fixture
def middlebury_like(tmp_path):
    root = tmp_path / "trainingH"
    rng = np.random.default_rng(7)
    for name in ("SceneA", "SceneB"):
        scene = root / name
        scene.mkdir(parents=True)
        texture = cv2.GaussianBlur((rng.random((64, 96, 3)) * 255).astype(np.uint8), (3, 3), 0)
        cv2.imwrite(str(scene / "im0.png"), texture[:, :-SHIFT])   # left(x)  = texture(x)
        cv2.imwrite(str(scene / "im1.png"), texture[:, SHIFT:])    # right(x) = texture(x+SHIFT)

        disparity = np.full((64, 96 - SHIFT), float(SHIFT), dtype=np.float32)
        disparity[:8, :] = np.inf            # an invalid band, as Middlebury marks them
        write_pfm(str(scene / "disp0GT.pfm"), disparity)

        nonocc = np.full((64, 96 - SHIFT), 128, dtype=np.uint8)    # 128 = occluded
        nonocc[16:, :] = 255                                        # 255 = non-occluded valid
        cv2.imwrite(str(scene / "mask0nocc.png"), nonocc)

        (scene / "calib.txt").write_text(
            "cam0=[1000.0 0 480; 0 1000.0 320; 0 0 1]\ncam1=[1000.0 0 500; 0 1000.0 320; 0 0 1]\n"
            "doffs=20.0\nbaseline=200.0\nwidth=91\nheight=64\nndisp=64\n")
    return str(root)


def test_middlebury_loader_reads_ground_truth_and_masks(middlebury_like):
    dataset = MiddleburyDataset(middlebury_like, mode=DatasetMode.BENCHMARK)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["left"].shape == (3, 64, 91)
    # inf pixels are zeroed and excluded by valid_gt_mask, never fed onward as inf.
    assert torch.isfinite(sample["disparity_gt"]).all()
    assert float(sample["valid_gt_mask"][0, :8, :].sum()) == 0.0
    assert float(sample["valid_gt_mask"][0, 8:, :].mean()) == 1.0
    # nonocc is the intersection of mask0nocc == 255 with the valid mask.
    assert float(sample["nonocc_mask"][0, :16, :].sum()) == 0.0
    assert float(sample["nonocc_mask"][0, 16:, :].mean()) == 1.0


def test_middlebury_calibration_units(middlebury_like):
    dataset = MiddleburyDataset(middlebury_like, mode=DatasetMode.BENCHMARK)
    metadata = dataset[0]["metadata"]
    assert metadata["focal_length"] == pytest.approx(1000.0)
    assert metadata["baseline"] == pytest.approx(0.2)     # 200 mm -> 0.2 m
    assert metadata["doffs"] == pytest.approx(20.0)
    assert metadata["ndisp"] == pytest.approx(64.0)


def test_middlebury_training_mode_hides_ground_truth(middlebury_like):
    dataset = MiddleburyDataset(middlebury_like, mode=DatasetMode.TRAIN)
    sample = dataset[0]
    for key in ("disparity_gt", "valid_gt_mask", "nonocc_mask"):
        assert key not in sample


def test_middlebury_protocol_reports_nonocc_and_all_separately(middlebury_like):
    """The paper's Table V metric set, on both occlusion variants."""
    model = StereoNet(StereoNetConfig.for_width(96, downsample=4, backbone_width=4,
                                                feature_channels=4))
    protocol = get_protocol("middlebury2014")
    assert protocol.evaluate_nonocc is True

    summary = evaluate_checkpoint(model, protocol, middlebury_like, torch.device("cpu"),
                                  PostProcessConfig(enabled=False),
                                  compute_confidence_metrics=False, progress=False)

    assert set(summary["disparity_metrics"]) == {"all", "nonocc"}
    for variant in ("all", "nonocc"):
        for key in ("image_bad_2", "image_bad_4", "image_avgerr", "image_rms",
                    "image_A90", "image_A95"):
            assert key in summary["disparity_metrics"][variant], f"{variant}/{key}"
            assert np.isfinite(summary["disparity_metrics"][variant][key])

    # nonocc is a strict subset of all, so it must score over fewer pixels.
    assert (summary["disparity_metrics"]["nonocc"]["num_valid_pixels"]
            < summary["disparity_metrics"]["all"]["num_valid_pixels"])

    # The published TEST-split row must be attached with its caveat, not silently compared.
    assert summary["protocol"]["split"] == "training"
    assert "TRAINING" in summary["protocol"]["notes"]
    assert "PRIMARY METRICS" in format_summary(summary)


def test_perfect_predictor_on_middlebury_protocol_scores_zero(middlebury_like):
    """Feeding back the ground truth must give bad2 = 0 on both variants."""
    from stereo.evaluation.disparity_metrics import DisparityAccumulator, disparity_valid_mask

    dataset = build_benchmark_dataset(DatasetSpec(type="middlebury", root=middlebury_like))
    for variant in ("all", "nonocc"):
        accumulator = DisparityAccumulator()
        for index in range(len(dataset)):
            batch = collate_samples([dataset[index]])
            disparity_gt = batch["disparity_gt"]
            valid = disparity_valid_mask(disparity_gt, None, 1e-3, batch["valid_gt_mask"])
            if variant == "nonocc":
                valid = valid & (batch["nonocc_mask"] > 0.5)
            accumulator.update(disparity_gt.clone(), disparity_gt, valid)
        results = accumulator.compute()
        assert results["image_bad_2"] == pytest.approx(0.0)
        assert results["image_avgerr"] == pytest.approx(0.0)
        assert results["image_A95"] == pytest.approx(0.0)
