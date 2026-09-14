"""Datasets, augmentation, configuration and post-processing."""

import os

import cv2
import numpy as np
import pytest
import torch

from stereo.data import (BatchGeometricAugment, DatasetMode, DatasetSpec, GeometricAugmentConfig,
                         PhotometricAugment, PhotometricAugmentConfig, ResizeConfig,
                         StereoFolderDataset, build_loader, build_training_datasets,
                         collate_samples)
from stereo.data.io import read_pfm


def make_folder(root, count=3, height=48, width=96, shift=6, name="cam"):
    base = os.path.join(root, name)
    os.makedirs(os.path.join(base, "left"))
    os.makedirs(os.path.join(base, "right"))
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 31)
    for index in range(count):
        texture = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(base, "left", f"{index:04d}.png"), texture)
        cv2.imwrite(os.path.join(base, "right", f"{index:04d}.png"), np.roll(texture, -shift, axis=1))
    return base


def test_folder_dataset_needs_only_left_and_right(tmp_path):
    root = make_folder(str(tmp_path))
    dataset = StereoFolderDataset(root, mode=DatasetMode.TRAIN)
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["left"].shape == (3, 48, 96)
    assert sample["right"].shape == (3, 48, 96)
    assert sample["metadata"]["sample_id"] == "0000"


def test_folder_dataset_reads_optional_calibration(tmp_path):
    root = make_folder(str(tmp_path))
    with open(os.path.join(root, "calib.txt"), "w") as handle:
        handle.write("cam0=[1075.0 0 640; 0 1075.0 360; 0 0 1]\nbaseline=120.0\ndoffs=1.5\nndisp=256\n")
    dataset = StereoFolderDataset(root, mode=DatasetMode.TRAIN)
    metadata = dataset[0]["metadata"]
    assert metadata["focal_length"] == pytest.approx(1075.0)
    assert metadata["baseline"] == pytest.approx(0.12)   # mm -> m
    assert metadata["doffs"] == pytest.approx(1.5)


def test_collate_keeps_metadata_as_lists(tmp_path):
    root = make_folder(str(tmp_path))
    dataset = StereoFolderDataset(root, mode=DatasetMode.TRAIN)
    batch = collate_samples([dataset[0], dataset[1]])
    assert batch["left"].shape == (2, 3, 48, 96)
    assert batch["metadata"]["sample_id"] == ["0000", "0001"]


def test_weighted_multi_dataset_sampling_hits_the_requested_proportions(tmp_path):
    big = make_folder(str(tmp_path), count=40, name="big")
    small = make_folder(str(tmp_path), count=4, name="small")
    specs = [DatasetSpec(type="folder", root=big, weight=0.25),
             DatasetSpec(type="folder", root=small, weight=0.75)]

    dataset, weights, summary = build_training_datasets(
        specs, DatasetMode.TRAIN, ResizeConfig(48, 96), None, seed=0)
    assert weights is not None
    assert [entry["size"] for entry in summary] == [40, 4]

    # Expected draw fraction per dataset is weight / sum(weight), independent of size.
    big_mass = float(weights[:40].sum())
    small_mass = float(weights[40:].sum())
    assert big_mass / (big_mass + small_mass) == pytest.approx(0.25, abs=1e-6)

    loader = build_loader(dataset, batch_size=2, sample_weights=weights,
                          samples_per_epoch=400, num_workers=0, seed=0)
    names = []
    for batch in loader:
        names.extend(batch["metadata"]["dataset"])
    fraction_small = sum(name == "small" for name in names) / len(names)
    assert 0.65 < fraction_small < 0.85, fraction_small


def test_photometric_augmentation_is_stereo_consistent_and_keeps_a_clean_pair():
    config = PhotometricAugmentConfig(enabled=True, probability=1.0, asymmetric_probability=0.0)
    augment = PhotometricAugment(config, seed=0)
    rng = np.random.default_rng(0)
    texture = rng.random((16, 32, 3)).astype(np.float32)
    sample = augment({"left": texture.copy(), "right": texture.copy()})

    assert np.allclose(sample["left_clean"], texture)
    assert not np.allclose(sample["left"], texture), "augmentation did nothing"
    # Identical inputs must stay identical after a stereo-consistent jitter,
    # otherwise brightness constancy is broken.
    assert np.allclose(sample["left"], sample["right"], atol=1e-6)


def test_batch_geometric_augmentation_resizes_both_views_identically():
    config = GeometricAugmentConfig(enabled=True, scale=(0.5, 0.5), aspect=(1.0, 1.0),
                                    size_divisor=16, min_size=16)
    augment = BatchGeometricAugment(config, seed=0)
    texture = torch.rand(2, 3, 64, 128)
    batch = {"left": texture, "right": texture.clone(), "metadata": {"focal_length": [100.0, 100.0]}}
    out, (scale_x, scale_y) = augment(batch)

    assert out["left"].shape == out["right"].shape == (2, 3, 32, 64)
    assert torch.allclose(out["left"], out["right"])
    assert scale_x == pytest.approx(0.5) and scale_y == pytest.approx(0.5)
    # Intrinsics must follow the resize so that depth = f * B / d still holds.
    assert out["metadata"]["focal_length"] == [50.0, 50.0]


def test_no_random_crop_transform_exists():
    """The project must not ship a crop augmentation at all."""
    import stereo.data.augmentation as augmentation
    assert not any("crop" in name.lower() for name in dir(augmentation))
    source = open(augmentation.__file__).read().lower()
    assert "randomcrop" not in source


def test_pfm_reader_roundtrip(tmp_path):
    """PFM stores rows bottom-to-top; the reader must flip them back."""
    path = str(tmp_path / "disp.pfm")
    data = np.arange(6, dtype=np.float32).reshape(2, 3)   # rows [[0,1,2],[3,4,5]]
    with open(path, "wb") as handle:
        handle.write(b"Pf\n3 2\n-1.0\n")
        handle.write(np.flipud(data).astype("<f4").tobytes())
    assert np.array_equal(read_pfm(path), data)


def test_config_loading_applies_the_dynamic_disparity_rule(tmp_path):
    from stereo.config import load_config
    path = tmp_path / "c.yaml"
    path.write_text("dynamic_disparity: true\nmodel:\n  downsample: 8\n"
                    "data:\n  resize:\n    height: 540\n    width: 960\n")
    config = load_config(str(path))
    assert config.model.num_disparities == 384   # min(960 // 2, 384)
    assert config.model.downsample == 8


def test_config_rejects_unknown_keys(tmp_path):
    from stereo.config import load_config
    path = tmp_path / "c.yaml"
    path.write_text("training:\n  nonsense_key: 3\n")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(str(path))


def test_shipped_configs_load():
    from stereo.config import load_config
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("train_unlabeled.yaml", "adapt_unlabeled.yaml", "evaluate.yaml"):
        config = load_config(os.path.join(here, "configs", name))
        assert config.model.num_disparities % config.model.downsample == 0


def test_postprocess_confidence_and_region_gates():
    from stereo.postprocess import PostProcessConfig, postprocess_disparity

    disparity = torch.full((1, 1, 64, 64), 10.0)
    confidence = torch.full((1, 1, 64, 64), 0.9)
    confidence[..., :8, :8] = 0.1          # a low-confidence corner

    config = PostProcessConfig(enabled=True, confidence_threshold=0.25, min_region_pixels=1)
    filtered, valid = postprocess_disparity(disparity, confidence, config)
    assert float(valid[..., :8, :8].sum()) == 0.0
    assert float(valid[..., 32:, 32:].mean()) == 1.0
    assert float(filtered[0, 0, 0, 0]) == 0.0

    # The region gate removes small confident islands.
    confidence_island = torch.full((1, 1, 64, 64), 0.1)
    confidence_island[..., :10, :10] = 0.9          # only 100 confident pixels
    config = PostProcessConfig(enabled=True, confidence_threshold=0.25, min_region_pixels=2000)
    _, valid = postprocess_disparity(disparity, confidence_island, config)
    assert float(valid.sum()) == 0.0

    # A large region survives.
    config = PostProcessConfig(enabled=True, confidence_threshold=0.25, min_region_pixels=2000)
    _, valid = postprocess_disparity(disparity, torch.full((1, 1, 64, 64), 0.9), config)
    assert float(valid.mean()) == 1.0


def test_postprocess_disabled_is_a_no_op():
    from stereo.postprocess import PostProcessConfig, postprocess_disparity
    disparity = torch.rand(1, 1, 16, 16)
    filtered, valid = postprocess_disparity(disparity, torch.zeros(1, 1, 16, 16),
                                            PostProcessConfig(enabled=False))
    assert torch.equal(filtered, disparity)
    assert float(valid.mean()) == 1.0
