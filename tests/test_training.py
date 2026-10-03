"""Teacher/student mechanics and an end-to-end training run on synthetic data."""

import cv2
import numpy as np
import pytest
import pathlib

import torch

from stereo.config import Config, LossWeights
from stereo.data import DatasetMode, StereoFolderDataset, collate_samples
from stereo.model import StereoNet, StereoNetConfig
from stereo.training import LabelFreeObjective, ObjectiveState


def tiny_model(width=96):
    return StereoNet(StereoNetConfig.for_width(width, downsample=4, backbone_width=4,
                                               feature_channels=4))


# --------------------------------------------------------------------------- #
# EMA teacher
# --------------------------------------------------------------------------- #


def test_trainer_runs_an_epoch_on_an_unlabeled_folder(unlabeled_dataset, tmp_path):
    """A full Trainer epoch, including validation and checkpointing, with no labels."""
    from stereo.data.augmentation import GeometricAugmentConfig, PhotometricAugmentConfig, ResizeConfig
    from stereo.data.registry import DatasetSpec
    from stereo.training import Trainer

    config = Config()
    config.model = StereoNetConfig.for_width(96, downsample=4, backbone_width=4, feature_channels=4)
    config.dynamic_disparity = False
    config.data.train = [DatasetSpec(type="folder", root=unlabeled_dataset)]
    config.data.validation = [DatasetSpec(type="folder", root=unlabeled_dataset)]
    config.data.resize = ResizeConfig(height=48, width=96)
    config.data.photometric_augmentation = PhotometricAugmentConfig(enabled=True)
    config.data.geometric_augmentation = GeometricAugmentConfig(enabled=True, scale=(0.9, 1.1),
                                                                aspect=(0.95, 1.05), min_size=32)
    config.training.epochs = 2
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.use_amp = False
    config.training.warmup_iterations = 1
    config.training.output_dir = str(tmp_path / "out")
    config.optimizer.warmup_iterations = 1

    trainer = Trainer(config, device=torch.device("cpu"))
    best = trainer.fit()

    assert len(trainer.history) == 2
    assert np.isfinite(trainer.history[-1]["train/total"])
    assert "val/epe" in trainer.history[-1]

    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["extra"]["selection_is_label_free"] is True
    assert payload["model_config"]["num_disparities"] == 48


# --------------------------------------------------------------------------- #
# Schedule sanity on short runs
# --------------------------------------------------------------------------- #

def test_warmup_is_clamped_to_the_run_length():
    """Warm-ups are configured in absolute iterations, which silently ruins a
    short run: 500 warm-up iterations across a 60-iteration run pins the
    learning rate at a few percent of its configured value forever, and leaves
    the losses gated behind the warm-up permanently disabled."""
    from stereo.training.loop import MAX_WARMUP_FRACTION, effective_warmup

    # A long run keeps the configured value.
    assert effective_warmup(500, 100_000) == 500
    # A short run clamps it.
    assert effective_warmup(500, 60) == max(1, int(60 * MAX_WARMUP_FRACTION))
    assert effective_warmup(500, 60) < 60
    # Never negative, never zero-length on a tiny run.
    assert effective_warmup(0, 60) == 0
    assert effective_warmup(500, 1) >= 1


def test_visualisations_are_written_on_the_configured_cadence(unlabeled_dataset, tmp_path):
    """left / right / disparity panels every N epochs, from clean images and
    without touching ground truth."""
    from stereo.data.augmentation import GeometricAugmentConfig, PhotometricAugmentConfig, ResizeConfig
    from stereo.data.registry import DatasetSpec
    from stereo.training import Trainer

    config = Config()
    config.model = StereoNetConfig.for_width(96, downsample=4, backbone_width=4, feature_channels=4)
    config.dynamic_disparity = False
    config.data.train = [DatasetSpec(type="folder", root=unlabeled_dataset)]
    config.data.validation = [DatasetSpec(type="folder", root=unlabeled_dataset)]
    config.data.resize = ResizeConfig(48, 96)
    config.data.photometric_augmentation = PhotometricAugmentConfig(enabled=False)
    config.data.geometric_augmentation = GeometricAugmentConfig(enabled=False)
    config.training.epochs = 5
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.use_amp = False
    config.training.output_dir = str(tmp_path / "out")
    config.training.visualize_every = 2

    Trainer(config, device=torch.device("cpu")).fit()

    directory = tmp_path / "out" / "visualizations"
    written = sorted(p.name for p in directory.glob("*.png"))
    # Epochs 0, 2 and 4 of a 5-epoch run.
    assert written == ["epoch_0000.png", "epoch_0002.png", "epoch_0004.png"], written
    assert (directory / "epoch_0000.png").stat().st_size > 1000


def test_visualisation_can_be_disabled(unlabeled_dataset, tmp_path):
    from stereo.data.augmentation import GeometricAugmentConfig, PhotometricAugmentConfig, ResizeConfig
    from stereo.data.registry import DatasetSpec
    from stereo.training import Trainer

    config = Config()
    config.model = StereoNetConfig.for_width(96, downsample=4, backbone_width=4, feature_channels=4)
    config.dynamic_disparity = False
    config.data.train = [DatasetSpec(type="folder", root=unlabeled_dataset)]
    config.data.resize = ResizeConfig(48, 96)
    config.data.photometric_augmentation = PhotometricAugmentConfig(enabled=False)
    config.data.geometric_augmentation = GeometricAugmentConfig(enabled=False)
    config.training.epochs = 2
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.use_amp = False
    config.training.output_dir = str(tmp_path / "out")
    config.training.visualize_every = 0

    Trainer(config, device=torch.device("cpu")).fit()
    assert not (tmp_path / "out" / "visualizations").exists()


def test_visualisation_is_one_row_per_sample_with_four_panels(unlabeled_dataset, tmp_path):
    """Layout contract: each sample gets its own row of left / right / disparity
    / warped, and each row is sized to that sample's own aspect ratio rather than
    to the padded batch height."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stereo.data.augmentation import GeometricAugmentConfig, PhotometricAugmentConfig, ResizeConfig
    from stereo.data.registry import DatasetSpec
    from stereo.training import Trainer

    config = Config()
    config.model = StereoNetConfig.for_width(96, downsample=4, backbone_width=4, feature_channels=4)
    config.dynamic_disparity = False
    config.data.train = [DatasetSpec(type="folder", root=unlabeled_dataset)]
    config.data.validation = [DatasetSpec(type="folder", root=unlabeled_dataset)]
    config.data.resize = ResizeConfig(48, 96)
    config.data.photometric_augmentation = PhotometricAugmentConfig(enabled=False)
    config.data.geometric_augmentation = GeometricAugmentConfig(enabled=False)
    config.training.output_dir = str(tmp_path / "out")
    config.training.num_workers = 0
    config.training.batch_size = 2
    config.training.visualize_samples = 2
    trainer = Trainer(config, device=torch.device("cpu"))
    path = trainer.save_visualization(0)
    assert path is not None and pathlib.Path(path).exists()

    figures = [plt.figure(n) for n in plt.get_fignums()]
    assert not figures, "the figure must be closed, or a long run leaks them"

    from PIL import Image
    with Image.open(path) as image:
        assert image.width > image.height, "four columns should be wider than tall"


def test_visualisation_handles_train_and_val_at_different_heights(unlabeled_dataset, tmp_path):
    """The figure draws from BOTH loaders, and with an aspect-preserving resize
    their batches have different heights -- so nothing may assume they stack."""
    import matplotlib
    matplotlib.use("Agg")

    from stereo.data.augmentation import (GeometricAugmentConfig, PhotometricAugmentConfig,
                                          ResizeConfig)
    from stereo.data.registry import DatasetSpec
    from stereo.training import Trainer

    tall = tmp_path / "tall"
    for side in ("left", "right"):
        (tall / side).mkdir(parents=True)
    import cv2
    import numpy as np
    for index in range(3):
        image = (np.random.default_rng(index).random((160, 96, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(tall / "left" / f"{index}.png"), image)
        cv2.imwrite(str(tall / "right" / f"{index}.png"), image)

    config = Config()
    config.model = StereoNetConfig.for_width(96, downsample=4, backbone_width=4, feature_channels=4)
    config.dynamic_disparity = False
    # Wide training images, tall validation images: different aspect ratios, so
    # preserve_aspect gives the two loaders different heights.
    config.data.train = [DatasetSpec(type="folder", root=unlabeled_dataset)]
    config.data.validation = [DatasetSpec(type="folder", root=str(tall))]
    config.data.resize = ResizeConfig(width=96, preserve_aspect=True)
    config.data.photometric_augmentation = PhotometricAugmentConfig(enabled=False)
    config.data.geometric_augmentation = GeometricAugmentConfig(enabled=False)
    config.training.output_dir = str(tmp_path / "out")
    config.training.num_workers = 0
    config.training.batch_size = 2
    config.training.visualize_samples = 2

    trainer = Trainer(config, device=torch.device("cpu"))
    path = trainer.save_visualization(0)
    assert path is not None and pathlib.Path(path).exists()
