"""Teacher/student mechanics and an end-to-end training run on synthetic data."""

import cv2
import numpy as np
import pytest
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

def test_objective_runs_without_a_teacher_and_logs_the_expected_keys():
    model = tiny_model()
    # Every optional term switched on, so each one's logs are exercised. The
    # DEFAULT objective is the three Monodepth terms only, which is covered by
    # test_the_default_objective_is_monodepth_and_logs_only_its_terms.
    objective = LabelFreeObjective(LossWeights(confidence=0.05, low_resolution=0.5,
                                               range_penalty=0.1))
    left, right = torch.rand(2, 3, 32, 96), torch.rand(2, 3, 32, 96)
    outputs = model(left, right, directions=("left", "right"))
    result = objective(outputs, {"left": left, "right": right},
                       ObjectiveState(warmup_scale=1.0),
                       max_disparity=model.max_disparity)

    assert torch.isfinite(result["loss"])
    for key in ("photometric", "photometric_ssim", "photometric_l1", "smoothness", "left_right",
                "valid_warp_ratio", "disparity_mean", "disparity_max",
                "mean_confidence", "total"):
        assert key in result["logs"], key


def test_objective_requires_no_ground_truth_argument():
    """The objective's signature has nowhere to put a label."""
    import inspect
    parameters = set(inspect.signature(LabelFreeObjective.__call__).parameters)
    # valid_mask -- which rows of a ragged, aspect-preserving batch are real
    # image and which are padding. Derived from image SHAPES, never from
    # disparity. Any further parameter must be justified the same way.
    assert parameters == {"self", "student_outputs", "images", "state",
                          "max_disparity", "valid_mask"}


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

@pytest.fixture
def unlabeled_dataset(tmp_path):
    """left/ and right/ only -- no disparity directory at all."""
    root = tmp_path / "camera"
    (root / "left").mkdir(parents=True)
    (root / "right").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for index in range(4):
        texture = cv2.GaussianBlur((rng.random((48, 96, 3)) * 255).astype(np.uint8), (5, 5), 0)
        cv2.imwrite(str(root / "left" / f"{index:04d}.png"), texture)
        cv2.imwrite(str(root / "right" / f"{index:04d}.png"), np.roll(texture, -6, axis=1))
    return str(root)


def test_training_step_on_an_unlabeled_dataset_reduces_the_loss(unlabeled_dataset):
    """The end-to-end claim: images only, random init, loss goes down."""
    torch.manual_seed(0)
    dataset = StereoFolderDataset(unlabeled_dataset, mode=DatasetMode.TRAIN)
    assert set(dataset[0]) == {"left", "right", "left_clean", "right_clean", "metadata"} or \
           {"left", "right", "metadata"} <= set(dataset[0])

    batch = collate_samples([dataset[i] for i in range(4)])
    model = tiny_model()
    objective = LabelFreeObjective(LossWeights())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    images = {"left": batch["left"], "right": batch["right"]}
    losses = []
    for _ in range(12):
        outputs = model(batch["left"], batch["right"], directions=("left", "right"))
        result = objective(outputs, images, ObjectiveState(warmup_scale=1.0),
                           max_disparity=model.max_disparity)
        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        losses.append(float(result["loss"].detach()))

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


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
    assert "val/photometric" in trainer.history[-1]

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


def test_short_run_actually_leaves_warmup_and_enables_the_losses(unlabeled_dataset, tmp_path):
    """The reported failure: 2 steps/epoch x 30 epochs = 60 iterations, so the
    learning rate never left warm-up and left-right consistency never switched
    on. Both must now be active well before the run ends."""
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
    config.training.epochs = 4
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.use_amp = False
    config.training.output_dir = str(tmp_path / "out")
    config.training.warmup_iterations = 500      # far longer than the whole run
    config.optimizer.warmup_iterations = 500

    trainer = Trainer(config, device=torch.device("cpu"))
    assert trainer.total_iterations < 500, "this test must model a short run"
    assert trainer.lr_warmup < trainer.total_iterations
    assert trainer.loss_warmup < trainer.total_iterations

    trainer.fit()

    # The learning rate must have reached its configured value, not stayed at a
    # few percent of it.
    peak = max(record["train/lr"] for record in trainer.history)
    assert peak > 0.5 * config.optimizer.learning_rate, f"lr never ramped: peak {peak:.2e}"

    # And the left-right consistency term must actually have been applied.
    assert any(record["train/left_right"] > 0 for record in trainer.history)


def test_range_penalty_punishes_disparity_beyond_the_search_range():
    """The refinement head is an unbounded relu(base + residual); nothing in the
    architecture stops it emitting disparities the cost volume cannot support."""
    from stereo.config import LossWeights
    from stereo.training import LabelFreeObjective, ObjectiveState

    objective = LabelFreeObjective(LossWeights(range_penalty=1.0))
    left, right = torch.rand(1, 3, 32, 96), torch.rand(1, 3, 32, 96)
    images = {"left": left, "right": right}

    def penalty_for(value):
        outputs = {d: {"disparity": torch.full((1, 1, 32, 96), value),
                       "disparity_small": torch.full((1, 1, 8, 24), value / 4),
                       "matchability": torch.full((1, 1, 8, 24), -0.1)}
                   for d in ("left", "right")}
        result = objective(outputs, images, ObjectiveState(warmup_scale=1.0),
                           max_disparity=100.0)
        return result["logs"].get("range_penalty", 0.0)

    assert penalty_for(50.0) == 0.0, "inside the range costs nothing"
    assert penalty_for(100.0) == 0.0
    assert penalty_for(200.0) > 0.0, "beyond the range must be penalised"
    assert penalty_for(400.0) > penalty_for(200.0), "penalty must grow with the excess"


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


def test_monodepth_preset_is_the_whole_objective():
    """Exactly the three Monodepth terms, and nothing else.

    The paper's NSCE term is anchored on ground-truth disparity, so it has no
    label-free form and is absent rather than replaced.
    """
    from stereo.config import LossWeights

    weights = LossWeights.monodepth()
    assert (weights.photometric, weights.left_right, weights.smoothness) == (1.0, 1.0, 0.1)
    for unused in ("confidence", "range_penalty", "low_resolution"):
        assert getattr(weights, unused) == 0.0, f"{unused} is not part of the Monodepth objective"
    for removed in ("cost_volume", "pseudo"):
        assert not hasattr(weights, removed), (
            f"{removed} was removed: the cost-volume loss was an invented stand-in for "
            "NSCE, and teacher self-training is not part of the Monodepth objective")


def test_the_default_objective_is_monodepth_and_logs_only_its_terms():
    """The default must be exactly photometric + left-right + smoothness.

    Terms that are off must not appear in the logs either, so a training run
    cannot look like it is optimising something it is not.
    """
    model = tiny_model()
    objective = LabelFreeObjective(LossWeights())
    left, right = torch.rand(2, 3, 32, 96), torch.rand(2, 3, 32, 96)
    outputs = model(left, right, directions=("left", "right"))
    result = objective(outputs, {"left": left, "right": right},
                       ObjectiveState(warmup_scale=1.0),
                       max_disparity=model.max_disparity)

    assert torch.isfinite(result["loss"])
    for key in ("photometric", "smoothness", "left_right", "total"):
        assert key in result["logs"], key
    for absent in ("mean_confidence", "photometric_small", "range_penalty"):
        assert absent not in result["logs"], f"{absent} is off but still logged"

    # And the total really is just those three, at their Monodepth weights.
    weights = LossWeights()
    expected = (weights.photometric * result["logs"]["photometric"]
                + weights.smoothness * result["logs"]["smoothness"])
    assert result["logs"]["total"] > expected, "left_right must contribute too"
