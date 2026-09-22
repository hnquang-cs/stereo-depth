"""Teacher/student mechanics and an end-to-end training run on synthetic data."""

import cv2
import numpy as np
import pytest
import torch

from stereo.config import Config, LossWeights, TeacherConfig
from stereo.data import DatasetMode, StereoFolderDataset, collate_samples
from stereo.model import StereoNet, StereoNetConfig
from stereo.training import EmaTeacher, LabelFreeObjective, ObjectiveState, pseudo_label_weight


def tiny_model(width=96):
    return StereoNet(StereoNetConfig.for_width(width, downsample=4, backbone_width=4,
                                               feature_channels=4))


# --------------------------------------------------------------------------- #
# EMA teacher
# --------------------------------------------------------------------------- #

def test_ema_teacher_starts_as_a_copy_and_moves_toward_the_student():
    student = tiny_model()
    teacher = EmaTeacher(student, decay=0.9)
    for a, b in zip(student.parameters(), teacher.model.parameters()):
        assert torch.equal(a, b)

    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(1.0)
    before = [p.clone() for p in teacher.model.parameters()]
    teacher.update(student)

    for old, new, target in zip(before, teacher.model.parameters(), student.parameters()):
        expected = 0.9 * old + 0.1 * target
        assert torch.allclose(new, expected, atol=1e-6)


def test_ema_teacher_has_no_gradients():
    student = tiny_model()
    teacher = EmaTeacher(student)
    assert all(not p.requires_grad for p in teacher.model.parameters())

    left, right = torch.rand(1, 3, 32, 96), torch.rand(1, 3, 32, 96)
    outputs = teacher.predict(left, right)
    for output in outputs.values():
        for tensor in output.values():
            assert not tensor.requires_grad
            assert tensor.grad_fn is None


def test_ema_teacher_averages_batchnorm_buffers():
    student = tiny_model()
    teacher = EmaTeacher(student, decay=0.5)
    buffers = dict(student.named_buffers())
    name = next(n for n, b in buffers.items() if "running_mean" in n)
    with torch.no_grad():
        buffers[name].fill_(4.0)
        dict(teacher.model.named_buffers())[name].fill_(0.0)
    teacher.update(student)
    assert float(dict(teacher.model.named_buffers())[name].mean()) == pytest.approx(2.0)


def test_teacher_does_not_receive_gradient_from_the_loss():
    student = tiny_model()
    teacher = EmaTeacher(student)
    objective = LabelFreeObjective(LossWeights(), TeacherConfig())

    left, right = torch.rand(2, 3, 32, 96), torch.rand(2, 3, 32, 96)
    teacher_outputs = teacher.predict(left, right)
    student_outputs = student(left, right, directions=("left", "right"))
    result = objective(student_outputs, {"left": left, "right": right},
                       ObjectiveState(pseudo_scale=1.0, warmup_scale=1.0),
                       teacher_outputs, max_disparity=student.max_disparity)
    result["loss"].backward()

    assert all(p.grad is None for p in teacher.model.parameters()), "the teacher got gradients"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_pseudo_label_ramp():
    assert pseudo_label_weight(0, start_epoch=10, ramp_epochs=5) == 0.0
    assert pseudo_label_weight(9, start_epoch=10, ramp_epochs=5) == 0.0
    assert pseudo_label_weight(10, start_epoch=10, ramp_epochs=5) == pytest.approx(0.2)
    assert pseudo_label_weight(14, start_epoch=10, ramp_epochs=5) == pytest.approx(1.0)
    assert pseudo_label_weight(50, start_epoch=10, ramp_epochs=5) == 1.0
    assert pseudo_label_weight(10, start_epoch=10, ramp_epochs=0) == 1.0


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #

def test_objective_runs_without_a_teacher_and_logs_the_expected_keys():
    model = tiny_model()
    objective = LabelFreeObjective(LossWeights(), TeacherConfig())
    left, right = torch.rand(2, 3, 32, 96), torch.rand(2, 3, 32, 96)
    outputs = model(left, right, directions=("left", "right"))
    result = objective(outputs, {"left": left, "right": right},
                       ObjectiveState(warmup_scale=1.0, pseudo_scale=0.0),
                       None, max_disparity=model.max_disparity)

    assert torch.isfinite(result["loss"])
    for key in ("photometric", "photometric_ssim", "photometric_l1", "smoothness", "left_right",
                "pseudo_valid_ratio", "valid_warp_ratio", "disparity_mean", "disparity_max",
                "mean_confidence", "total"):
        assert key in result["logs"], key
    assert result["logs"]["pseudo_valid_ratio"] == 0.0


def test_objective_requires_no_ground_truth_argument():
    """The objective's signature has nowhere to put a label."""
    import inspect
    parameters = set(inspect.signature(LabelFreeObjective.__call__).parameters)
    # valid_mask says which rows of a ragged, aspect-preserving batch are real
    # padding and which are image. It comes from image SHAPES, never from
    # disparity, so it carries no label. Any further parameter must be justified
    # the same way before being added here.
    assert parameters == {"self", "student_outputs", "images", "state", "teacher_outputs",
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
    objective = LabelFreeObjective(LossWeights(pseudo=0.0), TeacherConfig(enabled=False))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    images = {"left": batch["left"], "right": batch["right"]}
    losses = []
    for _ in range(12):
        outputs = model(batch["left"], batch["right"], directions=("left", "right"))
        result = objective(outputs, images, ObjectiveState(warmup_scale=1.0), None,
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
    config.teacher = TeacherConfig(enabled=True, start_epoch=0, ramp_epochs=1)
    config.training.epochs = 2
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.use_amp = False
    config.training.warmup_iterations = 1
    config.training.output_dir = str(tmp_path / "out")
    config.optimizer.warmup_iterations = 1

    trainer = Trainer(config, device=torch.device("cpu"))
    best = trainer.fit()

    assert trainer.teacher is not None, "the EMA teacher never started"
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
    config.teacher.enabled = False

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
    from stereo.config import LossWeights, TeacherConfig
    from stereo.training import LabelFreeObjective, ObjectiveState

    objective = LabelFreeObjective(LossWeights(range_penalty=1.0), TeacherConfig(enabled=False))
    left, right = torch.rand(1, 3, 32, 96), torch.rand(1, 3, 32, 96)
    images = {"left": left, "right": right}

    def penalty_for(value):
        outputs = {d: {"disparity": torch.full((1, 1, 32, 96), value),
                       "disparity_small": torch.full((1, 1, 8, 24), value / 4),
                       "matchability": torch.full((1, 1, 8, 24), -0.1)}
                   for d in ("left", "right")}
        result = objective(outputs, images, ObjectiveState(warmup_scale=1.0), None,
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
    config.teacher.enabled = False

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
    config.teacher.enabled = False

    Trainer(config, device=torch.device("cpu")).fit()
    assert not (tmp_path / "out" / "visualizations").exists()
