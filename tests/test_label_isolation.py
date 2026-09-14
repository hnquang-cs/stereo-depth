"""The label-leakage tests.

These are the tests that make the project's central claim checkable: ground
truth cannot influence training.  They work at three levels.

1. *Structural* -- a dataset in a training mode refuses to emit ground-truth
   keys, and the guard that the training loop calls rejects any batch carrying
   them.
2. *Behavioural* -- corrupting the ground-truth files on disk changes nothing
   about the training loss, the gradients, the teacher output or the
   pseudo-labels for a fixed batch.
3. *Static* -- no module reachable from the training path imports the
   evaluation package.
"""

import importlib
import os

import cv2
import numpy as np
import pytest
import torch

from stereo.data import DatasetMode, StereoFolderDataset, assert_label_free, collate_samples
from stereo.data.base import GROUND_TRUTH_KEYS


@pytest.fixture
def labelled_dataset(tmp_path):
    """A folder dataset that HAS ground truth, so leakage would be possible."""
    root = tmp_path / "data"
    for name in ("left", "right", "left_disparity"):
        (root / name).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(3):
        stem = f"{index:06d}"
        texture = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(root / "left" / f"{stem}.png"), texture)
        cv2.imwrite(str(root / "right" / f"{stem}.png"), np.roll(texture, -5, axis=1))
        np.savez(str(root / "left_disparity" / f"{stem}.npz"),
                 np.full((64, 96), 5.0, dtype=np.float32))
    return str(root)


def test_training_mode_never_returns_ground_truth(labelled_dataset):
    for mode in (DatasetMode.TRAIN, DatasetMode.VALIDATION):
        dataset = StereoFolderDataset(labelled_dataset, mode=mode)
        sample = dataset[0]
        for key in GROUND_TRUTH_KEYS:
            assert key not in sample, f"{key} leaked in mode {mode}"
        assert {"left", "right", "metadata"} <= set(sample)


def test_benchmark_mode_does_return_ground_truth(labelled_dataset):
    dataset = StereoFolderDataset(labelled_dataset, mode=DatasetMode.BENCHMARK)
    sample = dataset[0]
    assert "disparity_gt" in sample and "valid_gt_mask" in sample
    assert float(sample["disparity_gt"].mean()) == pytest.approx(5.0)


def test_assert_label_free_rejects_every_ground_truth_key():
    assert_label_free({"left": 1, "right": 2})
    for key in GROUND_TRUTH_KEYS:
        with pytest.raises(RuntimeError, match="ground-truth"):
            assert_label_free({"left": 1, "right": 2, key: 3})


def test_benchmark_datasets_refuse_augmentation(labelled_dataset):
    with pytest.raises(ValueError, match="must not be augmented"):
        StereoFolderDataset(labelled_dataset, mode=DatasetMode.BENCHMARK, transform=lambda s: s)


def _training_signature(dataset_root, seed=0):
    """Loss, gradient norm, teacher output and pseudo-mask for one fixed batch."""
    from stereo.config import LossWeights, TeacherConfig
    from stereo.model import StereoNet, StereoNetConfig
    from stereo.training import EmaTeacher, LabelFreeObjective, ObjectiveState

    torch.manual_seed(seed)
    dataset = StereoFolderDataset(dataset_root, mode=DatasetMode.TRAIN)
    batch = collate_samples([dataset[i] for i in range(2)])
    assert_label_free(batch)

    model = StereoNet(StereoNetConfig.for_width(96, downsample=4, backbone_width=4,
                                                feature_channels=4))
    teacher = EmaTeacher(model, 0.999)
    objective = LabelFreeObjective(LossWeights(), TeacherConfig())

    images = {"left": batch["left_clean"] if "left_clean" in batch else batch["left"],
              "right": batch["right_clean"] if "right_clean" in batch else batch["right"]}
    teacher_outputs = teacher.predict(images["left"], images["right"])
    outputs = model(batch["left"], batch["right"], directions=("left", "right"))
    result = objective(outputs, images, ObjectiveState(pseudo_scale=1.0, warmup_scale=1.0),
                       teacher_outputs, max_disparity=model.max_disparity)
    result["loss"].backward()

    gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    mask = objective._teacher_mask(teacher_outputs["left"], teacher_outputs, images, "left",
                                   model.max_disparity)
    return {
        "loss": float(result["loss"].detach()),
        "grad_norm": float(gradient.norm()),
        "teacher_disparity": teacher_outputs["left"]["disparity"].clone(),
        "pseudo_mask": mask.clone(),
    }


def test_corrupting_ground_truth_cannot_change_training(labelled_dataset):
    """The behavioural leakage test.

    Compute the full training signature, then overwrite every ground-truth file
    with nonsense, then compute it again.  Anything that changes would mean a
    ground-truth file reached the optimisation path.
    """
    before = _training_signature(labelled_dataset)

    disparity_dir = os.path.join(labelled_dataset, "left_disparity")
    for name in os.listdir(disparity_dir):
        np.savez(os.path.join(disparity_dir, name),
                 np.full((64, 96), -999.0, dtype=np.float32))
    # Also verify the benchmark path *does* see the change, so the test is not
    # passing simply because nothing reads these files at all.
    benchmark = StereoFolderDataset(labelled_dataset, mode=DatasetMode.BENCHMARK)
    assert float(benchmark[0]["disparity_gt"].mean()) != pytest.approx(5.0)

    after = _training_signature(labelled_dataset)

    assert before["loss"] == pytest.approx(after["loss"], rel=1e-9)
    assert before["grad_norm"] == pytest.approx(after["grad_norm"], rel=1e-9)
    assert torch.equal(before["teacher_disparity"], after["teacher_disparity"])
    assert torch.equal(before["pseudo_mask"], after["pseudo_mask"])


def test_deleting_ground_truth_does_not_break_training(labelled_dataset):
    """Training must work on a dataset that has only left/ and right/."""
    import shutil
    shutil.rmtree(os.path.join(labelled_dataset, "left_disparity"))
    signature = _training_signature(labelled_dataset)
    assert np.isfinite(signature["loss"])
    assert signature["grad_norm"] > 0


def test_training_modules_do_not_import_evaluation():
    """Static check: nothing on the training path pulls in the ground-truth code."""
    training_modules = [
        "stereo.training.loop", "stereo.training.objective", "stereo.training.teacher",
        "stereo.losses.photometric", "stereo.losses.smoothness", "stereo.losses.consistency",
        "stereo.losses.pseudo_label", "stereo.losses.confidence",
        "stereo.model.stereo_net", "stereo.geometry",
    ]
    for name in training_modules:
        module = importlib.import_module(name)
        source = open(module.__file__).read()
        for forbidden in ("from ..evaluation", "from stereo.evaluation", "import stereo.evaluation"):
            assert forbidden not in source, f"{name} imports the evaluation package ({forbidden})"


def test_no_supervised_loss_exists():
    """There is no ground-truth loss class anywhere in stereo/losses."""
    import stereo.losses as losses
    forbidden = {"DisparityLoss", "SupervisedLoss", "NsceLoss", "DepthLoss"}
    assert forbidden.isdisjoint(dir(losses))

    loss_dir = os.path.dirname(losses.__file__)
    for filename in os.listdir(loss_dir):
        if not filename.endswith(".py"):
            continue
        source = open(os.path.join(loss_dir, filename)).read()
        for token in ("disparity_gt", "depth_gt", "valid_gt_mask"):
            assert token not in source, f"{filename} references {token}"


def test_static_audit_script_passes():
    """The repository-wide audit must pass as part of the test suite."""
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, os.path.join(root, "scripts", "audit_label_leakage.py"), "--strict"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout
    # The audit must actually be looking at the training path, not vacuously passing.
    assert "FORBIDDEN" in result.stdout
