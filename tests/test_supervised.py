"""Semi-supervised training: labelled and unlabelled samples in one batch.

KITTI odometry has no disparity; FlyingThings3D and Middlebury do. Rather than
choose, the loss adapts per sample -- the supervised terms are masked means over
labelled pixels, so an unlabelled sample carries an all-zero mask and
contributes nothing to them while still contributing to the Monodepth terms.
"""

import numpy as np
import pytest
import torch

from stereo.config import LossWeights
from stereo.data.base import collate_samples
from stereo.losses import DisparityLoss, NsceLoss, labels_from_batch
from stereo.model import StereoNet, StereoNetConfig
from stereo.training import LabelFreeObjective, ObjectiveState


def _sample(labelled, height=32, width=64, disparity=8.0):
    sample = {"left": torch.rand(3, height, width), "right": torch.rand(3, height, width),
              "metadata": {}}
    if labelled:
        sample["disparity_gt"] = torch.full((1, height, width), disparity)
        sample["valid_gt_mask"] = torch.ones(1, height, width)
    return sample


# -- the losses ------------------------------------------------------------- #

def test_supervised_loss_ignores_unlabelled_samples():
    loss = DisparityLoss()
    predicted = torch.full((2, 1, 16, 32), 10.0)
    target = torch.full((2, 1, 16, 32), 14.0)
    mask = torch.ones_like(target)
    mask[1] = 0.0                                    # sample 1 has no label

    terms = loss(predicted, target, mask)
    assert float(terms["epe"]) == pytest.approx(4.0)
    # A masked MEAN, so the unlabelled half must not dilute it towards zero.
    assert float(loss(predicted, target, torch.ones_like(mask))["epe"]) == pytest.approx(4.0)
    assert float(loss(predicted, target, torch.zeros_like(mask))["loss"]) == 0.0


def test_nsce_target_peaks_at_the_true_disparity():
    """The whole point of NSCE: the cost volume is pushed to a single peak in the
    right bin, which the soft-argmin gradient alone cannot express."""
    loss = NsceLoss()
    bins, downsample = 16, 4
    target = torch.full((1, 1, 8, 8), 5.0 * downsample)    # bin 5
    mask = torch.ones_like(target)

    right = torch.full((1, bins, 8, 8), 10.0)
    right[:, 5] = 0.0                                       # peaked at the truth
    wrong = torch.full((1, bins, 8, 8), 10.0)
    wrong[:, 12] = 0.0                                      # peaked elsewhere
    flat = torch.zeros(1, bins, 8, 8)

    scores = {name: float(loss(volume, target, mask, downsample)["loss"])
              for name, volume in (("right", right), ("flat", flat), ("wrong", wrong))}
    assert scores["right"] < scores["flat"] < scores["wrong"], scores


def test_nsce_skips_disparities_outside_the_search_range():
    """Outside the range the Laplacian is truncated and the target is a fiction."""
    loss = NsceLoss()
    cost = torch.rand(1, 8, 4, 4)
    beyond = torch.full((1, 1, 4, 4), 500.0)
    terms = loss(cost, beyond, torch.ones_like(beyond), 4)
    assert float(terms["in_range_ratio"]) == 0.0
    assert float(terms["loss"]) == 0.0


# -- the data path ---------------------------------------------------------- #

def test_resizing_rescales_disparity_because_it_is_a_length():
    """The trap: cv2.resize resamples but does not rescale. A 100 px disparity at
    width 1242 is 51.5 px at width 640, and getting this wrong trains the model
    against a target that is silently ~2x too large."""
    from stereo.data.augmentation import ResizeConfig, ResizeSample

    resize = ResizeSample(ResizeConfig(width=640, preserve_aspect=True))
    out = resize({"left": np.zeros((375, 1242, 3), np.float32),
                  "right": np.zeros((375, 1242, 3), np.float32),
                  "disparity_gt": np.full((375, 1242), 100.0, np.float32),
                  "valid_gt_mask": np.ones((375, 1242), np.float32)})
    assert out["disparity_gt"].mean() == pytest.approx(100.0 * 640 / 1242, rel=1e-3)
    assert out["valid_gt_mask"].shape == out["disparity_gt"].shape


def test_a_batch_may_mix_labelled_and_unlabelled_samples():
    batch = collate_samples([_sample(True), _sample(False), _sample(True)])
    assert "disparity_gt" in batch
    assert [float(batch["valid_gt_mask"][i].mean()) for i in range(3)] == [1.0, 0.0, 1.0]

    assert labels_from_batch(collate_samples([_sample(False)])) is None
    assert labels_from_batch(batch) is not None


# -- end to end ------------------------------------------------------------- #

#: Supervised weights, stated explicitly: the DEFAULT objective is label-free,
#: so these tests must turn the terms on rather than assume them.
SUPERVISED = LossWeights(supervised=1.0, nsce=0.2)


def _objective_run(weights, labels):
    torch.manual_seed(0)
    model = StereoNet(StereoNetConfig.for_width(96, downsample=4, backbone_width=4,
                                                feature_channels=4))
    left, right = torch.rand(2, 3, 32, 96), torch.rand(2, 3, 32, 96)
    outputs = model(left, right, directions=("left", "right"))
    return LabelFreeObjective(weights)(outputs, {"left": left, "right": right},
                                       ObjectiveState(warmup_scale=1.0),
                                       max_disparity=model.max_disparity, labels=labels)


def test_the_objective_reports_supervised_terms_only_when_labels_are_present():
    labels = {"disparity_gt": torch.full((2, 1, 32, 96), 6.0),
              "valid_gt_mask": torch.ones(2, 1, 32, 96)}
    supervised = _objective_run(SUPERVISED, labels)["logs"]
    for key in ("supervised", "epe", "nsce", "labelled_ratio"):
        assert key in supervised, key

    unlabelled = _objective_run(SUPERVISED, None)["logs"]
    for key in ("supervised", "epe", "nsce"):
        assert key not in unlabelled, f"{key} reported without labels"
    # The Monodepth terms apply to every sample, labelled or not -- which is how
    # an unlabelled dataset such as KITTI odometry still trains the model.
    for key in ("photometric", "smoothness", "left_right"):
        assert key in unlabelled, key


def test_an_all_unlabelled_batch_contributes_nothing_supervised():
    labels = {"disparity_gt": torch.zeros(2, 1, 32, 96),
              "valid_gt_mask": torch.zeros(2, 1, 32, 96)}
    logs = _objective_run(SUPERVISED, labels)["logs"]
    assert float(logs["labelled_ratio"]) == 0.0
    assert float(logs["supervised"]) == 0.0


# -- the three requested changes -------------------------------------------- #

def test_nsce_is_normalised_by_a_uniform_prediction():
    """~1 when the cost volume knows nothing, ~0 when it is right, so the weight
    means the same thing at any search range."""
    import math

    loss = NsceLoss()
    for bins in (8, 16, 64):
        uniform = torch.zeros(1, bins, 4, 4)
        target = torch.full((1, 1, 4, 4), 2.0 * 4)
        value = float(loss(uniform, target, torch.ones_like(target), 4)["loss"])
        assert 0.8 < value < 1.2, f"{bins} bins: {value}"

    # A correct volume scores well below uniform, but NOT ~0: the target is a
    # Laplacian (lambda 0.3), not one-hot, so the floor is its own entropy --
    # roughly 0.1 normalised. Asserting 0 would be asserting the wrong minimum.
    target = torch.full((1, 1, 4, 4), 2.0 * 4)
    peaked = torch.full((1, 16, 4, 4), 20.0)
    peaked[:, 2] = 0.0
    right = float(loss(peaked, target, torch.ones_like(target), 4)["loss"])
    wrong = torch.full((1, 16, 4, 4), 20.0)
    wrong[:, 11] = 0.0
    assert right < 0.6 < float(loss(wrong, target, torch.ones_like(target), 4)["loss"])


def test_smoothness_applies_only_where_there_is_no_label():
    """Smoothness is a prior standing in for supervision. Where ground truth
    exists it constrains the field better, and the prior only biases it."""
    from stereo.losses import SmoothnessLoss

    torch.manual_seed(0)
    disparity = torch.rand(1, 1, 16, 32) * 10
    image = torch.rand(1, 3, 16, 32)
    smoothness = SmoothnessLoss()

    everywhere = float(smoothness(disparity, image, None))
    half = torch.ones(1, 1, 16, 32)
    half[..., :8, :] = 0.0                       # top half is labelled
    gated = float(smoothness(disparity, image, half))
    assert gated != everywhere
    # Fully labelled: the prior contributes nothing at all.
    assert float(smoothness(disparity, image, torch.zeros_like(half))) == 0.0


def test_the_objective_reports_weighted_contributions_that_sum_to_the_loss():
    """Raw values do not say what the optimiser follows; contributions do."""
    labels = {"disparity_gt": torch.full((2, 1, 32, 96), 6.0),
              "valid_gt_mask": torch.ones(2, 1, 32, 96)}
    result = _objective_run(SUPERVISED, labels)
    parts = result["parts"]
    assert parts, "no contributions recorded"
    assert sum(parts.values()) == pytest.approx(float(result["loss"]), rel=1e-4)
    for name in ("photo", "lr", "sL1", "nsce"):
        assert name in parts, name
    # Mirrored into logs so the trainer can print them without the tensor graph.
    for name, value in parts.items():
        assert result["logs"][f"part/{name}"] == pytest.approx(value)


def test_the_default_objective_reads_no_labels_at_all():
    """The default is label-free: supervised terms off, and labels ignored even
    when a batch carries them."""
    weights = LossWeights()
    assert weights.supervised == 0.0 and weights.nsce == 0.0
    assert not weights.uses_labels

    labels = {"disparity_gt": torch.full((2, 1, 32, 96), 6.0),
              "valid_gt_mask": torch.ones(2, 1, 32, 96)}
    logs = _objective_run(weights, labels)["logs"]
    for key in ("supervised", "epe", "nsce", "labelled_ratio"):
        assert key not in logs, f"{key} reported by a label-free objective"
