"""Are a dataset's disparity labels consistent with its images?"""

import pytest
import torch


def _labelled_pair(shift=8.0, size=(64, 128), scale_labels=1.0):
    """A pair whose left view IS the right view shifted by `shift`."""
    import torch.nn.functional as F

    from stereo.geometry import warp_right_to_left

    torch.manual_seed(0)
    right = F.avg_pool2d(torch.rand(1, 3, *size), 3, stride=1, padding=1)
    disparity = torch.full((1, 1, *size), shift)
    left, valid = warp_right_to_left(right, disparity)
    return {"left": left[0], "right": right[0],
            "disparity_gt": (disparity * scale_labels)[0],
            "valid_gt_mask": valid[0]}


def test_label_scale_check_accepts_consistent_labels():
    from stereo.data import check_label_scale

    report = check_label_scale([_labelled_pair() for _ in range(3)])
    assert report.consistent and report.best_factor == 1.0, str(report)


def test_label_scale_check_catches_a_mirror_that_forgot_to_rescale():
    """The failure it exists for: images resized, disparity copied through.

    Labels wrong by a constant factor are silent -- supervised training fits them
    and the model looks broken instead of the data.
    """
    from stereo.data import check_label_scale

    report = check_label_scale([_labelled_pair(shift=8.0, scale_labels=4.0) for _ in range(3)])
    assert not report.consistent
    assert report.best_factor == pytest.approx(0.25, rel=0.4), str(report)


def test_resizing_rescales_disparity_because_it_is_a_length():
    """cv2.resize resamples but does not rescale. A 100 px disparity at width
    1242 is 51.5 px at width 640, and getting this wrong trains the model against
    a target that is silently ~2x too large."""
    import numpy as np

    from stereo.data.augmentation import ResizeConfig, ResizeSample

    resize = ResizeSample(ResizeConfig(width=640, preserve_aspect=True))
    out = resize({"left": np.zeros((375, 1242, 3), np.float32),
                  "right": np.zeros((375, 1242, 3), np.float32),
                  "disparity_gt": np.full((375, 1242), 100.0, np.float32),
                  "valid_gt_mask": np.ones((375, 1242), np.float32)})
    assert out["disparity_gt"].mean() == pytest.approx(100.0 * 640 / 1242, rel=1e-3)
    assert out["valid_gt_mask"].shape == out["disparity_gt"].shape


def test_a_batch_may_mix_samples_with_and_without_labels():
    """Collate takes the union of keys, so a dataset missing ground truth does
    not break a batch -- it gets an all-zero valid mask."""
    from stereo.data.base import collate_samples

    with_label = _labelled_pair()
    with_label["metadata"] = {}
    without = {"left": torch.rand(3, 64, 128), "right": torch.rand(3, 64, 128), "metadata": {}}
    batch = collate_samples([with_label, without, with_label])
    assert [float(batch["valid_gt_mask"][i].mean()) > 0 for i in range(3)] == [True, False, True]


def test_middlebury_finds_disparity_under_either_release_naming(tmp_path):
    """The releases disagree on the filename and a mirror may use either.

    MiddEval3 ships disp0GT.pfm, the 2014 full release ships disp0.pfm.
    Hard-coding one made a mirror of the other load with no labels at all, which
    supervised training can only report as 'no disparity: unusable' -- a correct
    message about an avoidable problem.
    """
    import cv2
    import numpy as np

    from stereo.data import DatasetMode, build_dataset
    from stereo.data.registry import DatasetSpec

    def write_scene(root, disparity_name):
        scene = tmp_path / root / "Scene"
        scene.mkdir(parents=True)
        image = (np.random.default_rng(0).random((32, 48, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(scene / "im0.png"), image)
        cv2.imwrite(str(scene / "im1.png"), image)
        with open(scene / disparity_name, "wb") as handle:
            handle.write(b"Pf\n48 32\n-1.0\n")
            handle.write(np.full((32, 48), 7.0, dtype="<f4")[::-1].tobytes())
        return str(tmp_path / root)

    for release, name in (("eval3", "disp0GT.pfm"), ("full2014", "disp0.pfm")):
        root = write_scene(release, name)
        dataset = build_dataset(DatasetSpec(type="middlebury", root=root),
                                DatasetMode.TRAIN, None, with_labels=True)
        assert "disparity_gt" in dataset[0], f"{name} was not found"


def test_middlebury_says_what_it_looked_for_when_there_is_no_disparity(tmp_path):
    import cv2
    import numpy as np
    import pytest

    from stereo.data import DatasetMode, build_dataset
    from stereo.data.registry import DatasetSpec

    scene = tmp_path / "Scene"
    scene.mkdir(parents=True)
    image = (np.random.default_rng(0).random((32, 48, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(scene / "im0.png"), image)
    cv2.imwrite(str(scene / "im1.png"), image)

    with pytest.raises(RuntimeError, match="views but no ground truth") as error:
        build_dataset(DatasetSpec(type="middlebury", root=str(tmp_path)),
                      DatasetMode.BENCHMARK, None)
    assert "disp0GT.pfm" in str(error.value)


def test_a_spec_divides_into_disjoint_train_and_val_parts(labelled_dataset):
    """Validation must score only what training never sees. Outside Middlebury
    the val part is a contiguous block at the end, which keeps a sequence's
    neighbouring -- nearly identical -- frames on one side."""
    from stereo.data import DatasetMode
    from stereo.data.registry import DatasetSpec, build_training_datasets

    def indices(part):
        spec = DatasetSpec(type="folder", root=labelled_dataset, part=part, val_fraction=1 / 3)
        dataset, _, _ = build_training_datasets([spec], DatasetMode.VALIDATION, None, None,
                                                with_labels=True)
        return [dataset[i]["metadata"]["index"] for i in range(len(dataset))]

    train, val, everything = indices("train"), indices("val"), indices("all")
    assert sorted(train + val) == everything and not set(train) & set(val)
    assert val == everything[-len(val):] and len(val) == 2
