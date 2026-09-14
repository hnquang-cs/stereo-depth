"""Scene Flow layout discovery.

Scene Flow is 132 GB, so on Kaggle it is attached as a community mirror rather
than downloaded -- and those mirrors do not agree on a directory layout. These
tests build each plausible structure synthetically and check that the loader
finds it without being told where to look.
"""

import os

import cv2
import numpy as np
import pytest

from stereo.data import DatasetMode
from stereo.data.sceneflow import SceneFlowDataset, describe_tree, discover_sceneflow


def _png(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (np.random.default_rng(0).random((32, 48, 3)) * 255).astype(np.uint8))


def _pfm(path, value=7.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.full((32, 48), value, dtype=np.float32)
    with open(path, "wb") as handle:
        handle.write(b"Pf\n48 32\n-1.0\n")
        handle.write(np.flipud(data).astype("<f4").tobytes())


def build_official(root, split="TRAIN", pass_name="finalpass", with_disparity=True, prefix=""):
    base = os.path.join(root, prefix) if prefix else root
    for view in ("left", "right"):
        _png(os.path.join(base, f"frames_{pass_name}", split, "A", "0000", view, "0006.png"))
        if with_disparity:
            _pfm(os.path.join(base, "disparity", split, "A", "0000", view, "0006.pfm"))
    return root


def build_subset(root, split="train", with_disparity=True, prefix="FlyingThings3D_subset"):
    base = os.path.join(root, prefix) if prefix else root
    for view in ("left", "right"):
        for index in range(3):
            _png(os.path.join(base, split, "image_clean", view, f"{index:07d}.png"))
            if with_disparity:
                _pfm(os.path.join(base, split, "disparity", view, f"{index:07d}.pfm"))
    return root


def test_discovers_official_layout_at_the_root(tmp_path):
    root = build_official(str(tmp_path))
    layout = discover_sceneflow(root, "TRAIN")
    assert layout.kind == "official"
    assert layout.pass_name == "finalpass"
    assert layout.split == "TRAIN"
    assert layout.disparity_dir is not None


@pytest.mark.parametrize("prefix", ["sceneflow", "sceneflow/FlyingThings3D",
                                    "a/b/c/FlyingThings3D"])
def test_discovers_official_layout_when_nested(tmp_path, prefix):
    """Kaggle mirrors routinely wrap the tree in one or more extra folders."""
    root = build_official(str(tmp_path), prefix=prefix)
    layout = discover_sceneflow(root, "TRAIN")
    assert layout.kind == "official"
    assert layout.frames_dir.endswith(os.path.join("frames_finalpass", "TRAIN"))


def test_discovers_subset_layout_and_maps_test_to_val(tmp_path):
    """The subset release is flat and calls its held-out split 'val'."""
    build_subset(str(tmp_path), split="train")
    build_subset(str(tmp_path), split="val")

    train = discover_sceneflow(str(tmp_path), "TRAIN")
    assert train.kind == "subset" and train.split == "train"

    test = discover_sceneflow(str(tmp_path), "TEST")
    assert test.kind == "subset" and test.split == "val"


def test_falls_back_to_cleanpass_when_finalpass_is_absent(tmp_path):
    root = build_official(str(tmp_path), pass_name="cleanpass")
    assert discover_sceneflow(root, "TRAIN").pass_name == "cleanpass"


def test_images_only_mirror_trains_but_refuses_to_benchmark(tmp_path):
    """Label-free training needs no disparity; benchmarking must say so clearly."""
    root = build_official(str(tmp_path), with_disparity=False)
    layout = discover_sceneflow(root, "TRAIN")
    assert layout.disparity_dir is None

    dataset = SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.TRAIN)
    assert len(dataset) == 1
    assert "disparity_gt" not in dataset[0]

    with pytest.raises(FileNotFoundError, match="no disparity tree"):
        SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.BENCHMARK)


@pytest.mark.parametrize("builder,split", [(build_official, "TRAIN"), (build_subset, "TRAIN")])
def test_dataset_loads_both_layouts_in_both_modes(tmp_path, builder, split):
    root = builder(str(tmp_path))
    training = SceneFlowDataset(root, split=split, mode=DatasetMode.TRAIN)
    sample = training[0]
    assert sample["left"].shape == (3, 32, 48)
    assert not any(k in sample for k in ("disparity_gt", "valid_gt_mask"))

    benchmark = SceneFlowDataset(root, split=split, mode=DatasetMode.BENCHMARK)
    assert float(benchmark[0]["disparity_gt"].mean()) == pytest.approx(7.0)


def test_unrecognised_mirror_reports_what_it_actually_found(tmp_path):
    """The failure must be diagnosable without guessing."""
    _png(str(tmp_path / "some_random_folder" / "a.png"))
    with pytest.raises(FileNotFoundError) as error:
        discover_sceneflow(str(tmp_path), "TRAIN")
    message = str(error.value)
    assert "frames_finalpass" in message and "image_clean" in message
    assert "some_random_folder" in message, "the error must show the real tree"


def test_describe_tree_is_bounded(tmp_path):
    for i in range(40):
        _png(str(tmp_path / f"dir{i:03d}" / "x.png"))
    text = describe_tree(str(tmp_path), max_depth=2, max_entries=5)
    assert "and 35 more directories" in text
    assert len(text.splitlines()) < 30


def test_only_frames_present_in_both_views_are_indexed(tmp_path):
    root = build_official(str(tmp_path))
    # An extra left frame with no right partner must be skipped, not crash.
    _png(str(tmp_path / "frames_finalpass" / "TRAIN" / "A" / "0000" / "left" / "9999.png"))
    assert len(SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.TRAIN)) == 1
