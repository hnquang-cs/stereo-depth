"""Scene Flow layout discovery.

Scene Flow is 132 GB, so on Kaggle it is attached as a community mirror rather
than downloaded -- and those mirrors do not agree on a directory layout. These
tests build each structure seen in the wild and check that the loader finds it
without being told where to look.

Discovery is deliberately *structural*: it locates an image-pass directory and
then indexes every directory beneath it holding a ``left``/``right`` pair, at any
depth. That is what lets one code path cover official FlyingThings3D
(``TRAIN/<A|B|C>/<scene>/left``), Monkaa (``<scene>/left``), Driving
(``<focallength>/<direction>/<speed>/left``), the flat subset release, and
mirrors that nest or flatten any of them.
"""

import os

import cv2
import numpy as np
import pytest

from stereo.data import DatasetMode
from stereo.data.sceneflow import (SceneFlowDataset, describe_tree, discover_sceneflow,
                                   find_scene_dirs)


def _png(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (np.random.default_rng(0).random((32, 48, 3)) * 255).astype(np.uint8))


def _pfm(path, value=7.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.full((32, 48), value, dtype=np.float32)
    with open(path, "wb") as handle:
        handle.write(b"Pf\n48 32\n-1.0\n")
        handle.write(np.flipud(data).astype("<f4").tobytes())


def build(root, scenes, pass_dir="frames_cleanpass", prefix="", disparity=True,
          disparity_dir="disparity"):
    """Create ``<root>/<prefix>/<pass_dir>/<scene>/left|right/0006.png`` for each scene."""
    base = os.path.join(root, prefix) if prefix else root
    for scene in scenes:
        for view in ("left", "right"):
            _png(os.path.join(base, pass_dir, scene, view, "0006.png"))
            if disparity:
                _pfm(os.path.join(base, disparity_dir, scene, view, "0006.pfm"))
    return root


def test_official_layout_with_a_train_directory(tmp_path):
    build(str(tmp_path), ["TRAIN/A/0000"], pass_dir="frames_finalpass")
    layout = discover_sceneflow(str(tmp_path), "TRAIN")
    assert layout.split == "TRAIN" and layout.has_split
    assert layout.pass_name == "frames_finalpass"


def test_the_reported_kaggle_mirror(tmp_path):
    """arthurthom/sceneflow: double-nested, cleanpass only, no TRAIN level.

    This is the exact shape that made the previous name-matching discovery fail.
    """
    root = str(tmp_path)
    build(root, ["A/0000", "A/0001", "B/0000"], prefix="FlyingThings3D/FlyingThings3D")
    build(root, ["15mm_focallength/scene_backwards/fast"], prefix="Driving/Driving")
    build(root, ["eating_x2"], prefix="Monkaa/Monkaa")
    os.makedirs(os.path.join(root, "kitti2015", "training", "image_2"), exist_ok=True)

    layout = discover_sceneflow(root, "TRAIN")
    assert layout.subset == "FlyingThings3D", "FlyingThings3D is the paper's benchmark"
    assert layout.pass_name == "frames_cleanpass"
    assert not layout.has_split
    assert layout.disparity_dir is not None

    dataset = SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.TRAIN)
    assert len(dataset) == 3
    assert not any(k in dataset[0] for k in ("disparity_gt", "valid_gt_mask"))


@pytest.mark.parametrize("subset,scene,depth", [
    ("FlyingThings3D", "A/0000", 2),
    ("Driving", "15mm_focallength/scene_backwards/fast", 3),
    ("Monkaa", "eating_x2", 1),
])
def test_scene_directories_are_found_at_any_depth(tmp_path, subset, scene, depth):
    """Each Scene Flow subset nests its scenes differently; all must index."""
    root = build(str(tmp_path), [scene], prefix=f"{subset}/{subset}")
    dataset = SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.TRAIN, subset=subset)
    assert len(dataset) == 1
    assert dataset.entries[0].relative.count(os.sep) + 1 == depth
    assert dataset.layout.subset == subset


def test_subset_release_layout_and_val_maps_to_test(tmp_path):
    """FlyingThings3D_subset is flat and calls its held-out split 'val'."""
    root = str(tmp_path)
    for split in ("train", "val"):
        base = os.path.join(root, "FlyingThings3D_subset", split)
        for view in ("left", "right"):
            _png(os.path.join(base, "image_clean", view, "0000000.png"))
            _pfm(os.path.join(base, "disparity", view, "0000000.pfm"))

    train = discover_sceneflow(root, "TRAIN")
    assert train.split == "train" and train.has_split
    test = discover_sceneflow(root, "TEST")
    assert test.split == "val" and test.frames_dir != train.frames_dir


def test_finalpass_is_preferred_and_can_be_forced(tmp_path):
    root = str(tmp_path)
    build(root, ["A/0000"], pass_dir="frames_cleanpass")
    build(root, ["A/0000"], pass_dir="frames_finalpass")
    assert discover_sceneflow(root, "TRAIN").pass_name == "frames_finalpass"
    assert discover_sceneflow(root, "TRAIN", pass_name="cleanpass").pass_name == "frames_cleanpass"


def test_flyingthings3d_is_preferred_over_the_other_subsets(tmp_path):
    root = str(tmp_path)
    build(root, ["eating_x2"], prefix="Monkaa/Monkaa")
    build(root, ["A/0000"], prefix="FlyingThings3D/FlyingThings3D")
    assert discover_sceneflow(root, "TRAIN").subset == "FlyingThings3D"


def test_images_only_mirror_trains_but_refuses_to_benchmark(tmp_path):
    """Label-free training needs no disparity; benchmarking must say so clearly."""
    root = build(str(tmp_path), ["TRAIN/A/0000"], disparity=False)
    assert discover_sceneflow(root, "TRAIN").disparity_dir is None
    assert len(SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.TRAIN)) == 1
    with pytest.raises(FileNotFoundError, match="no disparity tree"):
        SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.BENCHMARK)


def test_benchmarking_an_unsplit_mirror_is_refused(tmp_path):
    """Without a train/test division the TEST split IS the training data.

    Scoring a model on its own training frames is not a benchmark, so this must
    fail loudly rather than quietly produce a flattering number.
    """
    root = build(str(tmp_path), ["A/0000"], prefix="FlyingThings3D/FlyingThings3D")
    assert not discover_sceneflow(root, "TEST").has_split

    assert len(SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.TRAIN)) == 1
    with pytest.raises(RuntimeError, match="contaminated by training data"):
        SceneFlowDataset(root, split="TEST", mode=DatasetMode.BENCHMARK)

    # Escape hatch, for a copy the model provably never trained on.
    allowed = SceneFlowDataset(root, split="TEST", mode=DatasetMode.BENCHMARK,
                               allow_unsplit_benchmark=True)
    assert float(allowed[0]["disparity_gt"].mean()) == pytest.approx(7.0)


def test_a_split_tree_does_not_leak_across_splits(tmp_path):
    """Asking for TEST must never silently fall back to TRAIN."""
    root = build(str(tmp_path), ["TRAIN/A/0000"])
    with pytest.raises(FileNotFoundError):
        discover_sceneflow(root, "TEST")


def test_unrecognised_mirror_reports_what_it_actually_found(tmp_path):
    _png(str(tmp_path / "some_random_folder" / "a.png"))
    with pytest.raises(FileNotFoundError) as error:
        discover_sceneflow(str(tmp_path), "TRAIN")
    message = str(error.value)
    assert "frames_" in message and "image_clean" in message
    assert "some_random_folder" in message, "the error must show the real tree"


def test_find_scene_dirs_does_not_descend_into_view_folders(tmp_path):
    root = build(str(tmp_path), ["A/0000"])
    scenes = find_scene_dirs(os.path.join(root, "frames_cleanpass"))
    assert scenes == [os.path.join("A", "0000")]


def test_only_frames_present_in_both_views_are_indexed(tmp_path):
    root = build(str(tmp_path), ["TRAIN/A/0000"])
    _png(os.path.join(root, "frames_cleanpass", "TRAIN", "A", "0000", "left", "9999.png"))
    assert len(SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.TRAIN)) == 1


def test_describe_tree_is_bounded(tmp_path):
    for i in range(40):
        _png(str(tmp_path / f"dir{i:03d}" / "x.png"))
    text = describe_tree(str(tmp_path), max_depth=2, max_entries=5)
    assert "and 35 more directories" in text
    assert len(text.splitlines()) < 30


def test_containing_directory_names_do_not_influence_selection(tmp_path):
    """Selection must depend on the dataset's own layout, not the path above it.

    Regression: ranking originally matched "finalpass"/"flyingthings3d" against
    the whole absolute path, so a containing folder called e.g.
    ``final-experiments`` or ``driving-project`` silently changed which image
    pass or subset got picked.
    """
    misleading = tmp_path / "final-driving-flyingthings3d-experiments"
    misleading.mkdir()
    root = str(misleading)
    build(root, ["A/0000"], pass_dir="frames_cleanpass", prefix="Monkaa/Monkaa")
    build(root, ["A/0000"], pass_dir="frames_finalpass", prefix="Monkaa/Monkaa")

    layout = discover_sceneflow(root, "TRAIN")
    # The only subset actually present is Monkaa, despite the path saying otherwise.
    assert layout.subset == "Monkaa"
    # And the genuine finalpass directory still wins on its own name.
    assert layout.pass_name == "frames_finalpass"
