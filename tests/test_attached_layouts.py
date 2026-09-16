"""Layout discovery for the three datasets attached on Kaggle.

Kaggle mirrors reshape public datasets freely, so each loader finds its data by
looking for the pair of views rather than by assuming a path. These tests build
the structures those mirrors actually use.
"""

import os

import cv2
import numpy as np
import pytest

from stereo.data import DatasetMode, KittiStereoDataset, MiddleburyDataset
from stereo.data.discovery import find_view_dir_pairs, view_image_dir


def _png(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (np.random.default_rng(0).random((32, 48, 3)) * 255).astype(np.uint8))


def _disparity_png(path, value=6.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full((32, 48), int(value * 256), dtype=np.uint16))


KITTI_CALIB = (
    "P_rect_02: 721.5 0 609.5 0 0 721.5 172.8 0 0 0 1 0\n"
    "P_rect_03: 721.5 0 609.5 -387.5 0 721.5 172.8 0 0 0 1 0\n"
)


# --------------------------------------------------------------------------- #
# KITTI raw / Eigen split
# --------------------------------------------------------------------------- #

def build_kitti_raw(root, drives=("2011_09_26/2011_09_26_drive_0001_sync",), frames=2):
    for drive in drives:
        for view in ("image_02", "image_03"):
            for index in range(frames):
                _png(os.path.join(root, drive, view, "data", f"{index:010d}.png"))
        date_dir = os.path.dirname(os.path.join(root, drive))
        with open(os.path.join(date_dir, "calib_cam_to_cam.txt"), "w") as handle:
            handle.write(KITTI_CALIB)
    return root


def test_kitti_raw_layout_is_detected_and_paired(tmp_path):
    """image_02/data + image_03/data are the rectified colour cameras."""
    root = build_kitti_raw(str(tmp_path))
    dataset = KittiStereoDataset(root, mode=DatasetMode.TRAIN)
    assert dataset.version == "raw"
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["left"].shape == (3, 32, 48)
    assert not any(k in sample for k in ("disparity_gt", "valid_gt_mask"))


def test_kitti_raw_reads_calibration_for_metric_depth(tmp_path):
    root = build_kitti_raw(str(tmp_path))
    metadata = KittiStereoDataset(root, mode=DatasetMode.TRAIN)[0]["metadata"]
    assert metadata["focal_length"] == pytest.approx(721.5)
    # baseline = (P3[0,3] - P2[0,3]) / -f = (-387.5 - 0) / -721.5
    assert metadata["baseline"] == pytest.approx(387.5 / 721.5, rel=1e-4)


def test_kitti_raw_handles_many_drives_and_nesting(tmp_path):
    build_kitti_raw(str(tmp_path / "kitti_eigen" / "raw"),
                    drives=("2011_09_26/2011_09_26_drive_0001_sync",
                            "2011_09_26/2011_09_26_drive_0002_sync",
                            "2011_09_30/2011_09_30_drive_0016_sync"))
    dataset = KittiStereoDataset(str(tmp_path), mode=DatasetMode.TRAIN)
    assert len(dataset) == 6
    assert len({e.pair_dir for e in dataset.entries}) == 3


def test_kitti_raw_refuses_to_benchmark(tmp_path):
    """Raw recordings have no disparity GT; the Eigen protocol is a depth benchmark."""
    root = build_kitti_raw(str(tmp_path))
    with pytest.raises(RuntimeError, match="no disparity ground truth"):
        KittiStereoDataset(root, mode=DatasetMode.BENCHMARK)


def test_kitti_2015_benchmark_layout_still_works(tmp_path):
    root = str(tmp_path / "training")
    for name in ("000000_10.png", "000000_11.png"):
        _png(os.path.join(root, "image_2", name))
        _png(os.path.join(root, "image_3", name))
    _disparity_png(os.path.join(root, "disp_occ_0", "000000_10.png"))
    os.makedirs(os.path.join(root, "calib_cam_to_cam"), exist_ok=True)
    with open(os.path.join(root, "calib_cam_to_cam", "000000.txt"), "w") as handle:
        handle.write(KITTI_CALIB)

    assert KittiStereoDataset(root, mode=DatasetMode.TRAIN).version == "2015"
    # Training uses every frame; benchmarking only the labelled _10 reference frames.
    assert len(KittiStereoDataset(root, mode=DatasetMode.TRAIN)) == 2
    benchmark = KittiStereoDataset(root, mode=DatasetMode.BENCHMARK)
    assert len(benchmark) == 1
    assert float(benchmark[0]["disparity_gt"].mean()) == pytest.approx(6.0, abs=1e-2)


def test_kitti_2015_is_preferred_over_raw_when_both_present(tmp_path):
    """A mirror bundling both should benchmark on the one that has ground truth."""
    build_kitti_raw(str(tmp_path / "raw"))
    root = str(tmp_path / "training")
    _png(os.path.join(root, "image_2", "000000_10.png"))
    _png(os.path.join(root, "image_3", "000000_10.png"))
    assert KittiStereoDataset(str(tmp_path), mode=DatasetMode.TRAIN).version == "2015"


def test_kitti_unrecognised_layout_reports_the_tree(tmp_path):
    _png(str(tmp_path / "nonsense" / "a.png"))
    with pytest.raises(RuntimeError) as error:
        KittiStereoDataset(str(tmp_path), mode=DatasetMode.TRAIN)
    assert "nonsense" in str(error.value)


# --------------------------------------------------------------------------- #
# Middlebury
# --------------------------------------------------------------------------- #

def build_middlebury(root, scenes=("Adirondack", "Motorcycle"), prefix="", ndisp=290):
    base = os.path.join(root, prefix) if prefix else root
    for scene in scenes:
        scene_dir = os.path.join(base, scene)
        _png(os.path.join(scene_dir, "im0.png"))
        _png(os.path.join(scene_dir, "im1.png"))
        with open(os.path.join(scene_dir, "calib.txt"), "w") as handle:
            handle.write(f"cam0=[1758 0 977; 0 1758 552; 0 0 1]\nbaseline=111\n"
                         f"doffs=0\nndisp={ndisp}\n")
    return root


@pytest.mark.parametrize("prefix", ["", "MiddEval3/trainingH", "middlebury/data",
                                    "MiddleburyStereoDataset/2014"])
def test_middlebury_scenes_are_found_at_any_depth(tmp_path, prefix):
    """Mirrors wrap the scenes in arbitrary extra folders."""
    root = build_middlebury(str(tmp_path), prefix=prefix)
    dataset = MiddleburyDataset(root, mode=DatasetMode.TRAIN)
    assert len(dataset) == 2
    assert not any(k in dataset[0] for k in ("disparity_gt", "valid_gt_mask"))


def test_middlebury_calibration_survives_nesting(tmp_path):
    root = build_middlebury(str(tmp_path), prefix="MiddEval3/trainingH")
    metadata = MiddleburyDataset(root, mode=DatasetMode.TRAIN)[0]["metadata"]
    assert metadata["focal_length"] == pytest.approx(1758.0)
    assert metadata["baseline"] == pytest.approx(0.111)
    assert metadata["ndisp"] == pytest.approx(290.0)


def test_middlebury_unrecognised_layout_reports_the_tree(tmp_path):
    _png(str(tmp_path / "wrong_place" / "img.png"))
    with pytest.raises(RuntimeError) as error:
        MiddleburyDataset(str(tmp_path), mode=DatasetMode.TRAIN)
    assert "wrong_place" in str(error.value)


# --------------------------------------------------------------------------- #
# Shared discovery
# --------------------------------------------------------------------------- #

def test_discovery_skips_ground_truth_trees(tmp_path):
    """A disparity tree mirrors the image tree's view folders; it must not be indexed."""
    for view in ("left", "right"):
        _png(str(tmp_path / "frames" / "scene" / view / "0.png"))
        _png(str(tmp_path / "disparity" / "scene" / view / "0.png"))
    found = [directory for directory, _ in find_view_dir_pairs(str(tmp_path))]
    assert any("frames" in d for d in found)
    assert not any("disparity" in d for d in found)


def test_view_image_dir_handles_the_kitti_raw_data_level(tmp_path):
    _png(str(tmp_path / "image_02" / "data" / "0.png"))
    assert view_image_dir(str(tmp_path), "image_02").endswith("data")
    _png(str(tmp_path / "left" / "0.png"))
    assert not view_image_dir(str(tmp_path), "left").endswith("data")


def test_monocular_dataset_is_named_as_such(tmp_path):
    """A tree with image_02 but no image_03 is monocular, not a misconfigured
    stereo tree. Saying so is the only actionable message: no reconfiguring can
    produce a second camera that was never recorded."""
    from stereo.data.discovery import describe_missing_stereo, find_unpaired_views

    root = str(tmp_path)
    for index in range(2):
        _png(os.path.join(root, "drive_0001", "image_02", "data", f"{index:010d}.png"))
    _png(os.path.join(root, "drive_0001", "proj_depth", "groundtruth", "0.png"))

    unpaired = find_unpaired_views(root)
    assert unpaired and unpaired[0][1:] == ("image_02", "image_03")
    assert "MONOCULAR" in describe_missing_stereo(root)

    with pytest.raises(RuntimeError) as error:
        KittiStereoDataset(root, mode=DatasetMode.TRAIN)
    message = str(error.value)
    assert "MONOCULAR" in message
    assert "image_02 (without image_03)" in message


def test_a_genuine_stereo_tree_reports_no_unpaired_views(tmp_path):
    from stereo.data.discovery import describe_missing_stereo
    build_kitti_raw(str(tmp_path))
    assert describe_missing_stereo(str(tmp_path)) == ""
