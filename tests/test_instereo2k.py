"""InStereo2K: the loader, and its preparation from a hand-made download.

No real InStereo2K files can be fetched automatically (its hosts are OneDrive
and Baidu), so its scale is pinned here on synthetic scenes, and the preparation
must flag the other scale in circulation -- torchvision's /1024 -- if the real
files turn out to use it.
"""

import os
import zipfile

import cv2
import numpy as np
import pytest

from stereo.data import DatasetMode, InStereo2kDataset, build_dataset, check_label_scale
from stereo.data.prepare import prepare_instereo2k, shrink_instereo2k_scene, verify
from stereo.data.registry import DatasetSpec


def _scene(directory, width=128, height=48, shift=4, scale=100, seed=0):
    """A textured scene whose right view is the left shifted by ``shift`` px."""
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    texture = cv2.GaussianBlur((rng.random((height, width + shift, 3)) * 255).astype(np.uint8),
                               (0, 0), 1.5)
    cv2.imwrite(str(directory / "left.png"), texture[:, :width])
    cv2.imwrite(str(directory / "right.png"), texture[:, shift:])
    for name in ("left_disp.png", "right_disp.png"):
        stored = np.full((height, width), shift * scale, dtype=np.uint16)
        stored[0, 0] = 0                                           # unknown
        cv2.imwrite(str(directory / name), stored)


@pytest.fixture
def instereo(tmp_path):
    root = tmp_path / "InStereo2K"
    for index in range(2):
        _scene(root / "train" / "part1" / f"00000{index}", seed=index)
    _scene(root / "test" / "000100", seed=9)
    return root


def test_disparity_is_the_stored_value_over_100(instereo):
    dataset = InStereo2kDataset(str(instereo), mode=DatasetMode.BENCHMARK)
    assert len(dataset) == 2                                       # the train split
    sample = dataset[0]
    assert float(sample["disparity_gt"][0, 1, 1]) == pytest.approx(4.0)
    assert float(sample["valid_gt_mask"][0, 0, 0]) == 0.0
    assert "disparity_gt_right" not in sample                      # only training flips
    assert check_label_scale([sample]).consistent


def test_splits_and_the_right_view_for_the_flip(instereo):
    assert len(InStereo2kDataset(str(instereo), split="test", mode=DatasetMode.BENCHMARK)) == 1
    assert len(InStereo2kDataset(str(instereo), split=None, mode=DatasetMode.BENCHMARK)) == 3
    training = build_dataset(DatasetSpec(type="instereo2k", root=str(instereo),
                                         options={"split": "train"}),
                             DatasetMode.TRAIN, with_labels=True)
    sample = training[0]
    assert float(sample["disparity_gt_right"][0, 1, 1]) == pytest.approx(4.0)
    assert float(sample["valid_gt_mask_right"][0, 0, 0]) == 0.0


def test_a_tree_without_scenes_says_what_it_looked_for(tmp_path):
    (tmp_path / "train" / "x").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="left_disp.png"):
        InStereo2kDataset(str(tmp_path))


def test_shrinking_keeps_the_encoding_and_the_alignment(tmp_path):
    _scene(tmp_path / "full", width=400, height=96, shift=16)
    assert shrink_instereo2k_scene(str(tmp_path / "full"), str(tmp_path / "small" / "train" / "a"),
                                   max_width=200) == 2
    sample = InStereo2kDataset(str(tmp_path / "small"), mode=DatasetMode.BENCHMARK)[0]
    assert sample["left"].shape[-1] == 200
    assert float(sample["disparity_gt"][0, 2, 2]) == pytest.approx(8.0)
    assert check_label_scale([sample]).consistent


def test_preparation_takes_an_archive_and_checks_it(instereo, tmp_path):
    archive = tmp_path / "instereo2k.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for directory, _, names in os.walk(instereo):
            for name in names:
                path = os.path.join(directory, name)
                handle.write(path, os.path.relpath(path, tmp_path))
    out = tmp_path / "prepared"
    prepare_instereo2k(str(out), str(archive), str(tmp_path / "cache"), max_width=960)
    assert len(InStereo2kDataset(str(out / "instereo2k"), split=None,
                                 mode=DatasetMode.BENCHMARK)) == 3
    assert verify(str(out)) == []


def test_preparation_flags_the_torchvision_scale(tmp_path):
    """Files stored at x1024 would read 10x too large at /100; say so, and how to fix it."""
    for index in range(3):
        _scene(tmp_path / "raw" / "train" / f"{index:06d}", scale=1024, seed=index)
    out = tmp_path / "prepared"
    prepare_instereo2k(str(out), str(tmp_path / "raw"), str(tmp_path / "cache"), max_width=960)
    problems = verify(str(out))
    assert len(problems) == 1 and "/1024" in problems[0] and "DISPARITY_SCALE" in problems[0]
