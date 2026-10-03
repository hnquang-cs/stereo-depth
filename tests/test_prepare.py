"""The one-time data preparation, without the network.

The downloads themselves were run for real; these pin the parts that transform
data -- shrinking, calibration rewriting, archive member selection -- so that a
change cannot silently produce labels that disagree with their images.
"""

import json
import os

import cv2
import numpy as np
import pytest

from stereo.data import DatasetMode, MiddleburyDataset, check_label_scale
from stereo.data.io import read_pfm, write_pfm
from stereo.data.prepare import (MANIFEST, fetch_segmented, find_prepared, kitti_member,
                                 middlebury_downloads, prepare_all, shrink_calib,
                                 shrink_disparity, shrink_scene)


def test_write_pfm_round_trips(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    data[0, 0] = np.inf
    write_pfm(str(tmp_path / "d.pfm"), data)
    assert np.array_equal(read_pfm(str(tmp_path / "d.pfm")), data)


def test_shrink_disparity_takes_each_blocks_median_in_shrunk_pixels():
    disparity = np.array([[8, 8, 4, 4],
                          [8, 20, 4, np.inf],
                          [np.inf, np.inf, 6, 6],
                          [np.inf, 2, 6, 6]], dtype=np.float32)
    shrunk = shrink_disparity(disparity, 2)
    assert shrunk[0, 0] == pytest.approx(4.0)     # median 8, an outlier ignored; / 2
    assert shrunk[0, 1] == pytest.approx(2.0)     # 3 of 4 known
    assert np.isinf(shrunk[1, 0])                 # 1 of 4 known: unknown
    assert shrunk[1, 1] == pytest.approx(3.0)


def test_shrink_calib_scales_the_camera_and_every_disparity_entry():
    text = ("cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]\n"
            "cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]\n"
            "doffs=131.111\nbaseline=193.001\nwidth=2964\nheight=1988\n"
            "ndisp=280\nisint=0\nvmin=31\nvmax=257\n")
    shrunk = dict(line.split("=", 1) for line in shrink_calib(text, 4, 741, 497).splitlines())
    cam0 = [float(v) for v in shrunk["cam0"].strip("[]").replace(";", " ").split()]
    assert cam0[0] == pytest.approx(3997.684 / 4) and cam0[4] == pytest.approx(3997.684 / 4)
    assert cam0[2] == pytest.approx((1176.728 + 0.5) / 4 - 0.5)
    assert cam0[6:] == [0.0, 0.0, 1.0]
    assert float(shrunk["doffs"]) == pytest.approx(131.111 / 4)
    assert float(shrunk["baseline"]) == pytest.approx(193.001)          # metres do not shrink
    assert (shrunk["width"], shrunk["height"], shrunk["ndisp"]) == ("741", "497", "70")
    assert float(shrunk["vmax"]) == pytest.approx(257 / 4)


def test_a_shrunk_scene_keeps_its_labels_aligned_with_its_images(tmp_path):
    """The check that matters: after shrinking, the right view shifted by the
    label still reconstructs the left -- at x1, not at the old scale."""
    shift, width, height = 16, 512, 96
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur((rng.random((height, width + shift, 3)) * 255).astype(np.uint8),
                               (0, 0), 2.0)
    source = tmp_path / "full" / "Scene-perfect"
    source.mkdir(parents=True)
    cv2.imwrite(str(source / "im0.png"), texture[:, :width])        # left(x)  = T(x)
    cv2.imwrite(str(source / "im1.png"), texture[:, shift:])        # right(x) = T(x + shift)
    disparity = np.full((height, width), float(shift), dtype=np.float32)
    disparity[:4] = np.inf
    write_pfm(str(source / "disp0.pfm"), disparity)
    (source / "calib.txt").write_text(f"cam0=[1000 0 256; 0 1000 48; 0 0 1]\n"
                                      f"width={width}\nheight={height}\nndisp=64\n")

    target = tmp_path / "prepared" / "Scene-perfect"
    assert shrink_scene(str(source), str(target), max_width=128) == 4

    sample = MiddleburyDataset(str(tmp_path / "prepared"), mode=DatasetMode.BENCHMARK)[0]
    assert sample["left"].shape[-2:] == (24, 128)
    known = sample["valid_gt_mask"] > 0
    assert float(sample["disparity_gt"][known].mean()) == pytest.approx(4.0)
    assert float(known[0, 0].float().mean()) == 0.0                 # the unknown band survives
    assert check_label_scale([sample]).consistent
    assert "width=128" in (target / "calib.txt").read_text()


def test_every_release_is_downloaded_once():
    jobs = middlebury_downloads("/cache")
    paths = [path for _, path in jobs]
    assert len(paths) == len(set(paths))
    urls = " ".join(url for url, _ in jobs)
    for fragment in ("MiddEval3-GT1-Q.zip", "scenes2001/data/barn2/disp6.pgm",
                     "scenes2003/newdata/full/teddyH-ppm-2.zip",
                     "scenes2005/HalfSize/zip-2views/Reindeer-2views.zip",
                     "scenes2006/ThirdSize/zip-2views/Wood2-2views.zip",
                     "scenes2014/datasets/Umbrella-perfect/disp1.pfm",
                     "scenes2021/data/traproom2/calib.txt"):
        assert fragment in urls, fragment
    assert "Computer" not in urls                     # 2005's withheld ground truth


def test_only_kittis_labelled_frames_are_taken():
    folders = ("training/image_2/", "training/disp_occ_0/", "training/calib_cam_to_cam/")
    assert kitti_member("training/image_2/000042_10.png", folders)
    assert kitti_member("training/disp_occ_0/000042_10.png", folders)
    assert kitti_member("training/calib_cam_to_cam/000042.txt", folders)
    assert not kitti_member("training/image_2/000042_11.png", folders)   # next frame, no label
    assert not kitti_member("testing/image_2/000042_10.png", folders)
    assert not kitti_member("training/disp_occ_1/000042_10.png", folders)


def test_the_download_cache_cannot_be_inside_the_output(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        prepare_all(str(tmp_path), cache=str(tmp_path / "cache"), middlebury=False, kitti=False)


def test_an_attached_prepared_dataset_is_found_by_its_manifest(tmp_path):
    prepared = tmp_path / "datasets" / "someone" / "stereo-training-data" / "stereo-data"
    prepared.mkdir(parents=True)
    (prepared / MANIFEST).write_text(json.dumps({"format": 1}))
    (tmp_path / "datasets" / "other" / "flying-things-3d" / "data").mkdir(parents=True)
    assert find_prepared(str(tmp_path)) == str(prepared)
    assert find_prepared(str(tmp_path / "datasets" / "other")) is None


def test_a_segmented_download_reassembles_the_file_exactly(tmp_path):
    """Large archives arrive as parallel byte ranges; joined, they must be the file."""
    payload = np.random.default_rng(0).integers(0, 256, 100_003, dtype=np.uint8).tobytes()
    (tmp_path / "source.bin").write_bytes(payload)
    target = tmp_path / "copy.bin"
    fetch_segmented(f"file://{tmp_path / 'source.bin'}", str(target), len(payload), segments=7)
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob("copy.bin.part*"))
