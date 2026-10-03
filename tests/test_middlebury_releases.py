"""Every Middlebury release is read with its own file names and disparity encoding.

The encodings were measured on downloaded data (see stereo/data/middlebury.py);
these tests pin each rule so a refactor cannot silently change a label scale.
"""

import os

import cv2
import numpy as np
import pytest

from stereo.data import DatasetMode, MiddleburyDataset
from stereo.data.io import image_width


def _image(path, width, value=None, height=6):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if value is None:
        image = (np.random.default_rng(width).random((height, width, 3)) * 255).astype(np.uint8)
    else:
        image = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.imwrite(path, image)


def _disparity(path, width, stored, height=6):
    """Stored disparity with one unknown pixel at (0, 0): 0 in PNG/PGM, inf in PFM."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".pfm"):
        data = np.full((height, width), stored, dtype="<f4")
        data[0, 0] = np.inf
        with open(path, "wb") as handle:
            handle.write(b"Pf\n%d %d\n-1.0\n" % (width, height) + data[::-1].tobytes())
    else:
        data = np.full((height, width), stored, dtype=np.uint8)
        data[0, 0] = 0
        cv2.imwrite(path, data)


def _scene(directory, left, right, disparity, width, stored):
    _image(str(directory / left), width)
    _image(str(directory / right), width)
    _disparity(str(directory / disparity), width, stored)


@pytest.mark.parametrize("release, scene, left, right, disparity, width, stored, pixels", [
    ("2001", "sawtooth", "im2.ppm", "im6.ppm", "disp2.pgm", 434, 80, 10.0),
    ("2003", "cones", "im2.png", "im6.png", "disp2.png", 450, 100, 25.0),
    ("2003", "conesH", "im2.ppm", "im6.ppm", "disp2.pgm", 900, 100, 50.0),
    ("2003", "conesF", "im2.ppm", "im6.ppm", "disp2.pgm", 1800, 100, 100.0),
    ("2005", "Art", "view1.png", "view5.png", "disp1.png", 463, 60, 20.0),
    ("2006", "Aloe", "view1.png", "view5.png", "disp1.png", 641, 60, 30.0),
    ("2006", "Aloe", "view1.png", "view5.png", "disp1.png", 1282, 60, 60.0),
    ("2014", "Adirondack-perfect", "im0.png", "im1.png", "disp0.pfm", 64, 7.5, 7.5),
    ("2021", "artroom1", "im0.png", "im1.png", "disp0.pfm", 64, 7.5, 7.5),
    ("MiddEval3", "Adirondack", "im0.png", "im1.png", "disp0GT.pfm", 64, 7.5, 7.5),
])
def test_each_release_is_read_in_pixels(tmp_path, release, scene, left, right, disparity,
                                        width, stored, pixels):
    _scene(tmp_path / "mirror" / scene, left, right, disparity, width, stored)
    dataset = MiddleburyDataset(str(tmp_path), mode=DatasetMode.BENCHMARK)
    assert dataset.releases == {release: 1}

    sample = dataset[0]
    assert sample["left"].shape[-1] == width
    assert float(sample["disparity_gt"][0, 1, 1]) == pytest.approx(pixels)
    assert float(sample["valid_gt_mask"][0, 0, 0]) == 0.0       # the unknown pixel
    assert float(sample["valid_gt_mask"][0, 1:, :].min()) == 1.0
    assert sample["metadata"]["release"] == release


@pytest.mark.parametrize("scene, default", [("Art", 1), ("Aloe", 2)])
def test_2005_and_2006_use_their_default_exposure(tmp_path, scene, default):
    """The release notes name Illum1/Exp1 for 2005 and Illum1/Exp2 for 2006.
    The first exposure found, Exp0, is as much as 9x darker."""
    directory = tmp_path / "ThirdSize" / scene
    for illumination in (1, 2, 3):
        for exposure in (0, 1, 2):
            for view in ("view1.png", "view5.png"):
                _image(str(directory / f"Illum{illumination}" / f"Exp{exposure}" / view), 440,
                       value=10 * (3 * illumination + exposure))
    _disparity(str(directory / "disp1.png"), 440, 60)

    sample = MiddleburyDataset(str(tmp_path), mode=DatasetMode.BENCHMARK)[0]
    assert float(sample["left"].mean()) * 255 == pytest.approx(10 * (3 + default), abs=0.5)


def test_files_that_look_like_disparity_are_never_read_as_it(tmp_path):
    """Beside disp0.pfm, 2014 ships the VERTICAL disparity (disp0y.pfm), a sample
    count (disp0-n.pgm) and a standard deviation (disp0-sd.pfm)."""
    scene = tmp_path / "Classroom1-imperfect"
    _image(str(scene / "im0.png"), 64)
    _image(str(scene / "im1.png"), 64)
    _disparity(str(scene / "disp0y.pfm"), 64, 0.4)
    _disparity(str(scene / "disp0-sd.pfm"), 64, 0.1)
    _disparity(str(scene / "disp0-n.pgm"), 64, 9)
    with pytest.raises(RuntimeError, match="views but no ground truth"):
        MiddleburyDataset(str(tmp_path))

    _disparity(str(scene / "disp0.pfm"), 64, 7.5)
    sample = MiddleburyDataset(str(tmp_path), mode=DatasetMode.BENCHMARK)[0]
    assert float(sample["disparity_gt"][0, 1, 1]) == pytest.approx(7.5)


def test_a_scene_in_several_copies_is_used_once(tmp_path):
    for name, width in (("conesQ", 450), ("conesH", 900), ("conesF", 1800)):
        _scene(tmp_path / "2003" / name, "im2.ppm", "im6.ppm", "disp2.pgm", width, 100)
    for variant in ("perfect", "imperfect"):
        _scene(tmp_path / "2014" / f"Piano-{variant}", "im0.png", "im1.png", "disp0.pfm", 64, 5.0)

    def chosen(**options):
        dataset = MiddleburyDataset(str(tmp_path), **options)
        return {os.path.basename(scene.directory) for scene in dataset.entries}, dataset.duplicates

    assert chosen() == ({"conesQ", "Piano-perfect"}, 3)                  # smallest, perfect
    assert chosen(min_width=600)[0] == {"conesH", "Piano-perfect"}       # smallest that is wide enough
    assert chosen(min_width=5000)[0] == {"conesF", "Piano-perfect"}      # none is: the widest


def test_scenes_without_ground_truth_are_skipped_and_reported(tmp_path):
    """MiddEval3 ships its test set without ground truth; supervised training
    could only waste draws on it."""
    for split, labelled in (("trainingQ", True), ("testQ", False)):
        for scene in ("A", "B"):
            directory = tmp_path / "MiddEval3" / split / scene
            _image(str(directory / "im0.png"), 64)
            _image(str(directory / "im1.png"), 64)
            if labelled:
                _disparity(str(directory / "disp0GT.pfm"), 64, 5.0)

    dataset = MiddleburyDataset(str(tmp_path))
    assert len(dataset) == 2
    assert dataset.unlabelled == [os.path.join("MiddEval3", "testQ", s) for s in ("A", "B")]
    assert "2 without ground truth skipped" in dataset.summary()


def test_releases_can_be_selected(tmp_path):
    _scene(tmp_path / "MiddEval3" / "trainingQ" / "Adirondack", "im0.png", "im1.png",
           "disp0GT.pfm", 64, 5.0)
    _scene(tmp_path / "2003" / "cones", "im2.png", "im6.png", "disp2.png", 450, 100)

    assert MiddleburyDataset(str(tmp_path)).releases == {"2003": 1, "MiddEval3": 1}
    assert MiddleburyDataset(str(tmp_path), releases=["MiddEval3"]).releases == {"MiddEval3": 1}
    with pytest.raises(ValueError, match="unknown Middlebury release"):
        MiddleburyDataset(str(tmp_path), releases=["2015"])


def test_image_width_reads_the_header(tmp_path):
    for name, shape in (("a.png", (5, 37, 3)), ("a.ppm", (5, 37, 3)), ("a.pgm", (5, 37))):
        cv2.imwrite(str(tmp_path / name), np.zeros(shape, dtype=np.uint8))
        assert image_width(str(tmp_path / name)) == 37
    (tmp_path / "commented.ppm").write_bytes(b"P6\n# made by 99 tools\n37 5\n255\n" + bytes(555))
    assert image_width(str(tmp_path / "commented.ppm")) == 37
