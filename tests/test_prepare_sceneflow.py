"""The Scene Flow preparation, on synthetic archives laid out like the official ones.

The real archives were streamed end to end (the first 600 files of each); these
pin the path -- ranged streaming, conversion, index, loading -- without the network.
"""

import bz2
import io
import os
import tarfile

import cv2
import numpy as np
import pytest

from stereo.data import DatasetMode, check_label_scale
from stereo.data import prepare_sceneflow as psf
from stereo.data.io import parse_pfm
from stereo.data.prepare import verify
from stereo.data.sceneflow import SceneFlowDataset

SHIFT = 8


def _pfm(disparity):
    header = b"Pf\n%d %d\n-1.0\n" % (disparity.shape[1], disparity.shape[0])
    return header + np.flipud(disparity).astype("<f4").tobytes()


def _archives(tmp_path):
    """FlyingThings3D-shaped image tar and disparity tar.bz2, one TRAIN and one TEST frame."""
    rng = np.random.default_rng(0)
    images, disparities = io.BytesIO(), io.BytesIO()
    with tarfile.open(fileobj=images, mode="w") as image_tar, \
            tarfile.open(fileobj=disparities, mode="w") as disparity_tar:
        for split in ("TRAIN", "TEST"):
            texture = cv2.GaussianBlur((rng.random((108, 192 + SHIFT, 3)) * 255).astype(np.uint8),
                                       (0, 0), 1.5)
            views = {"left": texture[:, :192], "right": texture[:, SHIFT:]}
            for view, image in views.items():
                data = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, 100])[1].tobytes()
                info = tarfile.TarInfo(f"frames_finalpass_webp/{split}/A/0000/{view}/0006.webp")
                info.size = len(data)
                image_tar.addfile(info, io.BytesIO(data))
                data = _pfm(np.full((108, 192), float(SHIFT), np.float32))
                info = tarfile.TarInfo(f"disparity/{split}/A/0000/{view}/0006.pfm")
                info.size = len(data)
                disparity_tar.addfile(info, io.BytesIO(data))
    (tmp_path / "images.tar").write_bytes(images.getvalue())
    (tmp_path / "disparity.tar.bz2").write_bytes(bz2.compress(disparities.getvalue()))
    return f"file://{tmp_path / 'images.tar'}", f"file://{tmp_path / 'disparity.tar.bz2'}"


def test_members_keep_the_official_tree():
    assert psf.destination("frames_finalpass_webp/TRAIN/A/0000/left/0006.webp") == \
        ("frames_finalpass/TRAIN/A/0000/left/0006.webp", "TRAIN")
    assert psf.destination("disparity/TEST/C/0149/right/0010.pfm")[1] == "TEST"
    assert psf.destination("frames_finalpass_webp/flower_storm_x2/left/0001.webp")[1] == "TRAIN"
    assert psf.destination("disparity/35mm_focallength/scene_forwards/fast/left/0023.pfm") == \
        ("disparity/35mm_focallength/scene_forwards/fast/left/0023.pfm", "TRAIN")


def test_disparity_is_encoded_in_128ths_with_zero_for_unknown():
    encoded = psf.encode_disparity(np.array([[1.5, np.inf, -2.0, 600.0]], np.float32))
    assert encoded.tolist() == [[192, 0, 0, 0]]                    # 600 px is out of range


def test_a_remote_file_reads_back_exactly_through_small_ranges(tmp_path):
    payload = np.random.default_rng(1).integers(0, 256, 10_007, dtype=np.uint8).tobytes()
    (tmp_path / "blob").write_bytes(payload)
    stream = psf.RemoteStream(f"file://{tmp_path / 'blob'}", str(tmp_path / "cache"),
                              chunk=1000, ahead=3, connections=2)
    assert io.BufferedReader(stream).read() == payload
    stream.close()
    assert not [name for name in os.listdir(tmp_path / "cache") if name.startswith("range")]


def test_a_streamed_subset_is_converted_indexed_and_checked(tmp_path, monkeypatch):
    urls = _archives(tmp_path)
    monkeypatch.setattr(psf, "archive_urls", lambda subset: urls)
    out = tmp_path / "out"
    frames = psf.prepare_sceneflow(str(out), ["flyingthings3d"], str(tmp_path / "cache"), workers=2)
    assert frames == {"flyingthings3d": {"frames_finalpass/TRAIN": 1, "frames_finalpass/TEST": 1}}

    root = str(out / "flyingthings3d")
    train = SceneFlowDataset(root, split="TRAIN", mode=DatasetMode.TRAIN)
    train.with_labels = True
    sample = train[0]
    assert sample["left"].shape[-2:] == (54, 96)                    # halved
    assert float(sample["disparity_gt"].mean()) == pytest.approx(SHIFT / 2)
    assert "disparity_gt_right" in sample                           # the flip can apply
    assert check_label_scale([sample]).consistent

    test = SceneFlowDataset(root, split="TEST", mode=DatasetMode.BENCHMARK)[0]
    assert test["left"].shape[-2:] == (108, 192)                    # full size
    assert float(test["disparity_gt"].mean()) == pytest.approx(SHIFT)
    assert not os.path.exists(out / "flyingthings3d" / "disparity" / "TEST" / "A" / "0000" / "right")
    assert verify(str(out)) == []
