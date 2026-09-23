"""Resuming a run from a packaged archive.

Kaggle sessions are time-limited, so a long run is done in stages: pack the
checkpoints, upload them somewhere durable, pull them back next session. These
tests cover the full round trip locally, and the Drive URL parsing that the
network path depends on.
"""

import json
import os
import zipfile

import cv2
import numpy as np
import pytest
import torch

from stereo.config import Config
from stereo.data.augmentation import GeometricAugmentConfig, PhotometricAugmentConfig, ResizeConfig
from stereo.data.registry import DatasetSpec
from stereo.model import StereoNetConfig
from stereo.training import Trainer
from stereo.utils.remote import RUN_FILES, package_run, parse_drive_file_id, restore_run


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "cam"
    (root / "left").mkdir(parents=True)
    (root / "right").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(4):
        texture = cv2.GaussianBlur((rng.random((48, 96, 3)) * 255).astype(np.uint8), (5, 5), 0)
        cv2.imwrite(str(root / "left" / f"{index}.png"), texture)
        cv2.imwrite(str(root / "right" / f"{index}.png"), np.roll(texture, -4, axis=1))
    return str(root)


def make_config(dataset, output_dir, epochs):
    config = Config()
    config.model = StereoNetConfig.for_width(96, downsample=4, backbone_width=4, feature_channels=4)
    config.dynamic_disparity = False
    config.data.train = [DatasetSpec(type="folder", root=dataset)]
    config.data.validation = [DatasetSpec(type="folder", root=dataset)]
    config.data.resize = ResizeConfig(48, 96)
    config.data.photometric_augmentation = PhotometricAugmentConfig(enabled=False)
    config.data.geometric_augmentation = GeometricAugmentConfig(enabled=False)
    config.training.epochs = epochs
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.use_amp = False
    config.training.output_dir = output_dir
    config.training.visualize_every = 0
    return config


# --------------------------------------------------------------------------- #
# Drive link parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url,expected", [
    ("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrS/view?usp=sharing", "1AbCdEfGhIjKlMnOpQrS"),
    ("https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQrS", "1AbCdEfGhIjKlMnOpQrS"),
    ("https://drive.google.com/uc?id=1AbCdEfGhIjKlMnOpQrS&export=download", "1AbCdEfGhIjKlMnOpQrS"),
    ("1AbCdEfGhIjKlMnOpQrSt", "1AbCdEfGhIjKlMnOpQrSt"),          # bare id
    ("https://example.com/checkpoints.zip", None),                # plain URL, not Drive
    ("", None),
])
def test_drive_link_shapes(url, expected):
    assert parse_drive_file_id(url) == expected


def test_a_drive_folder_link_is_rejected_with_advice():
    """Pasting the folder instead of the file is the common mistake."""
    with pytest.raises(ValueError, match="FOLDER link"):
        parse_drive_file_id("https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrS")


# --------------------------------------------------------------------------- #
# Packaging and restoring
# --------------------------------------------------------------------------- #

def test_package_and_restore_round_trip(dataset, tmp_path):
    first = str(tmp_path / "session1")
    Trainer(make_config(dataset, first, epochs=2), device=torch.device("cpu")).fit()

    archive = package_run(first)
    assert archive and zipfile.is_zipfile(archive)
    with zipfile.ZipFile(archive) as handle:
        assert set(handle.namelist()) <= set(RUN_FILES)
        assert "last.pt" in handle.namelist() and "history.json" in handle.namelist()

    second = str(tmp_path / "session2")
    found = restore_run(archive, second)
    assert "last.pt" in found
    for name in found:
        assert os.path.isfile(os.path.join(second, name))


def test_restore_finds_files_nested_anywhere_in_the_archive(tmp_path):
    """People zip the folder, not its contents; the layout must not matter."""
    source = tmp_path / "run"
    source.mkdir()
    (source / "last.pt").write_bytes(b"x")
    (source / "history.json").write_text("[]")
    archive = str(tmp_path / "nested.zip")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(source / "last.pt", arcname="outputs/train_unlabeled/last.pt")
        handle.write(source / "history.json", arcname="outputs/train_unlabeled/history.json")

    found = restore_run(archive, str(tmp_path / "out"))
    assert set(found) == {"last.pt", "history.json"}


def test_restoring_an_archive_without_run_files_says_what_it_held(tmp_path):
    archive = str(tmp_path / "wrong.zip")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("notes.txt", "nothing useful")
    with pytest.raises(FileNotFoundError) as error:
        restore_run(archive, str(tmp_path / "out"))
    assert "notes.txt" in str(error.value)


def test_a_non_archive_download_is_reported_clearly(tmp_path):
    """Drive serves an HTML permission page when a file is not shared; that must
    not surface as a baffling 'not a zip file'."""
    from stereo.utils.remote import download
    page = tmp_path / "page.html"
    page.write_text("<!DOCTYPE html><html>Google Drive - Permission denied</html>")
    with pytest.raises(RuntimeError, match="not an archive"):
        download(str(page), str(tmp_path / "out.bin"))


# --------------------------------------------------------------------------- #
# Actually resuming
# --------------------------------------------------------------------------- #

def test_resuming_continues_the_epoch_count_and_the_history(dataset, tmp_path):
    """The point of the whole mechanism: a second session must carry on rather
    than restart, and the curves must stay continuous."""
    first = str(tmp_path / "session1")
    trainer = Trainer(make_config(dataset, first, epochs=3), device=torch.device("cpu"))
    trainer.fit()
    assert [r["epoch"] for r in trainer.history] == [0, 1, 2]
    archive = package_run(first)

    second = str(tmp_path / "session2")
    restored = restore_run(archive, second)

    config = make_config(dataset, second, epochs=6)
    config.training.resume = restored["last.pt"]
    resumed = Trainer(config, device=torch.device("cpu"))

    assert resumed.start_epoch == 3, "must continue after the last completed epoch"
    assert [r["epoch"] for r in resumed.history] == [0, 1, 2], "earlier epochs must be kept"

    resumed.fit()
    epochs = [r["epoch"] for r in resumed.history]
    assert epochs == [0, 1, 2, 3, 4, 5], f"history is not continuous: {epochs}"

    with open(os.path.join(second, "history.json")) as handle:
        assert [r["epoch"] for r in json.load(handle)] == [0, 1, 2, 3, 4, 5]


def test_resuming_keeps_the_best_score_so_a_worse_checkpoint_cannot_overwrite_it(dataset, tmp_path):
    first = str(tmp_path / "session1")
    trainer = Trainer(make_config(dataset, first, epochs=3), device=torch.device("cpu"))
    trainer.fit()
    best_before = trainer.best_metric
    archive = package_run(first)

    second = str(tmp_path / "session2")
    restored = restore_run(archive, second)
    config = make_config(dataset, second, epochs=5)
    config.training.resume = restored["last.pt"]
    resumed = Trainer(config, device=torch.device("cpu"))

    assert resumed.best_metric == pytest.approx(best_before), \
        "the best label-free score must carry over, or a worse checkpoint overwrites the best"
