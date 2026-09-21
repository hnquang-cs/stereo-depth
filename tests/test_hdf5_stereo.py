"""Stereo pairs packed in an HDF5 container.

Some Kaggle mirrors ship one ``.hdf5`` instead of a tree of PNGs, with no
standard internal naming. These tests cover the packings that occur, and --
just as importantly -- that a container which is *not* stereo imagery is
refused with an explanation rather than silently mis-paired.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from stereo.data import DatasetMode
from stereo.data.hdf5_stereo import (Hdf5StereoDataset, detect_view_arrays, find_hdf5_files,
                                     inspect_hdf5, list_arrays)


def write_h5(path, arrays):
    with h5py.File(path, "w") as handle:
        for name, data in arrays.items():
            handle.create_dataset(name, data=data)
    return str(path)


def images(count=4, height=32, width=48, channels_last=True, dtype=np.uint8):
    rng = np.random.default_rng(0)
    shape = (count, height, width, 3) if channels_last else (count, 3, height, width)
    return (rng.random(shape) * 255).astype(dtype) if dtype == np.uint8 else rng.random(shape).astype(dtype)


def test_separate_view_arrays_are_detected(tmp_path):
    path = write_h5(tmp_path / "d.hdf5", {"left": images(), "right": images()})
    dataset = Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)
    assert dataset.left_key == "left" and dataset.right_key == "right"
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["left"].shape == (3, 32, 48)
    assert not any(k in sample for k in ("disparity_gt", "valid_gt_mask"))


@pytest.mark.parametrize("names", [("left_images", "right_images"), ("im0", "im1"),
                                   ("images_left", "images_right"), ("cam0", "cam1")])
def test_alternative_view_names(tmp_path, names):
    path = write_h5(tmp_path / "d.hdf5", {names[0]: images(), names[1]: images()})
    dataset = Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)
    assert (dataset.left_key, dataset.right_key) == names


def test_channels_first_arrays_are_transposed(tmp_path):
    path = write_h5(tmp_path / "d.hdf5",
                    {"left": images(channels_last=False), "right": images(channels_last=False)})
    assert Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)[0]["left"].shape == (3, 32, 48)


def test_uint8_is_scaled_to_unit_range(tmp_path):
    path = write_h5(tmp_path / "d.hdf5", {"left": images(), "right": images()})
    left = Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)[0]["left"]
    assert 0.0 <= float(left.min()) and float(left.max()) <= 1.0


def test_float_images_are_not_rescaled(tmp_path):
    left = np.full((2, 8, 8, 3), 0.5, dtype=np.float32)
    path = write_h5(tmp_path / "d.hdf5", {"left": left, "right": left.copy()})
    assert float(Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)[0]["left"].mean()) == pytest.approx(0.5)


def test_stacked_view_axis_is_detected(tmp_path):
    """Both views in one array, e.g. (N, 2, H, W, 3)."""
    rng = np.random.default_rng(0)
    stacked = (rng.random((3, 2, 16, 24, 3)) * 255).astype(np.uint8)
    path = write_h5(tmp_path / "d.hdf5", {"images": stacked})
    dataset = Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)
    assert dataset.stacked_key == "images"
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["left"].shape == (3, 16, 24)
    # The two views must be different slices, not the same one twice.
    assert not np.allclose(sample["left"].numpy(), sample["right"].numpy())


def test_explicit_keys_override_detection(tmp_path):
    path = write_h5(tmp_path / "d.hdf5", {"a": images(), "b": images()})
    dataset = Hdf5StereoDataset(path, mode=DatasetMode.TRAIN, left_key="a", right_key="b")
    assert len(dataset) == 4


def test_disparity_is_only_read_in_benchmark_mode(tmp_path):
    path = write_h5(tmp_path / "d.hdf5", {
        "left": images(), "right": images(),
        "disparity": np.full((4, 32, 48), 5.0, dtype=np.float32)})

    training = Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)
    assert "disparity_gt" not in training[0]

    benchmark = Hdf5StereoDataset(path, mode=DatasetMode.BENCHMARK)
    assert float(benchmark[0]["disparity_gt"].mean()) == pytest.approx(5.0)
    assert float(benchmark[0]["valid_gt_mask"].mean()) == pytest.approx(1.0)


def test_benchmark_without_disparity_is_refused(tmp_path):
    path = write_h5(tmp_path / "d.hdf5", {"left": images(), "right": images()})
    assert len(Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)) == 4
    with pytest.raises(FileNotFoundError, match="no disparity data"):
        Hdf5StereoDataset(path, mode=DatasetMode.BENCHMARK)


def test_a_non_stereo_container_is_refused_with_its_contents(tmp_path):
    """The critical case: a file named after a dataset that is not stereo imagery."""
    path = write_h5(tmp_path / "flying.hdf5", {
        "data": (np.random.default_rng(0).random((100, 32, 32, 3)) * 255).astype(np.uint8),
        "labels": np.arange(100)})
    with pytest.raises(RuntimeError) as error:
        Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)
    message = str(error.value)
    assert "could not find a stereo pair" in message
    assert "data" in message and "labels" in message, "must list what is actually there"
    assert "not a stereo dataset" in message


def test_single_view_container_is_refused(tmp_path):
    path = write_h5(tmp_path / "d.hdf5", {"left": images()})
    with pytest.raises(RuntimeError, match="could not find a stereo pair"):
        Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)


def test_finds_the_file_inside_a_nested_directory(tmp_path):
    nested = tmp_path / "data" / "data"
    nested.mkdir(parents=True)
    write_h5(nested / "flying.hdf5", {"left": images(), "right": images()})
    (tmp_path / "labelsflying.txt").write_text("x")
    assert len(find_hdf5_files(str(tmp_path))) == 1
    assert len(Hdf5StereoDataset(str(tmp_path), mode=DatasetMode.TRAIN)) == 4


def test_inspect_reports_structure(tmp_path):
    nested = tmp_path / "data" / "data"
    nested.mkdir(parents=True)
    write_h5(nested / "flying.hdf5", {"left": images(), "labels": np.arange(4)})
    report = inspect_hdf5(str(tmp_path))
    assert "flying.hdf5" in report
    assert "(4, 32, 48, 3)" in report
    assert "stereo container" in report


def test_inspect_summarises_rather_than_listing_every_array(tmp_path):
    """A keyed container holds one array per image -- tens of thousands once a
    dataset ships pre-augmented variants. Listing them all floods the log and
    buries the shapes, which are the only thing worth reading."""
    import h5py as _h5py
    path = str(tmp_path / "many.hdf5")
    with _h5py.File(path, "w") as handle:
        for index in range(300):
            handle.create_dataset(f"data/flying/left\\train/0000/aug_{index:04d}.png",
                                  data=np.zeros((8, 8, 3), dtype=np.uint8))
            handle.create_dataset(f"data/flying/right\\train/0000/aug_{index:04d}.png",
                                  data=np.zeros((8, 8, 3), dtype=np.uint8))

    report = inspect_hdf5(str(tmp_path))
    assert "600 arrays in total" in report
    assert "600 arrays  shape=(8, 8, 3)" in report
    # The whole point: a few lines, not six hundred.
    assert len(report.splitlines()) < 15, f"report is {len(report.splitlines())} lines"


def test_inspect_when_there_is_no_hdf5(tmp_path):
    assert "no HDF5 file found" in inspect_hdf5(str(tmp_path))


def test_groups_are_traversed(tmp_path):
    path = str(tmp_path / "d.hdf5")
    with h5py.File(path, "w") as handle:
        group = handle.create_group("train")
        group.create_dataset("left", data=images())
        group.create_dataset("right", data=images())
    assert {name for name, _, _ in list_arrays(path)} == {"train/left", "train/right"}
    dataset = Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)
    assert (dataset.left_key, dataset.right_key) == ("train/left", "train/right")


def test_detect_returns_none_when_ambiguous():
    assert detect_view_arrays([("features", (100, 64), "float32")]) is None


def test_dataset_is_picklable_for_dataloader_workers(tmp_path):
    """h5py handles cannot cross a fork, so the handle must not be pickled."""
    import pickle
    path = write_h5(tmp_path / "d.hdf5", {"left": images(), "right": images()})
    dataset = Hdf5StereoDataset(path, mode=DatasetMode.TRAIN)
    _ = dataset[0]                       # opens the handle
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[0]["left"].shape == (3, 32, 48)
