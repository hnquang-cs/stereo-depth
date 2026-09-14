"""Dataset contract, and the mechanism that keeps ground truth out of training.

Three modes, three different sample dictionaries
------------------------------------------------
``DatasetMode.TRAIN`` and ``DatasetMode.VALIDATION``
    ``{"left", "right", "metadata"}`` -- images only.  A dataset in these modes
    never opens a ground-truth file.  ``VALIDATION`` differs from ``TRAIN`` only
    in that augmentation is off; it is the *label-free* validation used for
    checkpoint selection.

``DatasetMode.BENCHMARK``
    adds ``{"disparity_gt", "valid_gt_mask"}`` and, where the benchmark provides
    it, ``"depth_gt"``.  Only :mod:`stereo.evaluation` constructs datasets in
    this mode.

The separation is enforced twice:

1. :meth:`StereoDataset.__getitem__` routes through ``_load_images`` and only
   calls ``_load_ground_truth`` when the mode is ``BENCHMARK``.
2. :func:`assert_label_free` raises on any sample or batch that carries a
   ground-truth key.  The training loop calls it on every batch, so a dataset
   that leaked labels would fail loudly on the first iteration rather than
   quietly improve the loss.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

#: Keys that may appear in a benchmark sample and must never appear in a training one.
GROUND_TRUTH_KEYS = ("disparity_gt", "depth_gt", "valid_gt_mask", "disparity_gt_right", "nonocc_mask")


class DatasetMode(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"

    @property
    def is_label_free(self) -> bool:
        return self is not DatasetMode.BENCHMARK


def assert_label_free(sample: Dict[str, Any], context: str = "training batch") -> None:
    """Raise if ``sample`` carries any ground-truth key.

    Cheap enough to run on every training batch, which is the point.
    """
    leaked = [key for key in GROUND_TRUTH_KEYS if key in sample]
    if leaked:
        raise RuntimeError(
            f"ground-truth keys {leaked} reached a {context}. Training must never see ground truth; "
            f"construct the dataset with DatasetMode.TRAIN or DatasetMode.VALIDATION.")


class StereoDataset(Dataset):
    """Base class for every stereo dataset in this repository.

    Subclasses implement:
        ``_num_samples()``
        ``_load_images(index) -> (left_hwc_float, right_hwc_float)``
        ``_load_ground_truth(index) -> dict``   (only called in BENCHMARK mode)
        ``_sample_metadata(index) -> dict``     (optional)

    Args:
        mode: see :class:`DatasetMode`.
        transform: callable applied to ``{"left", "right", ...}`` numpy arrays.
            In ``BENCHMARK`` mode this is expected to be ``None`` or identity:
            the benchmark protocol dictates the resolution, not augmentation.
        name: dataset name recorded in the metadata.
    """

    def __init__(self, mode: DatasetMode = DatasetMode.TRAIN, transform=None, name: str = "stereo"):
        self.mode = DatasetMode(mode)
        self.transform = transform
        self.name = name
        if self.mode is DatasetMode.BENCHMARK and transform is not None:
            raise ValueError("benchmark datasets must not be augmented; pass transform=None")

    # -- subclass hooks ----------------------------------------------------- #

    def _num_samples(self) -> int:
        raise NotImplementedError

    def _load_images(self, index: int):
        raise NotImplementedError

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError(f"{type(self).__name__} has no ground truth; it cannot be used for benchmarking")

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        return {}

    # -- dataset API -------------------------------------------------------- #

    def __len__(self) -> int:
        return self._num_samples()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        left, right = self._load_images(index)
        sample: Dict[str, Any] = {"left": left, "right": right}

        if self.transform is not None:
            sample = self.transform(sample)

        sample = {key: _to_chw_tensor(value) for key, value in sample.items()}
        metadata = {"dataset": self.name, "index": int(index)}
        metadata.update(self._sample_metadata(index))

        if self.mode is DatasetMode.BENCHMARK:
            ground_truth = self._load_ground_truth(index)
            for key, value in ground_truth.items():
                if key not in GROUND_TRUTH_KEYS:
                    raise RuntimeError(f"{type(self).__name__} returned unexpected ground-truth key {key!r}")
                sample[key] = _to_chw_tensor(value)
        else:
            assert_label_free(sample, context=f"{self.name} sample (mode={self.mode.value})")

        sample["metadata"] = metadata
        return sample


def _to_chw_tensor(array) -> torch.Tensor:
    """``(H, W)`` -> ``(1, H, W)`` and ``(H, W, C)`` -> ``(C, H, W)`` float32 tensor."""
    if torch.is_tensor(array):
        tensor = array
    else:
        array = np.ascontiguousarray(array)
        if array.dtype == np.bool_:
            array = array.astype(np.float32)
        tensor = torch.from_numpy(array)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    return tensor.to(torch.float32).contiguous()


def collate_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate that stacks tensors and keeps metadata as plain Python lists.

    Batching the metadata as lists rather than tensors avoids the usual mess of
    string fields in ``default_collate``, and keeps per-sample calibration easy
    to read back during evaluation.
    """
    batch: Dict[str, Any] = {}
    tensor_keys = [key for key in samples[0] if key != "metadata"]
    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    metadata_keys = set()
    for sample in samples:
        metadata_keys.update(sample.get("metadata", {}).keys())
    batch["metadata"] = {key: [sample.get("metadata", {}).get(key) for sample in samples]
                         for key in sorted(metadata_keys)}
    return batch


def metadata_tensor(metadata: Dict[str, List[Any]], key: str, default: Optional[float],
                    device=None) -> Optional[torch.Tensor]:
    """Pull a numeric metadata field out of a collated batch as a ``(B,)`` tensor.

    Returns ``None`` if the field is missing for any sample and no default is given.
    """
    values = metadata.get(key)
    if values is None:
        if default is None:
            return None
        values = [default]
    filled = []
    for value in values:
        if value is None:
            if default is None:
                return None
            value = default
        filled.append(float(value))
    return torch.tensor(filled, dtype=torch.float32, device=device)
