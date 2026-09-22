"""Building datasets and loaders from configuration.

Multi-dataset training draws from a :class:`~torch.utils.data.ConcatDataset` with
a :class:`~torch.utils.data.WeightedRandomSampler`.  Each sample of dataset ``i``
gets weight ``w_i / len(dataset_i)``, so the *expected fraction of draws* from
that dataset is exactly ``w_i / sum(w)`` regardless of how the dataset sizes
differ.  This is what makes a 15-scene Middlebury set usable next to a
20000-frame Scene Flow set.

Only stereo image pairs contribute to training; a dataset built here in a
training mode physically cannot return ground truth (see :mod:`stereo.data.base`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from .augmentation import PhotometricAugmentConfig, ResizeConfig, build_train_transform
from .base import DatasetMode, StereoDataset, collate_samples
from .eth3d import Eth3dDataset
from .hdf5_stereo import Hdf5StereoDataset
from .kitti import KittiStereoDataset
from .middlebury import MiddleburyDataset
from .sceneflow import SceneFlowDataset
from .stereo_folder import StereoFolderDataset

DATASET_TYPES = {
    "folder": StereoFolderDataset,
    "sceneflow": SceneFlowDataset,
    "kitti": KittiStereoDataset,
    "middlebury": MiddleburyDataset,
    "eth3d": Eth3dDataset,
    "hdf5": Hdf5StereoDataset,
}


@dataclass
class DatasetSpec:
    """One entry of a training mixture."""
    type: str = "folder"
    root: str = ""
    enabled: bool = True
    weight: float = 1.0
    #: Keep only the first ``fraction`` of the dataset (useful for quick runs).
    fraction: float = 1.0
    #: Hard cap on the number of pairs taken from this dataset. Applied after
    #: ``fraction``. ``None`` means no cap.
    max_samples: Optional[int] = None
    #: Extra keyword arguments forwarded to the dataset class (``split``, ``version``, ...).
    options: Dict[str, Any] = field(default_factory=dict)


def build_dataset(spec: DatasetSpec, mode: DatasetMode, transform=None) -> StereoDataset:
    """Instantiate one dataset from its spec."""
    if spec.type not in DATASET_TYPES:
        raise ValueError(f"unknown dataset type {spec.type!r}; known types: {sorted(DATASET_TYPES)}")
    dataset_class = DATASET_TYPES[spec.type]
    dataset = dataset_class(root=spec.root, mode=mode, transform=transform, **spec.options)
    return dataset


def build_training_datasets(specs: Sequence[DatasetSpec], mode: DatasetMode,
                            resize: Optional[ResizeConfig],
                            photometric: Optional[PhotometricAugmentConfig],
                            seed: int = 0):
    """Build the enabled datasets plus the per-sample weights for weighted sampling.

    Returns ``(concat_dataset, sample_weights, summary)``.  ``sample_weights`` is
    ``None`` when only one dataset is enabled (a plain shuffle is then used).
    """
    if mode is DatasetMode.BENCHMARK:
        raise ValueError("build_training_datasets is for label-free modes only")

    datasets: List[Any] = []
    weights: List[float] = []
    summary: List[Dict[str, Any]] = []

    for index, spec in enumerate(specs):
        if not spec.enabled:
            continue
        transform = build_train_transform(resize, photometric if mode is DatasetMode.TRAIN else None,
                                          seed=seed + index)
        dataset = build_dataset(spec, mode, transform)
        if spec.fraction < 1.0:
            keep = max(1, int(round(len(dataset) * spec.fraction)))
            dataset = Subset(dataset, range(keep))
        if spec.max_samples is not None and len(dataset) > spec.max_samples:
            dataset = Subset(dataset, range(max(1, spec.max_samples)))
        datasets.append(dataset)
        weights.append(spec.weight)
        summary.append({"name": spec.type, "root": spec.root, "size": len(dataset), "weight": spec.weight})

    if not datasets:
        raise RuntimeError("no enabled datasets")

    concat = ConcatDataset(datasets)
    if len(datasets) == 1:
        return concat, None, summary

    total_weight = sum(weights)
    sample_weights: List[float] = []
    for dataset, weight in zip(datasets, weights):
        per_sample = (weight / total_weight) / max(len(dataset), 1)
        sample_weights.extend([per_sample] * len(dataset))
    return concat, torch.tensor(sample_weights, dtype=torch.double), summary


def build_loader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 4,
                 sample_weights: Optional[torch.Tensor] = None, samples_per_epoch: Optional[int] = None,
                 seed: int = 0, drop_last: bool = True) -> DataLoader:
    """Data loader using :func:`~stereo.data.base.collate_samples`."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = None
    if sample_weights is not None:
        num_samples = samples_per_epoch or len(dataset)
        sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples,
                                        replacement=True, generator=generator)
        shuffle = False

    # persistent_workers keeps the worker pool alive between epochs: without it
    # every epoch pays process startup AND begins with a cold prefetch queue,
    # which shows up as the GPU stalling at each epoch boundary.
    # prefetch_factor deepens the queue so a slow sample does not stall the GPU.
    extra = {}
    if num_workers > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, collate_fn=collate_samples, drop_last=drop_last,
                      pin_memory=torch.cuda.is_available(), generator=generator, **extra)


def build_benchmark_dataset(spec: DatasetSpec) -> StereoDataset:
    """Instantiate a dataset in ``BENCHMARK`` mode -- the only mode that reads ground truth."""
    return build_dataset(spec, DatasetMode.BENCHMARK, transform=None)
