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

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from .augmentation import PhotometricAugmentConfig, ResizeConfig, build_train_transform
from .base import DatasetMode, StereoDataset, collate_samples
from .eth3d import Eth3dDataset
from .hdf5_stereo import Hdf5StereoDataset, find_hdf5_files
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
    #: What logs and validation scores call it; the type (plus version) when empty.
    name: str = ""
    enabled: bool = True
    weight: float = 1.0
    #: Keep only the first ``fraction`` of the dataset (useful for quick runs).
    fraction: float = 1.0
    #: Hard cap on the number of pairs taken from this dataset. Applied after
    #: ``fraction``. ``None`` means no cap.
    max_samples: Optional[int] = None
    #: Extra keyword arguments forwarded to the dataset class (``split``, ``version``, ...).
    options: Dict[str, Any] = field(default_factory=dict)
    #: Which side of a train/validation division to take: "all", "train" or "val".
    #: The same spec with part="train" and part="val" gives two disjoint sets.
    part: str = "all"
    #: Share of the dataset on the "val" side (see :func:`holdout_indices`).
    val_fraction: float = 0.1
    #: Multiplies every disparity label: the correction for a mirror that resized
    #: its images without rescaling disparity correctly. Set it only to a factor
    #: :func:`stereo.data.label_check.measure_label_scale` measured.
    disparity_scale: float = 1.0
    #: Keep every sample in memory once decoded and resized; random augmentation
    #: still runs on each draw. For small sets of large files (Middlebury, KITTI),
    #: whose decoding otherwise keeps the GPU waiting. Costs memory per worker.
    cache: bool = False


def build_dataset(spec: DatasetSpec, mode: DatasetMode, transform=None,
                  with_labels: bool = False) -> StereoDataset:
    """Instantiate one dataset from its spec.

    ``with_labels`` asks a TRAIN/VALIDATION dataset to load ground truth as well,
    for supervised training. Datasets that have none simply return none.
    """
    spec = resolve_container(spec)
    if spec.type not in DATASET_TYPES:
        raise ValueError(f"unknown dataset type {spec.type!r}; known types: {sorted(DATASET_TYPES)}")
    dataset_class = DATASET_TYPES[spec.type]
    dataset = dataset_class(root=spec.root, mode=mode, transform=transform, **spec.options)
    # Set after construction rather than threaded through six loader signatures,
    # none of which would use it for anything but forwarding.
    dataset.with_labels = bool(with_labels)
    dataset.disparity_scale = float(spec.disparity_scale)
    dataset.cache_in_memory = bool(spec.cache)
    return dataset


def resolve_container(spec: DatasetSpec) -> DatasetSpec:
    """The spec to use for a Scene Flow mirror packaged as an HDF5 container.

    Mirrors of FlyingThings3D ship either a tree of images or one container
    holding every array, and the two need different loaders. Deciding here, from
    the data, means training and evaluation cannot disagree about it.
    """
    if spec.type == "sceneflow" and spec.root and find_hdf5_files(spec.root, max_depth=2):
        options = {key: spec.options[key] for key in ("split", "exclude") if key in spec.options}
        return replace(spec, type="hdf5", options=options)
    return spec


def spec_name(spec: DatasetSpec) -> str:
    """A short name for logs: the spec's own, else the type plus any version (kitti2015)."""
    return spec.name or f"{spec.type}{spec.options.get('version', '')}"


def holdout_indices(dataset, fraction: float) -> List[int]:
    """The indices on the validation side of a train/validation division.

    A dataset that knows which of its samples share a scene divides by scene
    (Middlebury does; see ``MiddleburyDataset.holdout_indices``). Anything else
    gives validation a contiguous block at the end: neighbouring frames of one
    sequence are nearly the same image, and a block keeps them together.
    """
    if hasattr(dataset, "holdout_indices"):
        return dataset.holdout_indices(fraction)
    count = min(len(dataset) - 1, max(1, round(len(dataset) * fraction)))
    return list(range(len(dataset) - count, len(dataset)))


def build_training_datasets(specs: Sequence[DatasetSpec], mode: DatasetMode,
                            resize: Optional[ResizeConfig],
                            photometric: Optional[PhotometricAugmentConfig],
                            seed: int = 0, with_labels: bool = False,
                            flip=None):
    """Build the enabled datasets plus the per-sample weights for weighted sampling.

    Returns ``(concat_dataset, sample_weights, summary)``.  ``sample_weights`` is
    ``None`` when only one dataset is enabled (a plain shuffle is then used).
    """
    if mode is DatasetMode.BENCHMARK:
        raise ValueError("build_training_datasets builds TRAIN/VALIDATION datasets; "
                         "use build_benchmark_dataset for BENCHMARK mode")

    datasets: List[Any] = []
    weights: List[float] = []
    summary: List[Dict[str, Any]] = []

    for index, spec in enumerate(specs):
        if not spec.enabled:
            continue
        transform = build_train_transform(resize, photometric if mode is DatasetMode.TRAIN else None,
                                      flip=flip if mode is DatasetMode.TRAIN else None,
                                          seed=seed + index)
        dataset = build_dataset(spec, mode, transform, with_labels)
        if spec.part not in ("all", "train", "val"):
            raise ValueError(f"part must be 'all', 'train' or 'val', not {spec.part!r}")
        if spec.part != "all":
            held = set(holdout_indices(dataset, spec.val_fraction))
            dataset = Subset(dataset, [i for i in range(len(dataset))
                                       if (i in held) == (spec.part == "val")])
        if spec.fraction < 1.0:
            keep = max(1, int(round(len(dataset) * spec.fraction)))
            dataset = Subset(dataset, range(keep))
        if spec.max_samples is not None and len(dataset) > spec.max_samples:
            dataset = Subset(dataset, range(max(1, spec.max_samples)))
        datasets.append(dataset)
        weights.append(spec.weight)
        summary.append({"name": spec_name(spec), "root": spec.root, "size": len(dataset),
                        "weight": spec.weight, "part": spec.part})

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
                 seed: int = 0, drop_last: bool = True, persistent: bool = True,
                 pin: Optional[bool] = None) -> DataLoader:
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
        extra = {"persistent_workers": persistent, "prefetch_factor": 4}
    pin_memory = torch.cuda.is_available() if pin is None else pin
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, collate_fn=collate_samples, drop_last=drop_last,
                      pin_memory=pin_memory, generator=generator, **extra)


def build_benchmark_dataset(spec: DatasetSpec) -> StereoDataset:
    """Instantiate a dataset in ``BENCHMARK`` mode -- the only mode that reads ground truth."""
    return build_dataset(spec, DatasetMode.BENCHMARK, transform=None)
