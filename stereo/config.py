"""Configuration: plain dataclasses plus a small YAML loader.

No Hydra, no OmegaConf -- a nested dataclass tree and a recursive merge is all
this needs, and it keeps the defaults readable in one file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

import yaml

from .data.augmentation import GeometricAugmentConfig, PhotometricAugmentConfig, ResizeConfig
from .data.registry import DatasetSpec
from .losses.pseudo_label import PseudoLabelFilterConfig
from .model.stereo_net import StereoNetConfig
from .postprocess import PostProcessConfig


@dataclass
class LossWeights:
    """Weights of the label-free objective.  There is no ground-truth term.

    **The default is the Monodepth objective** (Godard et al. 2017, eq. 2):
    photometric + left-right + smoothness, and nothing else. That is the
    self-supervised half of what this package is: the paper's cost-volume
    architecture (arXiv:2109.11644) trained by Monodepth's self-supervised
    losses.

    The paper's own NSCE term is **absent, not replaced**. NSCE is anchored on
    ground-truth disparity, so it has no label-free form, and inventing a
    substitute for it would no longer be a reimplementation of the paper. The
    cost volume is therefore trained only by the gradient reaching it back
    through the soft-argmin.

    The remaining weights below are the paper's later stages (teacher
    self-training, matchability) and auxiliary regularisers. They default to 0.0
    so that the objective is exactly the three Monodepth terms; turn them on
    deliberately.

    The disparity-space terms (``left_right``, ``pseudo``, ``range_penalty``) are
    applied to quantities **normalised by the disparity search range**, so these
    weights mean the same thing at any resolution or disparity range. Without
    that normalisation they are pixel-scale numbers being weighed against a
    photometric residual in ``[0, 1]``, which is how a left-right weight of 0.5
    ended up eight times stronger than the entire photometric signal and
    collapsed training to a constant disparity field. It is also why Monodepth's
    ``a_lr = 1`` transfers: its disparity is a fraction of image width.
    """

    # -- the Monodepth objective ------------------------------------------- #
    photometric: float = 1.0        #: a_ap
    left_right: float = 1.0         #: a_lr, applied to (left-right error / max_disparity)
    smoothness: float = 0.1         #: a_ds

    # -- not part of it; off unless deliberately enabled -------------------- #
    #: Teacher pseudo-labels, the paper's Stage 2. Applied to
    #: (smooth-L1 teacher error / max_disparity).
    pseudo: float = 0.0
    #: Label-free matchability target. Not from either paper.
    confidence: float = 0.0
    #: The photometric/smoothness terms repeated on the low-resolution
    #: (soft-argmin) disparity -- the closest analogue of Monodepth's four-scale
    #: sum, and the only other route by which the cost volume gets a gradient.
    low_resolution: float = 0.0
    #: Keeps predictions inside the search range.
    range_penalty: float = 0.0

    @classmethod
    def monodepth(cls) -> "LossWeights":
        """The Monodepth objective: photometric + left-right + smoothness only.

        Godard et al. 2017, "Unsupervised Monocular Depth Estimation with
        Left-Right Consistency", eq. 2:

            C_s = a_ap (C_ap^l + C_ap^r) + a_ds (C_ds^l + C_ds^r)
                                         + a_lr (C_lr^l + C_lr^r)

        with ``a_ap = 1``, ``a_lr = 1``, ``a_ds = 0.1``, and the appearance term
        itself 0.85 SSIM + 0.15 L1 -- which is already
        :class:`~stereo.losses.PhotometricLoss`'s default.

        ``a_lr = 1`` is only meaningful because Monodepth's disparity is a
        *fraction of image width*, not a pixel count. This package's
        disparity-space terms are normalised by the search range for exactly that
        reason, so 1.0 here means what it means in the paper.

        Everything not in the paper is switched off: the teacher, the
        confidence target, the low-resolution copy and the range penalty.

        **This is the whole objective.** The paper's NSCE term is anchored on
        ground-truth disparity and so has no label-free form; it is simply absent
        rather than replaced. The cost volume is therefore trained only by the
        gradient that reaches it back through the soft-argmin.

        **Note.** Monodepth sums this over four output scales. The closest
        analogue here is ``low_resolution``, which applies the photometric and
        smoothness terms at cost-volume scale; it is 0.0 in this preset because
        the paper's objective as usually quoted has three terms. Set it to 1.0
        for a closer match to the paper's multi-scale behaviour.

        """
        return cls(photometric=1.0, left_right=1.0, smoothness=0.1,
                   low_resolution=0.0, pseudo=0.0, confidence=0.0,
                   range_penalty=0.0)


@dataclass
class TeacherConfig:
    """EMA teacher / student self-training (Stage 2)."""
    enabled: bool = True
    #: Epoch at which pseudo-labelling switches on. Before this, Stage 1 only.
    start_epoch: int = 10
    ema_decay: float = 0.999
    #: Epochs over which the pseudo-label weight ramps from 0 to its full value.
    ramp_epochs: int = 10
    filter: PseudoLabelFilterConfig = field(default_factory=PseudoLabelFilterConfig)
    #: Warn when the accepted fraction leaves this band (collapse monitor).
    min_valid_ratio: float = 0.05
    max_valid_ratio: float = 0.98


@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    momentum: float = 0.9
    #: "poly", "cosine" or "none".
    schedule: str = "poly"
    poly_exponent: float = 0.9
    warmup_iterations: int = 500
    grad_clip: float = 1.0


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    use_amp: bool = True
    seed: int = 1234
    #: Number of draws per epoch when weighted multi-dataset sampling is active.
    samples_per_epoch: Optional[int] = None
    log_every: int = 20
    #: Photometric-only warm-up before left-right consistency and confidence
    #: losses switch on, in iterations.
    warmup_iterations: int = 200
    output_dir: str = "outputs/train"
    #: Resume / initialise from this checkpoint (Stage 3 adaptation).
    init_checkpoint: Optional[str] = None
    resume: Optional[str] = None
    #: Label-free validation criterion used for "best" checkpoint selection.
    selection_metric: str = "val/photometric"
    max_steps_per_epoch: Optional[int] = None
    #: Save a left/right/disparity figure every N epochs (0 disables).
    visualize_every: int = 5
    #: How many stereo pairs to put in each figure.
    visualize_samples: int = 2


@dataclass
class DataConfig:
    train: List[DatasetSpec] = field(default_factory=list)
    validation: List[DatasetSpec] = field(default_factory=list)
    resize: ResizeConfig = field(default_factory=ResizeConfig)
    photometric_augmentation: PhotometricAugmentConfig = field(default_factory=PhotometricAugmentConfig)
    geometric_augmentation: GeometricAugmentConfig = field(default_factory=GeometricAugmentConfig)


@dataclass
class EvaluationConfig:
    protocol: str = "sceneflow"
    dataset_root: str = ""
    checkpoint: str = ""
    output_dir: str = "outputs/evaluation"
    max_samples: Optional[int] = None
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    save_visualizations: int = 0


@dataclass
class Config:
    model: StereoNetConfig = field(default_factory=StereoNetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    #: When true, ``model.num_disparities`` is recomputed from ``data.resize.width``
    #: with the min(width // 2, 384) policy.
    dynamic_disparity: bool = True


# --------------------------------------------------------------------------- #
# YAML <-> dataclass
# --------------------------------------------------------------------------- #

def _build(cls, value):
    """Recursively instantiate dataclass ``cls`` from a plain dict."""
    if value is None:
        return cls()
    if not isinstance(value, dict):
        return value

    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, raw in value.items():
        if key not in fields:
            raise ValueError(f"unknown config key {key!r} for {cls.__name__}; "
                             f"known keys: {sorted(fields)}")
        default = fields[key].default_factory() if fields[key].default_factory is not dataclasses.MISSING \
            else fields[key].default

        if is_dataclass(default) and isinstance(raw, dict):
            kwargs[key] = _build(type(default), raw)
        elif key in ("train", "validation") and isinstance(raw, list):
            kwargs[key] = [_build(DatasetSpec, item) for item in raw]
        elif isinstance(default, tuple) and isinstance(raw, list):
            kwargs[key] = tuple(raw)
        else:
            kwargs[key] = raw
    return cls(**kwargs)


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config into :class:`Config`, applying dotted-key overrides."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(raw, dotted, value)
    config = _build(Config, raw)

    if config.dynamic_disparity:
        config.model = StereoNetConfig.for_width(
            config.data.resize.width,
            downsample=config.model.downsample,
            max_disparities_cap=config.model.max_disparities_cap,
            feature_channels=config.model.feature_channels,
            backbone_width=config.model.backbone_width,
            cost_volume_channels=config.model.cost_volume_channels)
    return config


def _set_dotted(mapping: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = mapping
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def config_to_yaml(config: Config) -> str:
    return yaml.safe_dump(dataclasses.asdict(config), sort_keys=False, default_flow_style=False)
