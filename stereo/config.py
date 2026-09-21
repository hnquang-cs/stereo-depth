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

    The disparity-space terms (``left_right``, ``pseudo``, ``range_penalty``) are
    applied to quantities **normalised by the disparity search range**, so these
    weights mean the same thing at any resolution or disparity range. Without
    that normalisation they are pixel-scale numbers being weighed against a
    photometric residual in ``[0, 1]``, which is how a left-right weight of 0.5
    ended up eight times stronger than the entire photometric signal and
    collapsed training to a constant disparity field.
    """
    photometric: float = 1.0
    smoothness: float = 0.1
    #: Applied to (left-right error / max_disparity).
    #:
    #: DEFAULT 0.0, set from measurement rather than from theory. A 4-weight,
    #: 3-seed sweep on a synthetic pair with known disparity found this term
    #: monotonically harmful -- correlation with the true disparity fell
    #: 0.606 -> 0.360 -> 0.273 -> 0.144 for weights 0.0, 0.25, 1.0, 4.0, and the
    #: photometric loss worsened alongside it. A constant disparity field is
    #: *exactly* left-right consistent while any real field is not, so the term
    #: rewards flatness, and nothing measured here offsets that.
    #:
    #: Caveat: that sweep is a single image pair with a deliberately small model
    #: over 400 steps, which has no occlusions and no generalisation pressure --
    #: precisely the conditions under which this regulariser would be expected to
    #: earn its keep. It may well help on real multi-image training; there is
    #: simply no evidence here that it does. Raise it if you can measure a gain.
    #:
    #: Left-right consistency is still computed and still used as a *signal* --
    #: for occlusion detection and pseudo-label filtering -- neither of which
    #: depends on this weight.
    left_right: float = 0.0
    #: Applied to (smooth-L1 teacher error / max_disparity).
    pseudo: float = 10.0
    confidence: float = 0.05
    #: Weight of the extra photometric/smoothness terms on the low-resolution
    #: (soft-argmin) disparity, which gives the cost volume a direct gradient.
    low_resolution: float = 0.5
    #: Cross-entropy from the network's cost volume to a photometric target.
    #: This is the label-free stand-in for the paper's NSCE loss, which anchors
    #: the cost volume at the ground-truth disparity. Without something in this
    #: role the cost volume receives only the indirect "move the mean" gradient
    #: through soft-argmin, and was measured not to learn at all: the coarse
    #: disparity sat at the midpoint of its search range while the refinement
    #: reduced the photometric loss by other means.
    cost_volume: float = 1.0
    #: Penalty on disparity predicted beyond the cost volume's search range.
    #: The refinement head is an unbounded ``relu(base + residual)``, so nothing
    #: in the architecture stops it emitting values the cost volume cannot
    #: support -- a randomly initialised model was measured emitting 20778 px
    #: against a 315 px range. Supervised training pulls those back via the
    #: ground-truth loss; label-free training has no such anchor.
    range_penalty: float = 0.1


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
