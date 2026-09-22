"""Augmentation that preserves stereo geometry.

Two separate stages, for two different reasons.

Per-sample photometric augmentation (:class:`PhotometricAugment`)
    Brightness/contrast/saturation/hue/gamma jitter.  By default the **same**
    random parameters are applied to the left and the right image, because
    independent jitter breaks the brightness-constancy assumption the
    photometric loss rests on.  Optional mild asymmetric jitter is available but
    off by default.  The un-jittered pair is kept as ``left_clean`` /
    ``right_clean`` and is what the photometric loss reconstructs, mirroring the
    reference implementation's "uncorrupted" images.

Per-batch geometric augmentation (:class:`BatchGeometricAugment`)
    Random scale, random aspect ratio and random resize.  It is applied to a
    whole batch at once, because the alternative way to combine a random scale
    with a fixed tensor shape is random cropping, which this project does not
    use (crop-based augmentation changes the effective field of view and, for
    left/right crops at different offsets, silently changes disparity).  One
    random output size per batch keeps the batch rectangular without cropping
    anything.

Intrinsics and disparity both scale with the horizontal resize factor; the
helpers here update the metadata so that ``depth = f * B / d`` keeps holding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch.nn.functional as F

from ..geometry import RESIZE_ALIGN_CORNERS


# --------------------------------------------------------------------------- #
# Per-sample transforms (numpy, HWC float in [0, 1])
# --------------------------------------------------------------------------- #

@dataclass
class PhotometricAugmentConfig:
    enabled: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.02
    gamma: Tuple[float, float] = (0.8, 1.2)
    probability: float = 0.5
    #: Probability of also applying a *small* independent jitter per view.  Keep
    #: low: it models real left/right gain mismatch but weakens brightness constancy.
    asymmetric_probability: float = 0.0
    asymmetric_scale: float = 0.05


#: Weights that turn RGB into the unweighted channel mean, as a cv2.transform
#: matrix. Built once: cv2.transform is ~35x faster than ``mean(axis=2)``.
_CHANNEL_MEAN = np.full((1, 3), 1.0 / 3.0, dtype=np.float32)


def _apply_jitter(image: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Stereo-consistent colour jitter.

    Uses OpenCV rather than numpy for the three heavy operations. They were
    measured, on a 432x640x3 float32 image, at:

        np.power  6.01 ms -> cv2.pow        1.76 ms
        mean(0,1) 3.38 ms -> cv2.mean       0.62 ms
        mean(2)   2.45 ms -> cv2.transform  0.07 ms

    which is ~9.4 ms per call, and this runs twice per pair. That mattered:
    profiling the loader showed colour jitter was 80-92% of all per-sample CPU
    time (17-43 ms of a 37 ms average pair), starving the GPU. The results agree
    to float32 rounding (max difference 2.3e-06), so this is a speed change only.
    """
    out = np.clip(image, 0.0, 1.0)
    if abs(params["gamma"] - 1.0) > 1e-6:
        out = cv2.pow(out, params["gamma"])
    out = out * params["brightness"]
    mean = np.asarray(cv2.mean(out)[:out.shape[2]], dtype=np.float32).reshape(1, 1, -1)
    out = (out - mean) * params["contrast"] + mean
    grey = cv2.transform(out, _CHANNEL_MEAN)[..., None]
    out = (out - grey) * params["saturation"] + grey
    if abs(params["hue"]) > 1e-6:
        hsv = cv2.cvtColor(np.clip(out, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + params["hue"] * 360.0) % 360.0
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class PhotometricAugment:
    """Stereo-consistent colour jitter producing clean and augmented image pairs."""

    def __init__(self, config: PhotometricAugmentConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = random.Random(seed)

    def _sample_params(self, scale: float = 1.0) -> Dict[str, float]:
        cfg = self.config
        jitter = lambda amount: 1.0 + self.rng.uniform(-amount, amount) * scale
        return {
            "brightness": jitter(cfg.brightness),
            "contrast": jitter(cfg.contrast),
            "saturation": jitter(cfg.saturation),
            "hue": self.rng.uniform(-cfg.hue, cfg.hue) * scale,
            "gamma": self.rng.uniform(*cfg.gamma) if scale >= 1.0 else 1.0,
        }

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        left, right = sample["left"], sample["right"]
        # The clean pair is the photometric-loss reference and the teacher input.
        sample = dict(sample)
        sample["left_clean"] = left.copy()
        sample["right_clean"] = right.copy()

        if not self.config.enabled or self.rng.random() > self.config.probability:
            return sample

        shared = self._sample_params()
        left_aug = _apply_jitter(left, shared)
        right_aug = _apply_jitter(right, shared)

        if self.rng.random() < self.config.asymmetric_probability:
            left_aug = _apply_jitter(left_aug, self._sample_params(self.config.asymmetric_scale))
            right_aug = _apply_jitter(right_aug, self._sample_params(self.config.asymmetric_scale))

        sample["left"] = left_aug
        sample["right"] = right_aug
        return sample


@dataclass
class ResizeConfig:
    """Training resize: fixed width, with the aspect ratio preserved by default.

    Only the **width** is canonical, because only the width affects disparity: a
    horizontal resize scales disparity by the same factor, a vertical one does
    not change it at all. So fixing the width fixes what the search range means,
    and the height is free to follow the source aspect ratio.

    ``height`` is then a *fallback* used when ``preserve_aspect`` is off, and the
    per-sample heights that come out of this are reconciled at batch level by
    :func:`stereo.data.base.collate_samples`, which pads to the batch maximum and
    emits a ``valid_mask``.

    Why preserve the ratio: inference already does
    (:func:`stereo.model.canonical_size`). Squashing 1242x375 KITTI to 640x384
    during training and then running it at 640x193 shows the network two
    different geometries for the same scene. Matching them removes that
    train/test mismatch.
    """
    height: int = 384
    width: int = 640
    preserve_aspect: bool = True
    #: Heights are rounded to this so the encoder never pads internally.
    size_divisor: int = 16


class ResizeSample:
    """Resize both views to the configured width (no cropping).

    With ``preserve_aspect`` the height follows the source ratio, so different
    samples come out at different heights; the collate pads them together.
    """

    def __init__(self, config: ResizeConfig):
        self.config = config

    def target_height(self, source_height: int, source_width: int) -> int:
        if not self.config.preserve_aspect:
            return self.config.height
        scaled = source_height * self.config.width / max(source_width, 1)
        divisor = self.config.size_divisor
        return max(int(round(scaled / divisor)) * divisor, divisor)

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        reference = sample.get("left")
        if reference is None:
            return dict(sample)
        height = self.target_height(reference.shape[0], reference.shape[1])
        target = (self.config.width, height)  # cv2 takes (w, h)
        out = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                out[key] = cv2.resize(value, target, interpolation=cv2.INTER_LINEAR)
            else:
                out[key] = value
        return out


class Compose:
    def __init__(self, transforms: List[Any]):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def build_train_transform(resize: Optional[ResizeConfig],
                          photometric: Optional[PhotometricAugmentConfig],
                          seed: Optional[int] = None) -> Optional[Compose]:
    transforms: List[Any] = []
    if resize is not None:
        transforms.append(ResizeSample(resize))
    if photometric is not None and photometric.enabled:
        transforms.append(PhotometricAugment(photometric, seed))
    else:
        transforms.append(_KeepClean())
    return Compose(transforms) if transforms else None


class _KeepClean:
    """Add ``left_clean``/``right_clean`` aliases when photometric jitter is off."""

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        sample = dict(sample)
        sample.setdefault("left_clean", sample["left"])
        sample.setdefault("right_clean", sample["right"])
        return sample


# --------------------------------------------------------------------------- #
# Per-batch geometric transform (torch, BCHW)
# --------------------------------------------------------------------------- #

@dataclass
class GeometricAugmentConfig:
    enabled: bool = True
    #: Multiplicative range applied to both axes together (random scale).
    scale: Tuple[float, float] = (0.8, 1.2)
    #: Extra multiplicative range applied to the width only (aspect-ratio change).
    aspect: Tuple[float, float] = (0.9, 1.1)
    #: Output sizes are rounded to this multiple so no internal padding is needed.
    size_divisor: int = 16
    min_size: int = 64
    keys: Tuple[str, ...] = ("left", "right", "left_clean", "right_clean", "valid_mask")


class BatchGeometricAugment:
    """Resize a whole batch to one randomly drawn size.

    Because it runs on the batch, every sample keeps its own content but shares
    the output geometry -- no cropping, no ragged tensors.  The scale factors are
    returned so that any disparity produced at the old size can be rescaled with
    :func:`stereo.geometry.scale_disparity`.
    """

    def __init__(self, config: GeometricAugmentConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = random.Random(seed)

    def _round(self, value: float) -> int:
        divisor = self.config.size_divisor
        rounded = int(round(value / divisor)) * divisor
        return max(rounded, self.config.min_size)

    def __call__(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Tuple[float, float]]:
        if not self.config.enabled:
            return batch, (1.0, 1.0)

        reference = batch["left"]
        _, _, height, width = reference.shape
        scale = self.rng.uniform(*self.config.scale)
        aspect = self.rng.uniform(*self.config.aspect)
        new_height = self._round(height * scale)
        new_width = self._round(width * scale * aspect)
        if (new_height, new_width) == (height, width):
            return batch, (1.0, 1.0)

        out = dict(batch)
        for key in self.config.keys:
            if key in batch and batch[key] is not None:
                out[key] = F.interpolate(batch[key], size=(new_height, new_width),
                                         mode="bilinear", align_corners=RESIZE_ALIGN_CORNERS)
        if out.get("valid_mask") is not None:
            # Resampling a 0/1 mask bilinearly blurs its edge; re-binarise so a
            # padded row never counts as partially real.
            out["valid_mask"] = (out["valid_mask"] > 0.999).to(batch["left"].dtype)
        scale_x = new_width / width
        scale_y = new_height / height
        out["metadata"] = _rescale_metadata(batch.get("metadata", {}), scale_x, scale_y)
        return out, (scale_x, scale_y)


def _rescale_metadata(metadata: Dict[str, List[Any]], scale_x: float, scale_y: float) -> Dict[str, List[Any]]:
    """Update pixel-unit calibration so ``depth = f * B / d`` survives a resize.

    ``f_x`` and ``c_x`` scale with the horizontal factor; ``f_y`` and ``c_y``
    with the vertical one.  The baseline is metric and does not change.
    """
    if not metadata:
        return metadata
    out = dict(metadata)
    for key, factor in (("focal_length", scale_x), ("principal_point_x", scale_x),
                        ("focal_length_y", scale_y), ("principal_point_y", scale_y)):
        if key in out and out[key] is not None:
            out[key] = [None if v is None else float(v) * factor for v in out[key]]
    return out
