"""Run a trained model on stereo images of any size, ratio or resolution.

Why this module exists
----------------------
``num_disparities`` is a *construction-time* parameter: the aggregation stack
flattens the disparity axis into channels, so the search range is part of the
weights. A model built for 320 disparities cannot load weights trained with
112 -- ``load_state_dict`` refuses, with a shape mismatch on
``aggregation.conv2d.0.conv1.weight``.

That makes the ``min(width // 2, 384)`` policy incompatible with running one
model on arbitrary input: deriving the range from the input width means a
*different, incompatible network per resolution* (measured: 4,163,434
parameters at width 224 against 7,976,942 at width 960).

The way out is that disparity is a purely horizontal quantity that scales
*linearly* with horizontal resize. So for a fixed camera and scene,

    d / width

is invariant, while ``d`` is not. Fixing ``num_disparities`` at a declared
``canonical_width`` therefore fixes the *fraction* of image width the model
searches, and any other resolution is handled by resizing to that width and
scaling the answer back. Height needs no such treatment -- it never enters the
disparity -- so it is free, and the rest of the network is fully convolutional.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..geometry import RESIZE_ALIGN_CORNERS, resize_disparity

#: Smallest height worth feeding the encoder, which downsamples by 16.
MIN_HEIGHT = 64


def canonical_size(height: int, width: int, canonical_width: int,
                   preserve_aspect: bool = True) -> tuple:
    """The ``(height, width)`` to run the model at for an input of this size.

    Aspect ratio is preserved by default, so a 1242x375 input becomes 640x193
    rather than being squashed to the training height. Disparity is unaffected
    either way -- only the horizontal factor enters it -- but preserving the
    ratio keeps the *image statistics* closer to what the features expect.
    """
    if preserve_aspect:
        scaled = int(round(height * canonical_width / width))
        return max(scaled, MIN_HEIGHT), canonical_width
    return max(height, MIN_HEIGHT), canonical_width


@torch.no_grad()
def predict_disparity(model, left: torch.Tensor, right: torch.Tensor,
                      canonical_width: Optional[int] = None,
                      preserve_aspect: bool = True) -> Dict[str, torch.Tensor]:
    """Predict disparity for a stereo pair of any shape.

    The result is at the *input's* resolution and in the *input's* pixels, so it
    can be compared against ground truth for that image with no further
    rescaling.

    Args:
        model: a :class:`~stereo.model.stereo_net.StereoNet`.
        left, right: ``(B, 3, H, W)`` rectified pair, any ``H`` and ``W``.
        canonical_width: width to run at. Defaults to the model's own
            ``canonical_width``, which is the width its search range was
            declared at.
        preserve_aspect: see :func:`canonical_size`.

    Returns:
        ``{direction: {"disparity": (B, 1, H, W), "confidence": (B, 1, H, W)}}``.
    """
    if left.shape != right.shape:
        raise ValueError(f"view shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    height, width = left.shape[-2:]
    target_width = canonical_width or getattr(model, "canonical_width", width)

    run_height, run_width = canonical_size(height, width, target_width, preserve_aspect)
    if (run_height, run_width) != (height, width):
        resize = lambda t: F.interpolate(t, size=(run_height, run_width), mode="bilinear",
                                         align_corners=RESIZE_ALIGN_CORNERS)
        left, right = resize(left), resize(right)

    outputs = model(left, right)

    results = {}
    for direction, values in outputs.items():
        disparity = values["disparity"]
        if disparity.shape[-2:] != (height, width):
            # resize_disparity rescales the *values* by the width ratio as well
            # as resampling, which is exactly the inverse of the resize above.
            disparity = resize_disparity(disparity, (height, width))
        entry = {"disparity": disparity}
        if "confidence" in values:
            entry["confidence"] = F.interpolate(values["confidence"], size=(height, width),
                                                mode="bilinear", align_corners=RESIZE_ALIGN_CORNERS)
        results[direction] = entry
    return results
