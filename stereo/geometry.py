"""Stereo geometry: disparity conventions, warping, rescaling, depth conversion.

Everything in this module is derived, not guessed.  The derivations live in the
docstrings so that sign conventions can be checked by reading rather than by
running experiments.

Rectified stereo setup
----------------------
Left camera at the origin, right camera translated by the baseline ``B`` along
the camera ``+x`` axis.  Both cameras share focal length ``f`` and principal
point ``cx``.  A 3D point ``(X, Y, Z)`` projects to::

    x_L = f * X / Z           + cx
    x_R = f * (X - B) / Z     + cx

Subtracting::

    x_R = x_L - f * B / Z

So with the standard definition ``d = x_L - x_R`` we get ``d = f * B / Z >= 0``
for points in front of the cameras.  Two immediate consequences that fix every
sign in this file:

1. **Left-referenced disparity** ``d_L(x, y)``: the pixel ``(x, y)`` of the left
   image corresponds to ``(x - d_L(x, y), y)`` in the right image.  Therefore
   the left image is reconstructed from the right image by sampling *leftwards*::

       I_L_hat(x, y) = I_R(x - d_L(x, y), y)

2. **Right-referenced disparity** ``d_R(x, y)``: the pixel ``(x, y)`` of the
   right image corresponds to ``(x + d_R(x, y), y)`` in the left image, so::

       I_R_hat(x, y) = I_L(x + d_R(x, y), y)

Both disparities are non-negative and expressed in **pixels of the map they
belong to** (a disparity map at 1/4 resolution holds quarter-resolution pixels).

grid_sample normalisation
-------------------------
``align_corners=True`` maps pixel *centres* 0 and ``W-1`` to -1 and +1::

    u = 2 * x / (W - 1) - 1

which is an exact, invertible mapping for continuous ``x``.  We use it
everywhere so that a warp by an integer disparity is an exact pixel shift.
``align_corners=False`` (``u = (2x + 1)/W - 1``) would be equally valid but is
easier to get wrong, so it is deliberately not used for warping.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Disparity search range
# --------------------------------------------------------------------------- #

def compute_num_disparities(width: int, downsample: int = 4, max_disparities: int = 384) -> int:
    """Full-resolution disparity search range implied by an image width.

    This is the single place where the policy ``min(width // 2, 384)`` lives.

    NOTE: this width-driven rule is a *new implementation choice*, not the
    policy of the original paper.  The paper picks the range per dataset by
    hand (256 for Scene Flow/KITTI, 384 for the authors' camera, 512 for
    Middlebury).  The rule below automates that choice; the ``384`` cap matches
    the headline configuration of the paper.

    The result is floored to a multiple of ``downsample`` because the cost
    volume is built at ``1 / downsample`` resolution and needs an integral
    number of disparity levels.

        224 -> 112      512 -> 256      640 -> 320
       1024 -> 384     1920 -> 384

    Args:
        width: full-resolution image width in pixels.
        downsample: cost-volume downsample factor (4 or 8 in the paper).
        max_disparities: hard cap on the search range.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if downsample <= 0:
        raise ValueError(f"downsample must be positive, got {downsample}")
    num = min(width // 2, max_disparities)
    num = (num // downsample) * downsample
    return max(num, downsample)


# --------------------------------------------------------------------------- #
# Horizontal warping
# --------------------------------------------------------------------------- #

def _sample_at_x(source: torch.Tensor, sample_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ``source`` at absolute horizontal coordinates ``sample_x``.

    Args:
        source: ``(B, C, H, W)`` tensor to sample from.
        sample_x: ``(B, 1, H, W)`` absolute (sub-pixel) x coordinates.

    Returns:
        ``(warped, valid)`` where ``warped`` is ``(B, C, H, W)`` and ``valid`` is
        a ``(B, 1, H, W)`` float mask that is 1 where ``sample_x`` fell inside
        ``[0, W - 1]`` and 0 where it fell outside (out-of-range samples are
        clamped to the border, so their colour is meaningless).
    """
    batch, _, height, width = source.shape
    if sample_x.shape[0] != batch or sample_x.shape[-2:] != (height, width):
        raise ValueError(f"sample_x {tuple(sample_x.shape)} incompatible with source {tuple(source.shape)}")

    device, dtype = source.device, source.dtype
    sample_x = sample_x.to(dtype)

    valid = ((sample_x >= 0.0) & (sample_x <= width - 1)).to(dtype)

    # Normalise to [-1, 1] (align_corners=True convention).
    if width > 1:
        norm_x = 2.0 * sample_x / (width - 1) - 1.0
    else:
        norm_x = torch.zeros_like(sample_x)
    rows = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    if height > 1:
        norm_y = (2.0 * rows / (height - 1) - 1.0).expand_as(sample_x)
    else:
        norm_y = torch.zeros_like(sample_x)

    grid = torch.stack([norm_x.squeeze(1), norm_y.squeeze(1)], dim=-1)  # (B, H, W, 2)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped, valid


def _x_coords(disparity: torch.Tensor) -> torch.Tensor:
    _, _, _, width = disparity.shape
    xs = torch.arange(width, device=disparity.device, dtype=disparity.dtype)
    return xs.view(1, 1, 1, width)


def warp_right_to_left(right: torch.Tensor, disparity_left: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the left view from the right view.

    ``I_L_hat(x, y) = I_R(x - d_L(x, y), y)`` -- see the module docstring.

    Args:
        right: ``(B, C, H, W)`` right-view tensor (image, disparity, anything).
        disparity_left: ``(B, 1, H, W)`` left-referenced disparity, in pixels.

    Returns:
        ``(reconstructed_left, valid_mask)``.
    """
    return _sample_at_x(right, _x_coords(disparity_left) - disparity_left)


def warp_left_to_right(left: torch.Tensor, disparity_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the right view from the left view.

    ``I_R_hat(x, y) = I_L(x + d_R(x, y), y)`` -- see the module docstring.
    """
    return _sample_at_x(left, _x_coords(disparity_right) + disparity_right)


def flip_lr(tensor: torch.Tensor) -> torch.Tensor:
    """Horizontal mirror, ``flip(a)[x] = a[W - 1 - x]``."""
    return torch.flip(tensor, dims=[-1])


# --------------------------------------------------------------------------- #
# Left-right consistency
# --------------------------------------------------------------------------- #

def left_right_difference(disparity_left: torch.Tensor,
                          disparity_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Signed geometric disagreement of the two disparity maps, in the left frame.

    For a point visible in both views, the right-referenced disparity sampled at
    the corresponding right pixel must equal the left-referenced disparity::

        x_R      = x - d_L(x)
        d_R(x_R) = x_L - x_R = d_L(x)

    So ``d_L(x) - d_R(x - d_L(x))`` is zero for a consistent, non-occluded
    pixel.  Sampling ``d_R`` at ``x - d_L(x)`` is exactly ``warp_right_to_left``
    applied to the disparity map instead of the image.

    Returns:
        ``(difference, valid_mask)``, both ``(B, 1, H, W)``.
    """
    warped_right_disp, valid = warp_right_to_left(disparity_right, disparity_left)
    return disparity_left - warped_right_disp, valid


# --------------------------------------------------------------------------- #
# Resizing
# --------------------------------------------------------------------------- #

def scale_disparity(disparity: torch.Tensor, scale_x: float) -> torch.Tensor:
    """Rescale disparity *values* for a horizontal image scaling of ``scale_x``.

    Disparity is a difference of x coordinates, so if ``x -> x * sx`` then
    ``d -> d * sx``.  Interpolating a disparity map without this multiplication
    is a bug; that is why resizing and rescaling are bundled in
    :func:`resize_disparity`.
    """
    return disparity * scale_x


#: Resizing throughout this project uses ``align_corners=False`` (half-pixel
#: centres, the same convention as ``cv2.resize``), because that is the
#: convention whose coordinate map is a clean ``x_new = (x_old + 0.5) * s - 0.5``
#: -- differences of x then scale by exactly ``s = W_new / W_old``.  Under
#: ``align_corners=True`` the factor would instead be ``(W_new-1)/(W_old-1)``,
#: which is a different number and a classic source of sub-pixel disparity drift.
#: Warping is the one place that uses ``align_corners=True``, because there the
#: normalisation is done by hand and is exact (see :func:`_sample_at_x`).
RESIZE_ALIGN_CORNERS = False


def resize_scale_x(old_width: int, new_width: int, align_corners: bool = RESIZE_ALIGN_CORNERS) -> float:
    """Horizontal coordinate scale factor implied by a resize, hence the disparity factor."""
    if align_corners:
        return (new_width - 1) / max(old_width - 1, 1)
    return new_width / old_width


def resize_disparity(disparity: torch.Tensor, size: Tuple[int, int], mode: str = "bilinear",
                     align_corners: bool = RESIZE_ALIGN_CORNERS) -> torch.Tensor:
    """Resize a disparity map to ``size`` = ``(height, width)`` *and* rescale values.

    Values are multiplied by :func:`resize_scale_x`, which is the factor matching
    the interpolation convention actually used.  Resizing and rescaling are
    bundled here so they cannot drift apart.
    """
    _, _, _, old_width = disparity.shape
    new_height, new_width = size
    if mode == "nearest":
        resized = F.interpolate(disparity, size=size, mode="nearest")
    else:
        resized = F.interpolate(disparity, size=size, mode=mode, align_corners=align_corners)
    return scale_disparity(resized, resize_scale_x(old_width, new_width, align_corners))


# --------------------------------------------------------------------------- #
# Depth
# --------------------------------------------------------------------------- #

def disparity_to_depth(disparity: torch.Tensor,
                       focal_length: torch.Tensor | float,
                       baseline: torch.Tensor | float,
                       min_disparity: float = 1e-3,
                       max_depth: float = 1e4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Metric depth ``Z = f * B / d`` with explicit handling of degenerate disparity.

    Zero, negative, NaN and Inf disparities have no valid depth; they are
    reported through the returned mask and the depth is filled with ``max_depth``
    so that downstream code never sees Inf/NaN.

    Args:
        disparity: ``(B, 1, H, W)`` disparity in pixels of this map.
        focal_length: scalar or ``(B,)`` / ``(B, 1, 1, 1)`` focal length in the
            same pixel units as ``disparity``.
        baseline: scalar or per-batch baseline in metres.

    Returns:
        ``(depth, valid)``.
    """
    if not torch.is_tensor(focal_length):
        focal_length = torch.as_tensor(focal_length, dtype=disparity.dtype, device=disparity.device)
    if not torch.is_tensor(baseline):
        baseline = torch.as_tensor(baseline, dtype=disparity.dtype, device=disparity.device)
    focal_length = focal_length.reshape(-1, 1, 1, 1).to(disparity.dtype).to(disparity.device)
    baseline = baseline.reshape(-1, 1, 1, 1).to(disparity.dtype).to(disparity.device)

    finite = torch.isfinite(disparity)
    positive = disparity > min_disparity
    valid = finite & positive

    safe_disparity = torch.where(valid, disparity, torch.full_like(disparity, min_disparity))
    depth = focal_length * baseline / safe_disparity
    depth = torch.where(valid, depth, torch.full_like(depth, max_depth))
    depth = torch.clamp(depth, max=max_depth)
    return depth, valid.to(disparity.dtype)


def depth_to_disparity(depth: torch.Tensor,
                       focal_length: torch.Tensor | float,
                       baseline: torch.Tensor | float,
                       min_depth: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`disparity_to_depth`; returns ``(disparity, valid)``."""
    if not torch.is_tensor(focal_length):
        focal_length = torch.as_tensor(focal_length, dtype=depth.dtype, device=depth.device)
    if not torch.is_tensor(baseline):
        baseline = torch.as_tensor(baseline, dtype=depth.dtype, device=depth.device)
    focal_length = focal_length.reshape(-1, 1, 1, 1).to(depth.dtype).to(depth.device)
    baseline = baseline.reshape(-1, 1, 1, 1).to(depth.dtype).to(depth.device)

    valid = torch.isfinite(depth) & (depth > min_depth)
    safe_depth = torch.where(valid, depth, torch.full_like(depth, min_depth))
    disparity = focal_length * baseline / safe_depth
    disparity = torch.where(valid, disparity, torch.zeros_like(disparity))
    return disparity, valid.to(depth.dtype)


# --------------------------------------------------------------------------- #
# Padding for arbitrary input sizes
# --------------------------------------------------------------------------- #

def pad_to_multiple(tensor: torch.Tensor, divisor: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad ``(B, C, H, W)`` on the **right and bottom** to a multiple of ``divisor``.

    Padding on the right/bottom leaves every pixel's x coordinate unchanged, so
    disparity values are unaffected.  Padding on the *left* would shift x and
    silently corrupt disparity, which is why it is never done.

    Returns ``(padded, (pad_right, pad_bottom))``.
    """
    _, _, height, width = tensor.shape
    pad_bottom = (divisor - height % divisor) % divisor
    pad_right = (divisor - width % divisor) % divisor
    if pad_bottom == 0 and pad_right == 0:
        return tensor, (0, 0)
    padded = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    return padded, (pad_right, pad_bottom)


def unpad(tensor: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
    """Undo :func:`pad_to_multiple`."""
    pad_right, pad_bottom = padding
    height, width = tensor.shape[-2:]
    return tensor[..., : height - pad_bottom, : width - pad_right]
