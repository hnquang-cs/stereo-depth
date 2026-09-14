"""Geometry: disparity sign, warping, rescaling, depth, padding.

These tests use synthetic data with an analytically known answer, so a failure
points at a derivation error rather than at a tuning problem.
"""

import numpy as np
import pytest
import torch

from stereo.geometry import (compute_num_disparities, depth_to_disparity, disparity_to_depth,
                             flip_lr, left_right_difference, pad_to_multiple, resize_disparity,
                             scale_disparity, unpad, warp_left_to_right, warp_right_to_left)


def make_shifted_pair(height=16, width=64, shift=7, seed=0):
    """Left/right pair related by an exact integer horizontal shift.

    Building the right image as ``I_R(x) = I_L(x + d)`` makes the left-referenced
    disparity exactly ``d``: the left pixel ``x`` matches the right pixel
    ``x - d`` because ``I_R(x - d) = I_L(x - d + d) = I_L(x)``.
    """
    generator = torch.Generator().manual_seed(seed)
    texture = torch.rand((1, 3, height, width + shift), generator=generator)
    left = texture[..., :width]
    right = texture[..., shift:shift + width]
    return left, right


def test_disparity_sign_convention_left():
    """I_L(x) = I_R(x - d): warping the right image left by d reconstructs the left."""
    shift = 7
    left, right = make_shifted_pair(shift=shift)
    disparity = torch.full((1, 1, left.shape[2], left.shape[3]), float(shift))

    reconstructed, valid = warp_right_to_left(right, disparity)
    # Only the region where x - d >= 0 has a real correspondence.
    region = slice(shift, None)
    assert torch.allclose(reconstructed[..., region], left[..., region], atol=1e-5)
    assert valid[..., :shift].sum() == 0
    assert valid[..., region].min() == 1.0


def test_disparity_sign_convention_right():
    """I_R(x) = I_L(x + d): warping the left image right by d reconstructs the right."""
    shift = 7
    left, right = make_shifted_pair(shift=shift)
    width = left.shape[3]
    disparity = torch.full((1, 1, left.shape[2], width), float(shift))

    reconstructed, valid = warp_left_to_right(left, disparity)
    region = slice(0, width - shift)
    assert torch.allclose(reconstructed[..., region], right[..., region], atol=1e-5)
    assert valid[..., width - shift:].sum() == 0


def test_wrong_sign_does_not_reconstruct():
    """A sign error must fail loudly, so assert the opposite warp is wrong."""
    shift = 7
    left, right = make_shifted_pair(shift=shift)
    disparity = torch.full((1, 1, left.shape[2], left.shape[3]), float(shift))
    wrong, _ = warp_left_to_right(right, disparity)  # deliberately the wrong direction
    error = (wrong[..., shift:-shift] - left[..., shift:-shift]).abs().mean()
    assert error > 0.05, "the wrong warp direction accidentally reconstructs the image"


def test_subpixel_warp_is_linear_interpolation():
    """A half-pixel shift must produce the average of the two neighbours."""
    ramp = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)
    disparity = torch.full((1, 1, 1, 8), 0.5)
    warped, _ = warp_right_to_left(ramp, disparity)
    expected = torch.tensor([0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).view(1, 1, 1, 8)
    assert torch.allclose(warped, expected, atol=1e-5)


def test_left_right_consistency_is_zero_for_consistent_disparities():
    """A constant fronto-parallel surface gives d_L = d_R, hence zero disagreement."""
    shift = 5.0
    shape = (1, 1, 8, 32)
    disparity_left = torch.full(shape, shift)
    disparity_right = torch.full(shape, shift)
    difference, valid = left_right_difference(disparity_left, disparity_right)
    assert torch.allclose(difference[valid > 0.5], torch.zeros_like(difference[valid > 0.5]), atol=1e-5)


def test_left_right_consistency_detects_disagreement():
    shape = (1, 1, 8, 32)
    difference, valid = left_right_difference(torch.full(shape, 5.0), torch.full(shape, 9.0))
    assert torch.allclose(difference[valid > 0.5], torch.full_like(difference[valid > 0.5], -4.0), atol=1e-5)


def test_left_right_consistency_on_a_slanted_surface():
    """d_R(x - d_L(x)) == d_L(x) must hold for a non-constant, self-consistent field.

    Take d_L(x) = a * x with 0 < a < 1.  Then x_R = x - a*x = (1-a)x, so
    d_R(x_R) = a*x = a/(1-a) * x_R.
    """
    width, a = 64, 0.25
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    disparity_left = (a * xs).expand(1, 1, 4, width).contiguous()
    disparity_right = ((a / (1 - a)) * xs).expand(1, 1, 4, width).contiguous()
    difference, valid = left_right_difference(disparity_left, disparity_right)
    assert valid.min() == 1.0
    assert difference.abs().max() < 1e-3


def test_scale_disparity_follows_horizontal_scale():
    disparity = torch.full((1, 1, 8, 32), 10.0)
    assert torch.allclose(scale_disparity(disparity, 2.0), torch.full_like(disparity, 20.0))
    assert torch.allclose(scale_disparity(disparity, 0.5), torch.full_like(disparity, 5.0))


@pytest.mark.parametrize("new_width,expected", [(64, 20.0), (16, 5.0), (32, 10.0)])
def test_resize_disparity_rescales_values(new_width, expected):
    disparity = torch.full((1, 1, 8, 32), 10.0)
    resized = resize_disparity(disparity, (8, new_width))
    assert resized.shape[-1] == new_width
    assert torch.allclose(resized, torch.full_like(resized, expected), atol=1e-4)


def test_resize_disparity_keeps_the_warp_correct():
    """Resizing image and disparity together must still reconstruct the image.

    A smooth (band-limited) texture is used on purpose: downsampling white noise
    aliases, and that would test the resampler rather than the disparity rescaling.
    """
    import torch.nn.functional as F
    from stereo.geometry import RESIZE_ALIGN_CORNERS

    height, width, shift = 16, 64, 8
    xs = torch.arange(width + shift, dtype=torch.float32)
    ys = torch.arange(height, dtype=torch.float32)
    texture = (torch.sin(2 * np.pi * xs / 32.0).view(1, 1, 1, -1)
               + torch.cos(2 * np.pi * ys / 16.0).view(1, 1, -1, 1)) * 0.25 + 0.5
    texture = texture.expand(1, 3, height, width + shift).contiguous()
    left = texture[..., :width]
    right = texture[..., shift:shift + width]
    disparity = torch.full((1, 1, height, width), float(shift))

    resize = lambda t: F.interpolate(t, size=(8, 32), mode="bilinear",
                                     align_corners=RESIZE_ALIGN_CORNERS)
    small_left, small_right = resize(left), resize(right)
    small_disparity = resize_disparity(disparity, (8, 32))
    assert float(small_disparity.mean()) == pytest.approx(shift * 32 / 64, abs=1e-4)

    reconstructed, valid = warp_right_to_left(small_right, small_disparity)
    error = ((reconstructed - small_left).abs() * valid).sum() / valid.sum().clamp(min=1)
    assert float(error) < 0.02, f"resized warp error {float(error):.4f}"


def test_resize_scale_x_matches_the_align_corners_convention():
    from stereo.geometry import resize_scale_x
    assert resize_scale_x(64, 32, align_corners=False) == pytest.approx(0.5)
    assert resize_scale_x(64, 32, align_corners=True) == pytest.approx(31 / 63)


def test_depth_conversion_roundtrip():
    focal, baseline = 1000.0, 0.12
    depth = torch.tensor([[[[1.0, 2.0, 5.0, 10.0]]]])
    disparity, valid = depth_to_disparity(depth, focal, baseline)
    assert torch.allclose(disparity, focal * baseline / depth, atol=1e-4)
    back, _ = disparity_to_depth(disparity, focal, baseline)
    assert torch.allclose(back, depth, atol=1e-3)
    assert valid.min() == 1.0


def test_depth_handles_degenerate_disparity():
    disparity = torch.tensor([[[[0.0, -1.0, float("nan"), float("inf"), 10.0]]]])
    depth, valid = disparity_to_depth(disparity, 1000.0, 0.1, max_depth=100.0)
    assert torch.isfinite(depth).all(), "degenerate disparity produced non-finite depth"
    assert valid.flatten().tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert depth[0, 0, 0, 4] == pytest.approx(10.0, abs=1e-4)


@pytest.mark.parametrize("width,expected", [(224, 112), (512, 256), (640, 320),
                                            (1024, 384), (1920, 384), (2560, 384)])
def test_dynamic_disparity_rule(width, expected):
    assert compute_num_disparities(width, downsample=4) == expected


def test_dynamic_disparity_alignment():
    """The result is always a multiple of the cost-volume downsample factor."""
    for width in range(64, 2048, 7):
        for downsample in (4, 8, 16):
            num = compute_num_disparities(width, downsample)
            assert num % downsample == 0
            assert num <= min(width // 2, 384) or num == downsample


def test_padding_roundtrip_preserves_x_coordinates():
    tensor = torch.rand((1, 3, 375, 1242))
    padded, padding = pad_to_multiple(tensor, 16)
    assert padded.shape[-2] % 16 == 0 and padded.shape[-1] % 16 == 0
    # Padding is on the right/bottom only, so original pixels keep their coordinates.
    assert torch.allclose(padded[..., :375, :1242], tensor)
    assert torch.allclose(unpad(padded, padding), tensor)


def test_flip_is_an_involution():
    tensor = torch.rand((1, 3, 4, 9))
    assert torch.allclose(flip_lr(flip_lr(tensor)), tensor)
