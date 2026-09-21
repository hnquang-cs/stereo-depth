"""Label-free shaping of the cost volume.

The measured failure this addresses: with only the indirect gradient that
reaches the cost volume through soft-argmin, the coarse disparity sat at the
midpoint of its search range and moved 1.4 px in 600 steps, while the
refinement network drove the photometric loss to 0.0047 by other means. The
network reconstructed the image almost perfectly and matched nothing.
"""

import numpy as np
import pytest
import torch

from stereo.losses import CostVolumeLoss, photometric_cost_volume
from tests.helpers import make_shifted_pair


def test_photometric_cost_volume_finds_a_known_shift():
    """The whole idea rests on this: the true matching cost needs no labels."""
    shift = 9
    left, right = make_shifted_pair(height=32, width=128, shift=shift)
    volume, valid = photometric_cost_volume(left, right, 24, "left")

    # Ignore the left border, where no correspondence exists.
    interior = volume[:, :, :, shift + 12:]
    assert float(interior[:, shift].mean()) == pytest.approx(0.0, abs=1e-3)
    assert int(interior.mean(dim=(0, 2, 3)).argmin()) == shift
    # Every other candidate must be clearly worse.
    others = [float(interior[:, d].mean()) for d in range(24) if d != shift]
    assert min(others) > 0.05


def test_photometric_cost_volume_right_direction():
    shift = 7
    left, right = make_shifted_pair(height=32, width=128, shift=shift)
    volume, _ = photometric_cost_volume(right, left, 20, "right")
    interior = volume[:, :, :, 10:-shift - 12]
    assert int(interior.mean(dim=(0, 2, 3)).argmin()) == shift


def test_the_validity_mask_marks_the_border():
    left, right = make_shifted_pair(height=8, width=32, shift=4)
    _, valid = photometric_cost_volume(left, right, 6, "left")
    assert float(valid[:, 0].mean()) == 1.0
    assert float(valid[0, 5, :, :5].sum()) == 0.0    # d=5 invalidates the first 5 columns


def test_loss_is_lower_when_the_cost_volume_agrees_with_photometry():
    """A cost volume peaked at the true disparity must beat a flat one."""
    shift = 6
    left, right = make_shifted_pair(height=32, width=96, shift=shift)
    loss_fn = CostVolumeLoss()
    shape = (1, 16, 32, 96)

    flat = torch.zeros(shape)
    correct = torch.full(shape, 5.0)
    correct[:, shift] = 0.0
    wrong = torch.full(shape, 5.0)
    wrong[:, 13] = 0.0

    correct_loss = float(loss_fn(correct, left, right)["loss"])
    flat_loss = float(loss_fn(flat, left, right)["loss"])
    wrong_loss = float(loss_fn(wrong, left, right)["loss"])
    assert correct_loss < flat_loss < wrong_loss


def test_ambiguous_regions_are_excluded():
    """Textureless areas match every disparity equally and teach nothing; forcing
    the cost volume to imitate a flat target would inject noise."""
    uniform = torch.full((1, 3, 32, 96), 0.5)      # no texture at all
    cost = torch.zeros((1, 16, 32, 96))
    terms = CostVolumeLoss(min_confidence=0.05)(cost, uniform, uniform)
    assert float(terms["supervised_ratio"]) < 0.05, "a blank image must supervise almost nothing"

    left, right = make_shifted_pair(height=32, width=96, shift=6)
    textured = CostVolumeLoss(min_confidence=0.05)(cost, left, right)
    assert float(textured["supervised_ratio"]) > 0.5, "a textured pair must supervise most pixels"


def test_the_target_carries_no_gradient():
    left, right = make_shifted_pair(height=16, width=64, shift=5)
    cost = torch.zeros((1, 12, 16, 64), requires_grad=True)
    CostVolumeLoss()(cost, left, right)["loss"].backward()
    assert cost.grad is not None and torch.isfinite(cost.grad).all()


def test_the_loss_reads_no_ground_truth():
    """It is built from the two input images alone -- that is the point."""
    import inspect
    from stereo.losses import cost_volume_loss
    source = inspect.getsource(cost_volume_loss)
    for token in ("disparity_gt", "depth_gt", "valid_gt_mask"):
        assert token not in source
    parameters = set(inspect.signature(CostVolumeLoss.forward).parameters)
    assert parameters == {"self", "cost", "reference", "source", "direction"}


# --------------------------------------------------------------------------- #
# The two bugs that made this loss useless, and cost days to find
# --------------------------------------------------------------------------- #

def _blurred_shifted_pair(height, width, shift, blur, seed=0):
    """A pair with real image statistics: smooth texture, exact known shift."""
    import cv2
    rng = np.random.default_rng(seed)
    texture = cv2.GaussianBlur((rng.random((height, width + shift, 3)) * 255).astype(np.uint8),
                               (0, 0), blur).astype(np.float32) / 255.0
    tensor = torch.from_numpy(texture).permute(2, 0, 1)[None]
    return tensor[..., :width], tensor[..., shift:shift + width]


def _target_of(loss_fn, left, right, num_disparities, scale):
    """The soft target the loss builds, and the expectation soft-argmin takes."""
    import torch.nn.functional as F
    from stereo.losses import pooled_photometric_cost_volume
    volume, valid = pooled_photometric_cost_volume(left, right, num_disparities, scale, "left")
    masked = volume + (1.0 - valid) * 1e3
    best = masked.min(dim=1, keepdim=True).values
    spread = masked.masked_fill(valid < 0.5, float("nan"))
    spread = (torch.nanmean((spread - best).abs(), dim=1, keepdim=True)
              .nan_to_num(1.0).clamp(min=1e-6))
    target = F.softmin((masked - best) / spread / loss_fn.temperature, dim=1) * valid
    target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-6)
    index = torch.arange(num_disparities).view(1, -1, 1, 1).float()
    return target, (target * index).sum(dim=1)


def test_the_target_is_correct_in_its_EXPECTATION_not_just_its_argmin():
    """The bug that made this loss produce a confidently wrong disparity.

    soft-argmin takes the target's EXPECTATION. A target can have a near-perfect
    argmin and still be useless if a long tail drags its mean: measured, an
    argmin 0.78 px from the truth accompanied an expectation of 9.14 against a
    true disparity of 2.49. Training fitted that faithfully and produced exactly
    that wrong answer. Validate the statistic that is actually consumed.
    """
    shift, scale, num_disparities = 12, 4, 16
    left, right = make_shifted_pair(height=32, width=256, shift=shift)
    _, expectation = _target_of(CostVolumeLoss(), left, right, num_disparities, scale)

    interior = expectation[:, :, num_disparities + 8:]
    truth = shift / scale
    assert float((interior - truth).abs().mean()) < 1.0, (
        f"target expectation {float(interior.mean()):.2f} vs truth {truth:.2f}")


def test_a_soft_temperature_biases_the_expectation_toward_mid_range():
    """Guards the temperature choice: too soft and the target is useless even
    though its argmin stays perfect."""
    shift, scale, num_disparities = 12, 4, 16
    left, right = make_shifted_pair(height=32, width=256, shift=shift)
    truth = shift / scale

    sharp = _target_of(CostVolumeLoss(temperature=0.08), left, right, num_disparities, scale)[1]
    soft = _target_of(CostVolumeLoss(temperature=2.0), left, right, num_disparities, scale)[1]
    interior = slice(num_disparities + 8, None)

    sharp_error = float((sharp[:, :, interior] - truth).abs().mean())
    soft_error = float((soft[:, :, interior] - truth).abs().mean())
    assert sharp_error < soft_error / 2, (
        f"a soft temperature must visibly bias the expectation: "
        f"sharp {sharp_error:.2f} vs soft {soft_error:.2f}")


def test_matching_is_done_at_full_resolution_not_on_downsampled_images():
    """The other bug: downsampling destroys the texture that makes matching
    possible. Measured, matching 1/4-resolution images put the argmin 4.56 px
    from a truth of 1.5-4.5 px -- an error as large as the signal -- while
    pooling full-resolution evidence gave 0.78 px."""
    import torch.nn.functional as F
    from stereo.losses import photometric_cost_volume, pooled_photometric_cost_volume

    # A disparity that is NOT a multiple of the downsample factor, on a smooth
    # texture. With a clean multiple of 4 on white noise, downsampling happens to
    # preserve the shift exactly and both methods look fine -- the failure is
    # specific to sub-pixel low-resolution disparity and real image statistics.
    scale, num_disparities = 4, 16
    shift = 10                     # 2.5 px at 1/4 resolution
    left, right = _blurred_shifted_pair(height=64, width=256, shift=shift, blur=2.0)
    truth = shift / scale
    interior = slice(num_disparities + 8, None)

    pooled, _ = pooled_photometric_cost_volume(left, right, num_disparities, scale, "left")
    pooled_error = float((pooled[:, :, :, interior].argmin(dim=1).float() - truth).abs().mean())

    small = (F.interpolate(left, scale_factor=1 / scale, mode="bilinear", align_corners=False),
             F.interpolate(right, scale_factor=1 / scale, mode="bilinear", align_corners=False))
    naive, _ = photometric_cost_volume(*small, num_disparities, "left")
    naive_error = float((naive[:, :, :, interior].argmin(dim=1).float() - truth).abs().mean())

    assert pooled_error < naive_error, (
        f"pooling full-resolution evidence ({pooled_error:.2f} px) must beat matching "
        f"downsampled images ({naive_error:.2f} px)")
    assert pooled_error < 1.0


def test_the_temperature_is_scale_free():
    """A fixed absolute temperature does not survive a change of exposure or
    contrast; standardising the cost per pixel makes it invariant."""
    shift, scale, num_disparities = 12, 4, 16
    left, right = make_shifted_pair(height=32, width=256, shift=shift)
    loss_fn = CostVolumeLoss()
    interior = slice(num_disparities + 8, None)

    normal = _target_of(loss_fn, left, right, num_disparities, scale)[1][:, :, interior]
    # Halve the contrast and shift the exposure: the same scene, differently lit.
    dim = _target_of(loss_fn, left * 0.4 + 0.3, right * 0.4 + 0.3,
                     num_disparities, scale)[1][:, :, interior]
    assert float((normal - dim).abs().mean()) < 0.3, (
        "the target must not depend on image contrast or exposure")


@pytest.mark.parametrize("width", [96, 90, 91, 100])
@pytest.mark.parametrize("scale", [4, 8])
def test_works_when_the_width_is_not_a_multiple_of_the_scale(width, scale):
    """The model pads its input to a multiple of its stride and crops the output,
    so the cost volume's width is not always floor(W / scale). The target must
    line up with whatever the model produced."""
    left, right = make_shifted_pair(height=32, width=width, shift=6)
    for cost_width in (width // scale, width // scale + 1):
        cost = torch.zeros((1, 12, max(1, 32 // scale), cost_width))
        terms = CostVolumeLoss()(cost, left, right)
        assert torch.isfinite(terms["loss"])
