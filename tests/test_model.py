"""Model: arbitrary resolutions, the mirror trick, and the cost-volume indexing."""

import pytest
import torch

from stereo.geometry import flip_lr
from stereo.model import StereoNet, StereoNetConfig, correlation_volume, matchability, soft_argmin


def small_config(width=224, downsample=4):
    return StereoNetConfig.for_width(width, downsample=downsample, backbone_width=8,
                                     feature_channels=8)


@pytest.mark.parametrize("height,width", [(224, 224), (240, 320), (384, 384), (192, 640)])
def test_forward_at_various_resolutions(height, width):
    model = StereoNet(small_config(width)).eval()
    left, right = torch.rand(1, 3, height, width), torch.rand(1, 3, height, width)
    with torch.no_grad():
        output = model.forward_left(left, right)
    assert output["disparity"].shape == (1, 1, height, width)
    assert output["disparity_small"].shape == (1, 1, height // 4, width // 4)
    assert torch.isfinite(output["disparity"]).all()
    assert (output["disparity"] >= 0).all(), "disparity must be non-negative by construction"


@pytest.mark.parametrize("height,width", [(375, 1242), (540, 960), (100, 150)])
def test_forward_with_padding(height, width):
    """Sizes that are not multiples of 16 must go through pad -> infer -> unpad."""
    model = StereoNet(small_config(320)).eval()
    left, right = torch.rand(1, 3, height, width), torch.rand(1, 3, height, width)
    with torch.no_grad():
        output = model.forward_left(left, right)
    assert output["disparity"].shape == (1, 1, height, width)
    assert torch.isfinite(output["disparity"]).all()


def test_bidirectional_outputs_have_matching_shapes():
    model = StereoNet(small_config(224)).eval()
    left, right = torch.rand(1, 3, 96, 224), torch.rand(1, 3, 96, 224)
    with torch.no_grad():
        outputs = model(left, right, directions=("left", "right"))
    assert set(outputs) == {"left", "right"}
    assert outputs["left"]["disparity"].shape == outputs["right"]["disparity"].shape


def test_confidence_is_exp_matchability_and_bounded():
    model = StereoNet(small_config(224)).eval()
    with torch.no_grad():
        output = model.forward_left(torch.rand(1, 3, 64, 224), torch.rand(1, 3, 64, 224))
    assert torch.allclose(output["confidence"], torch.exp(output["matchability"]), atol=1e-6)
    num_disparities = output["cost"].shape[1]
    assert output["confidence"].min() >= 1.0 / num_disparities - 1e-5
    assert output["confidence"].max() <= 1.0 + 1e-5


def test_cost_volume_indexing():
    """volume[:, :, d, :, x] must be ref[x] * src[x - d], and zero for x < d."""
    reference = torch.arange(1, 7, dtype=torch.float32).view(1, 1, 1, 6)
    source = torch.arange(10, 70, 10, dtype=torch.float32).view(1, 1, 1, 6)
    volume = correlation_volume(reference, source, num_disparities=3)
    assert volume.shape == (1, 1, 3, 1, 6)
    for d in range(3):
        for x in range(6):
            expected = reference[0, 0, 0, x] * source[0, 0, 0, x - d] if x >= d else 0.0
            assert volume[0, 0, d, 0, x] == pytest.approx(float(expected))


def test_mirror_trick_gives_the_right_referenced_volume():
    """The derivation in stereo/model/cost_volume.py, checked numerically.

    corr(flip(src), flip(ref))[d, x] must equal src[x'] * ref[x' + d] at
    x' = W - 1 - x, i.e. the right-referenced cost volume, mirrored.
    """
    width, num_disparities = 12, 4
    generator = torch.Generator().manual_seed(0)
    reference = torch.rand((1, 2, 1, width), generator=generator)
    source = torch.rand((1, 2, 1, width), generator=generator)

    mirrored = correlation_volume(flip_lr(source), flip_lr(reference), num_disparities)
    unmirrored = flip_lr(mirrored)

    for d in range(num_disparities):
        for x in range(width):
            expected = (source[0, :, 0, x] * reference[0, :, 0, x + d]
                        if x + d < width else torch.zeros(2))
            assert torch.allclose(unmirrored[0, :, d, 0, x], expected, atol=1e-6), (d, x)


def test_soft_argmin_peaks_at_the_minimum_cost():
    """A cost volume with a sharp minimum at index k must regress to k."""
    num_disparities = 16
    for k in (0, 3, 9, 15):
        cost = torch.full((1, num_disparities, 1, 1), 50.0)
        cost[0, k] = 0.0
        assert float(soft_argmin(cost)) == pytest.approx(float(k), abs=1e-3)


def test_matchability_is_negative_entropy():
    num_disparities = 8
    peaked = torch.full((1, num_disparities, 1, 1), 100.0)
    peaked[0, 2] = 0.0
    flat = torch.zeros((1, num_disparities, 1, 1))

    assert float(torch.exp(matchability(peaked))) == pytest.approx(1.0, abs=1e-4)
    assert float(torch.exp(matchability(flat))) == pytest.approx(1.0 / num_disparities, abs=1e-5)
    assert float(matchability(flat)) < float(matchability(peaked))


def test_checkpoint_roundtrip_rebuilds_the_architecture(tmp_path):
    from stereo.utils.checkpoint import build_model_from_checkpoint, save_checkpoint
    model = StereoNet(small_config(512, downsample=8))
    path = str(tmp_path / "model.pt")
    save_checkpoint(path, model)
    restored = build_model_from_checkpoint(path)
    assert restored.num_disparities == model.num_disparities
    assert restored.scale == model.scale
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_num_disparities_is_baked_into_the_weights():
    """Documented limitation: the search range is part of the architecture."""
    narrow = StereoNet(small_config(224))
    wide = StereoNet(small_config(1024))
    assert narrow.num_disparities == 112 and wide.num_disparities == 384
    with pytest.raises(RuntimeError):
        wide.load_state_dict(narrow.state_dict())
