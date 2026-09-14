"""Architecture fidelity against the reference implementation.

These tests instantiate the original ``mmstereo`` model and compare it with this
re-implementation at identical settings.  They are skipped automatically when
the reference repository is not present (set ``MMSTEREO_PATH`` to point at it),
so a clone of this repository alone still has a green suite.

Measured result with the reference repository present, at the reference's own
``config_sceneflow.yaml`` model settings (num_disparities=256, downsample=4,
fe_features=16, fe_internal_features=16):

    reference             5,661,646 parameters
    this implementation   5,661,646 parameters
      feature extractor     708,000 / 708,000
      cost aggregation    1,857,848 / 1,857,848
      refinement          3,095,798 / 3,095,798
"""

import os
import sys

import pytest
import torch

from stereo.model import StereoNet, StereoNetConfig

DEFAULT_MMSTEREO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "mmstereo")
MMSTEREO_PATH = os.environ.get("MMSTEREO_PATH", DEFAULT_MMSTEREO)


def load_reference(num_disparities, downsample, features, backbone):
    """Import the reference model, stubbing the one helper it needs from its utils."""
    if not os.path.isdir(os.path.join(MMSTEREO_PATH, "models")):
        pytest.skip(f"reference implementation not found at {MMSTEREO_PATH}")

    import types
    if "utils" not in sys.modules:
        stub = types.ModuleType("utils")
        def split_outputs(outputs):
            half = outputs.shape[0] // 2
            return outputs[:half], outputs[half:]
        stub.split_outputs = split_outputs
        sys.modules["utils"] = stub

    if MMSTEREO_PATH not in sys.path:
        sys.path.insert(0, MMSTEREO_PATH)
    try:
        from models.hdrn_alpha_stereo import hdrn_alpha_stereo
    except Exception as error:                                   # pragma: no cover
        pytest.skip(f"could not import the reference model: {error}")

    hparams = types.SimpleNamespace(num_disparities=num_disparities,
                                    downsample_factor=downsample,
                                    fe_features=features,
                                    fe_internal_features=backbone)
    return hdrn_alpha_stereo(hparams)


def count(module):
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize("num_disparities,downsample", [(256, 4), (256, 8)])
def test_parameter_count_matches_the_reference(num_disparities, downsample):
    reference = load_reference(num_disparities, downsample, 16, 16)
    mine = StereoNet(StereoNetConfig(num_disparities=num_disparities, downsample=downsample,
                                     feature_channels=16, backbone_width=16,
                                     cost_volume_channels=4))
    assert count(mine) == count(reference)


def test_per_component_parameter_counts_match_the_reference():
    reference = load_reference(256, 4, 16, 16)
    mine = StereoNet(StereoNetConfig(num_disparities=256, downsample=4, feature_channels=16,
                                     backbone_width=16, cost_volume_channels=4))
    assert count(mine.feature_extractor) == count(reference.features) + count(reference.score_features)
    assert count(mine.aggregation) == count(reference.process_cost_volume)
    assert count(mine.refinement) == count(reference.disparity_refinement)


def test_output_shapes_and_disparity_bounds_match_the_reference():
    reference = load_reference(256, 4, 16, 16).eval()
    mine = StereoNet(StereoNetConfig(num_disparities=256, downsample=4, feature_channels=16,
                                     backbone_width=16, cost_volume_channels=4)).eval()
    left, right = torch.rand(1, 3, 128, 256), torch.rand(1, 3, 128, 256)
    with torch.no_grad():
        reference_output, _ = reference(left, right)
        my_output = mine.forward_left(left, right)

    for key in ("disparity", "disparity_small", "matchability"):
        assert my_output[key].shape == reference_output[key].shape, key
    assert mine.max_disparity == int(reference_output["max_disparity"])
    assert mine.max_disparity_small == int(reference_output["max_disparity_small"])
    assert mine.scale == int(reference_output["scale"])


def test_cost_volume_matches_the_reference_indexing():
    """Our single left-referenced volume must equal the reference's is_right=False branch."""
    if not os.path.isdir(os.path.join(MMSTEREO_PATH, "layers")):
        pytest.skip(f"reference implementation not found at {MMSTEREO_PATH}")
    if MMSTEREO_PATH not in sys.path:
        sys.path.insert(0, MMSTEREO_PATH)
    from layers.cost_volume import cost_volume as reference_cost_volume

    from stereo.geometry import flip_lr
    from stereo.model import correlation_volume

    generator = torch.Generator().manual_seed(0)
    left = torch.rand((1, 3, 4, 16), generator=generator)
    right = torch.rand((1, 3, 4, 16), generator=generator)

    assert torch.allclose(correlation_volume(left, right, 6),
                          reference_cost_volume(left, right, 6, False), atol=1e-6)

    # And the mirror trick must equal the reference's is_right=True branch, mirrored.
    mirrored = correlation_volume(flip_lr(right), flip_lr(left), 6)
    assert torch.allclose(flip_lr(mirrored),
                          reference_cost_volume(left, right, 6, True), atol=1e-6)


def test_dynamic_rule_diverges_from_the_papers_middlebury_choice():
    """Documented divergence: min(width // 2, 384) is below Middlebury's 512.

    The paper uses num_disparities=512 for Middlebury. The requested width rule
    caps at 384, which cannot represent larger disparities. The cap is
    configurable precisely so the paper's setting can be restored.
    """
    capped = StereoNetConfig.for_width(2872, downsample=8)
    assert capped.num_disparities == 384

    paper = StereoNetConfig.for_width(2872, downsample=8, max_disparities_cap=512)
    assert paper.num_disparities == 512
    assert StereoNet(paper).max_disparity == (512 // 8 - 1) * 8 - 1
