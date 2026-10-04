"""The paper's supervised objective, as config_sceneflow.yaml specifies it."""

import pytest
import torch

from stereo.losses import PaperLossWeights, PaperObjective, downsample_disparity
from stereo.model import StereoNet, StereoNetConfig


def test_weights_match_the_published_config():
    weights = PaperLossWeights()
    assert (weights.disparity, weights.nsce, weights.smoothness) == (100.0, 0.2, 20.0)
    assert weights.stdmean_scaled is True


def test_coarse_ground_truth_is_max_pooled_then_rescaled():
    """Max, not mean: averaging across a depth discontinuity gives a disparity
    that is true on neither side. The max takes the nearer surface."""
    block = torch.tensor([[[[8.0, 4.0], [2.0, 6.0]]]])
    assert float(downsample_disparity(block, 2)) == pytest.approx(8.0 / 2)
    assert torch.equal(downsample_disparity(block, 1), block)


def test_the_disparity_loss_is_supervised_at_both_scales():
    """Deep supervision: the full-resolution and the cost-volume outputs each get
    a disparity loss, which is what the reference does for every output it emits."""
    torch.manual_seed(0)
    net = StereoNet(StereoNetConfig(num_disparities=32, canonical_width=96)).eval()
    left, right = torch.rand(2, 3, 64, 96), torch.rand(2, 3, 64, 96)
    outputs = net(left, right, directions=("left",))["left"]
    truth = torch.full((2, 1, 64, 96), 8.0)
    valid = torch.ones_like(truth)

    parts = PaperObjective()(outputs, left, truth, valid, 4)["parts"]
    assert "disp_1x" in parts and "disp_4x" in parts
    assert "smooth" in parts and "nsce" in parts


def test_a_frame_with_too_little_ground_truth_is_skipped():
    """Below 3% labelled, the per-sample std/mean the loss divides by is
    meaningless, and the reference skips the frame outright."""
    objective = PaperObjective()
    disparity = torch.full((2, 1, 32, 32), 5.0)
    target = torch.full((2, 1, 32, 32), 9.0)

    dense = torch.ones_like(target)
    sparse = torch.zeros_like(target)
    sparse[..., :1, :1] = 1.0                      # 1 pixel of 1024, ~0.1%

    _, counted_dense = objective.disparity_loss(disparity, target, dense)
    _, counted_sparse = objective.disparity_loss(disparity, target, sparse)
    assert counted_dense == 2 and counted_sparse == 0


def test_stdmean_scaling_removes_the_advantage_of_being_close():
    """Without it a frame of large disparities dominates a batch by being nearer
    the camera rather than by being harder."""
    objective = PaperObjective()
    near = torch.full((1, 1, 32, 32), 100.0)
    far = torch.full((1, 1, 32, 32), 10.0)
    valid = torch.ones_like(near)

    # Each prediction is wrong by 10% of its own truth, so the two frames are
    # equally hard and should contribute comparably.
    scaled_near, _ = objective.disparity_loss(near * 1.1, near, valid)
    scaled_far, _ = objective.disparity_loss(far * 1.1, far, valid)
    plain = PaperObjective(PaperLossWeights(stdmean_scaled=False))
    raw_near, _ = plain.disparity_loss(near * 1.1, near, valid)
    raw_far, _ = plain.disparity_loss(far * 1.1, far, valid)

    # Not exactly equal: smooth-L1 is quadratic below 1 and linear above, so a
    # 10% error is not proportional across scales. What matters is that the
    # imbalance shrinks by an order of magnitude -- measured 19.0x to 1.9x.
    assert float(raw_near / raw_far) > 10.0
    assert float(scaled_near / scaled_far) < float(raw_near / raw_far) / 5


def test_the_objective_actually_trains():
    """A short overfit: the loss and EPE must fall substantially."""
    torch.manual_seed(0)
    net = StereoNet(StereoNetConfig(num_disparities=32, canonical_width=96))
    objective = PaperObjective()
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)

    left = torch.rand(1, 3, 64, 96)
    right = torch.roll(left, shifts=-6, dims=-1)
    truth = torch.full((1, 1, 64, 96), 6.0)
    valid = torch.ones_like(truth)

    first = last = None
    for step in range(60):
        outputs = net(left, right, directions=("left",))["left"]
        result = objective(outputs, left, truth, valid, 4)
        optimiser.zero_grad()
        result["loss"].backward()
        optimiser.step()
        if step == 0:
            first = result["logs"]["epe"]
        last = result["logs"]["epe"]
    assert last < first * 0.5, f"EPE did not fall: {first:.2f} -> {last:.2f}"


# -- the paper's augmentation ----------------------------------------------- #

def test_horizontal_flip_swaps_the_views_as_well_as_mirroring():
    """Mirroring both views of a stereo pair turns a left-referenced pair into a
    right-referenced one. The views must be swapped too, or the geometry is
    inverted and the disparity sign is wrong."""
    import numpy as np

    from stereo.data import HorizontalFlip, HorizontalFlipConfig

    flip = HorizontalFlip(HorizontalFlipConfig(probability=1.0), seed=0)
    left = np.arange(12, dtype=np.float32).reshape(3, 4)
    right = left + 100.0

    out = flip({"left": left.copy(), "right": right.copy()})
    assert np.allclose(out["left"], right[:, ::-1]), "the new left must be the mirrored RIGHT"
    assert np.allclose(out["right"], left[:, ::-1])


def test_horizontal_flip_refuses_a_labelled_pair_without_the_right_disparity():
    """After the swap the new left label is the OLD RIGHT view's disparity. With
    only a left label there is nothing correct to put there, so the sample is
    left alone rather than trained against the wrong view."""
    import numpy as np

    from stereo.data import HorizontalFlip, HorizontalFlipConfig

    flip = HorizontalFlip(HorizontalFlipConfig(probability=1.0), seed=0)
    left = np.arange(12, dtype=np.float32).reshape(3, 4)
    sample = {"left": left.copy(), "right": left + 100.0,
              "disparity_gt": np.full((3, 4), 5.0, np.float32)}
    assert np.allclose(flip(sample)["left"], left), "a left-only label must not be flipped"

    sample["disparity_gt_right"] = np.full((3, 4), 7.0, np.float32)
    flipped = flip(sample)
    assert float(flipped["disparity_gt"].mean()) == 7.0
    assert float(flipped["disparity_gt_right"].mean()) == 5.0


def test_horizontal_flip_swaps_the_valid_masks_with_the_labels():
    """The two views' unknown regions differ -- on up to 13% of pixels in
    Middlebury 2021 -- so the new left label needs the old RIGHT mask. nonocc
    describes only the old left view, so it is dropped."""
    import numpy as np

    from stereo.data import HorizontalFlip, HorizontalFlipConfig

    flip = HorizontalFlip(HorizontalFlipConfig(probability=1.0), seed=0)
    left_valid = np.ones((3, 4), np.float32)
    left_valid[:, 0] = 0
    right_valid = np.ones((3, 4), np.float32)
    right_valid[:, 3] = 0
    sample = {"left": np.zeros((3, 4), np.float32), "right": np.ones((3, 4), np.float32),
              "disparity_gt": np.full((3, 4), 5.0, np.float32), "valid_gt_mask": left_valid,
              "disparity_gt_right": np.full((3, 4), 7.0, np.float32),
              "valid_gt_mask_right": right_valid, "nonocc_mask": left_valid.copy()}
    out = flip(sample)
    assert np.array_equal(out["valid_gt_mask"], right_valid[:, ::-1])
    assert np.array_equal(out["valid_gt_mask_right"], left_valid[:, ::-1])
    assert "nonocc_mask" not in out

    # Without a right mask, the right label itself says where it is unknown.
    del sample["valid_gt_mask_right"]
    sample["disparity_gt_right"][:, 1] = 0.0
    out = flip(sample)
    assert float(out["valid_gt_mask"][:, ::-1][:, 1].sum()) == 0.0
    assert float(out["valid_gt_mask"].sum()) == 9.0


def test_resized_labels_stay_centred_on_their_images():
    """Plain nearest-neighbour rounds down, which put every resized label about
    half a target pixel up and left of its image (-0.50 px for KITTI's
    1242 -> 224). Resampling a map of each pixel's own column shows where each
    target pixel's label really came from."""
    import numpy as np

    from stereo.data.augmentation import ResizeConfig, ResizeSample

    for width, target in ((1242, 224), (960, 448), (741, 224)):
        columns = np.tile(np.arange(width, dtype=np.float32), (8, 1))
        sample = {"left": np.zeros((8, width, 3), np.float32), "disparity_gt": columns,
                  "valid_gt_mask": np.ones((8, width), np.float32)}
        out = ResizeSample(ResizeConfig(width=target, height=8, preserve_aspect=False))(sample)
        came_from = out["disparity_gt"][0] / (target / width)          # undo the value rescale
        centres = (np.arange(target) + 0.5) * width / target - 0.5
        offset = (came_from - centres) * target / width
        assert abs(float(offset.mean())) < 0.05, (width, target, float(offset.mean()))


def test_the_batch_rescale_keeps_labels_centred_too():
    import torch

    from stereo.data import BatchGeometricAugment, GeometricAugmentConfig

    augment = BatchGeometricAugment(GeometricAugmentConfig(scale=(0.7, 0.7), aspect=(1.0, 1.0),
                                                           size_divisor=1, min_size=8), seed=0)
    width = 300
    columns = torch.arange(width, dtype=torch.float32).view(1, 1, 1, -1).expand(1, 1, 30, width)
    out, (scale_x, _) = augment({"left": torch.zeros(1, 3, 30, width), "disparity_gt": columns.clone()})
    target = out["disparity_gt"].shape[-1]
    came_from = out["disparity_gt"][0, 0, 0] / scale_x
    centres = (torch.arange(target) + 0.5) * width / target - 0.5
    assert abs(float(((came_from - centres) * target / width).mean())) < 0.1   # was -0.45 px


def test_grids_are_matched_by_padding_not_stretching():
    import torch

    from stereo.losses.paper_objective import match_grid

    grid = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
    grown = match_grid(grid, (4, 5))
    assert torch.equal(grown[..., :3, :4], grid)               # every shared pixel unchanged
    assert float(grown[..., 3, :].abs().sum() + grown[..., :, 4].abs().sum()) == 0.0
    assert torch.equal(match_grid(grid, (2, 3)), grid[..., :2, :3])


def test_nsce_is_trained_on_the_same_coarse_target_as_the_coarse_loss():
    """The reference builds both from the max-pooled ground truth; re-shrinking
    the full-resolution labels by nearest-neighbour for NSCE differed from it."""
    import torch

    from stereo.losses import PaperObjective
    from stereo.losses.paper_objective import downsample_disparity

    objective = PaperObjective()
    seen = {}
    original = objective.nsce.forward

    def capture(cost, target, mask, downsample):
        seen.update(target=target, mask=mask, downsample=downsample)
        return original(cost, target, mask, downsample)

    objective.nsce.forward = capture
    disparity_gt = torch.rand(2, 1, 32, 64) * 20 + 1
    outputs = {"disparity": torch.rand(2, 1, 32, 64), "disparity_small": torch.rand(2, 1, 8, 16),
               "cost": torch.rand(2, 16, 8, 16)}
    objective(outputs, torch.rand(2, 3, 32, 64), disparity_gt, torch.ones(2, 1, 32, 64), 4)
    assert seen["downsample"] == 1
    assert torch.equal(seen["target"], downsample_disparity(disparity_gt, 4))
