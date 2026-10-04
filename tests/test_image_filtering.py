"""Training and test images reach the network filtered alike, and the answer
returns to full size in a chosen way.

Training resizes with OpenCV in the loader; evaluation resizes tensors in
torch. If the two filtered differently, the network would be tested on images
unlike the ones it learned from. Measured on FlyingThings3D at 960x540, plain
bilinear made the test images 13% harder to match than the training images.
"""

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from stereo.data.augmentation import ResizeConfig, ResizeSample
from stereo.geometry import resize_images
from stereo.model import canonical_size, predict_disparity


def _tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]


def _array(tensor):
    return tensor[0].numpy().transpose(1, 2, 0)


@pytest.mark.parametrize("source, target", [
    ((540, 960), (126, 224)),       # FlyingThings3D TEST at the network's width
    ((375, 1242), (64, 224)),       # KITTI
    ((270, 480), (128, 224)),       # the prepared FlyingThings3D TRAIN
    ((540, 960), (270, 480)),       # a whole factor, OpenCV's separate fast path
    ((497, 741), (144, 224)),       # a prepared Middlebury scene
    ((100, 224), (96, 224)),        # one axis kept
])
def test_shrinking_matches_opencv_area_exactly(source, target):
    image = np.random.default_rng(0).random(source + (3,)).astype(np.float32)
    ours = _array(resize_images(_tensor(image), target))
    theirs = cv2.resize(image, target[::-1], interpolation=cv2.INTER_AREA)
    assert np.abs(ours - theirs).max() < 1e-5


def test_enlarging_stays_bilinear():
    image = torch.rand(1, 3, 40, 100)
    expected = F.interpolate(image, size=(48, 224), mode="bilinear", align_corners=False)
    assert torch.equal(resize_images(image, (48, 224)), expected)
    assert resize_images(image, (40, 100)) is image


def test_the_training_resize_and_the_test_resize_agree():
    """The same image, resized by the loader and by inference, to the same size."""
    image = np.random.default_rng(1).random((512, 896, 3)).astype(np.float32)
    trained = ResizeSample(ResizeConfig(width=224, preserve_aspect=True))({"left": image})["left"]
    run_size = canonical_size(512, 896, 224)
    assert trained.shape[:2] == run_size == (128, 224)
    tested = _array(resize_images(_tensor(image), run_size))
    assert np.abs(trained - tested).max() < 1e-5


def test_shrinking_does_not_alias():
    """A one-pixel checkerboard has no detail left at 960 -> 224. Bilinear reads
    only the four pixels nearest each centre and invents a pattern from it."""
    board = (np.indices((540, 960)).sum(0) % 2).astype(np.float32)
    board = np.repeat(board[..., None], 3, axis=2)
    shrunk = ResizeSample(ResizeConfig(width=224))({"left": board})["left"]
    assert shrunk.std() < 0.01 and shrunk.mean() == pytest.approx(0.5, abs=0.01)
    bilinear = cv2.resize(board, shrunk.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    assert bilinear.std() > 0.1                  # what training used to see (measured 0.16)


class _StepModel(torch.nn.Module):
    """Answers 4 px left of the middle and 8 px right of it, at any size."""

    canonical_width = 32

    def forward(self, left, right, directions=("left",)):
        disparity = torch.full_like(left[:, :1], 4.0)
        disparity[..., left.shape[-1] // 2:] = 8.0
        return {"left": {"disparity": disparity}}


def test_nearest_upsampling_keeps_a_depth_edge():
    views = torch.rand(1, 3, 40, 128), torch.rand(1, 3, 40, 128)
    nearest = predict_disparity(_StepModel(), *views, upsample="nearest")["left"]["disparity"]
    assert sorted(torch.unique(nearest).tolist()) == [16.0, 32.0]      # x4 back to 128 px
    bilinear = predict_disparity(_StepModel(), *views)["left"]["disparity"]
    assert len(torch.unique(bilinear)) > 2                              # blended across it
    with pytest.raises(ValueError, match="upsample"):
        predict_disparity(_StepModel(), *views, upsample="bicubic")
