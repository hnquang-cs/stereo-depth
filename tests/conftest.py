import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_pair(root, index, height, width, shift):
    """A stereo pair related by an exact horizontal shift, with its disparity."""
    import cv2

    generator = np.random.default_rng(index)
    texture = cv2.GaussianBlur(generator.random((height, width + shift, 3)).astype(np.float32),
                               (5, 5), 0)
    cv2.imwrite(os.path.join(root, "left", f"{index:03d}.png"),
                (texture[:, :width] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(root, "right", f"{index:03d}.png"),
                (texture[:, shift:shift + width] * 255).astype(np.uint8))
    # 16-bit PNG at 1/256 px, the convention the folder loader reads.
    cv2.imwrite(os.path.join(root, "left_disparity", f"{index:03d}.png"),
                np.full((height, width), shift * 256, np.uint16))


@pytest.fixture
def labelled_dataset(tmp_path):
    """A small stereo folder dataset WITH disparity.

    Training is supervised, so the shared fixture provides labels; a dataset
    without them has nothing for the objective to optimise.
    """
    root = tmp_path / "stereo"
    for side in ("left", "right", "left_disparity"):
        (root / side).mkdir(parents=True)
    for index in range(6):
        _write_pair(str(root), index, height=48, width=96, shift=6)
    return str(root)


#: The old name, from when training was label-free. Kept so tests that only need
#: images do not have to care.
@pytest.fixture
def unlabeled_dataset(labelled_dataset):
    return labelled_dataset
