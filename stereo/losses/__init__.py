"""Losses.

Training is supervised: :class:`PaperObjective` is the paper's objective and
:mod:`stereo.losses.supervised` holds the two terms that read ground truth.
:class:`SmoothnessLoss` is a regulariser on the disparity field and reads
nothing.
"""

from .paper_objective import PaperLossWeights, PaperObjective, downsample_disparity
from .smoothness import SmoothnessLoss, gradient_x, gradient_y
from .supervised import DisparityLoss, NsceLoss, labels_from_batch, masked_mean

__all__ = ["PaperObjective", "PaperLossWeights", "downsample_disparity",
           "DisparityLoss", "NsceLoss", "labels_from_batch", "masked_mean",
           "SmoothnessLoss", "gradient_x", "gradient_y"]
