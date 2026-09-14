"""Synthetic test fixtures shared across multiple test modules.

Deliberately not named ``test_*.py``: pytest would try to collect it as a test
module in its own right, and a test module importing from *another* test
module depends on pytest's import-mode internals (whether ``tests`` resolves
as a namespace package, what pytest has already inserted into ``sys.path``,
whether some unrelated installed package also exposes a top-level ``tests``
module) -- behaviour that differs across Python/pytest versions and
environments. A plain helper module sidesteps all of that.
"""

import torch


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
