"""Seeding, and an honest note about what remains non-deterministic."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

REMAINING_NONDETERMINISM = """\
Seeding covers Python, NumPy and PyTorch (CPU and CUDA). What is still not
bit-reproducible:
  * cuDNN picks algorithms by benchmarking unless torch.use_deterministic_algorithms
    is on; several ops used here (bilinear grid_sample backward, 3D convolution
    backward) have no deterministic CUDA kernel, so full determinism would force
    much slower code paths.
  * DataLoader workers interleave non-deterministically across processes.
  * Mixed precision changes accumulation order between runs on different hardware.
Set deterministic=True to trade speed for reproducibility where the kernels allow it.
"""


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
