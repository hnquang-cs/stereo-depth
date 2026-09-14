"""Visualisation helpers for training monitoring and benchmark error maps.

Sparse ground truth is never densified for display: invalid pixels are drawn as
a neutral grey so a sparse LiDAR panel looks sparse.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:  # pragma: no cover - plotting is optional
    HAVE_MATPLOTLIB = False


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """``(3, H, W)`` or ``(1, 3, H, W)`` float tensor -> ``(H, W, 3)`` uint8-ready array."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    return np.clip(tensor.detach().cpu().float().permute(1, 2, 0).numpy(), 0.0, 1.0)


def colorize(values: torch.Tensor, vmin: Optional[float] = None, vmax: Optional[float] = None,
             mask: Optional[torch.Tensor] = None, cmap: str = "magma",
             invalid_colour=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Colour-map a ``(1, H, W)`` / ``(1, 1, H, W)`` map, painting invalid pixels grey."""
    if not HAVE_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for colorize()")
    array = values.detach().cpu().float()
    while array.ndim > 2:
        array = array[0]
    array = array.numpy()

    valid = np.isfinite(array)
    if mask is not None:
        mask_array = mask.detach().cpu().float()
        while mask_array.ndim > 2:
            mask_array = mask_array[0]
        valid &= mask_array.numpy() > 0.5

    if vmin is None:
        vmin = float(np.percentile(array[valid], 2)) if valid.any() else 0.0
    if vmax is None:
        vmax = float(np.percentile(array[valid], 98)) if valid.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    normalised = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    coloured = matplotlib.colormaps[cmap](normalised)[..., :3]
    coloured[~valid] = invalid_colour
    return coloured


def save_evaluation_figure(path: str, panels: Dict[str, np.ndarray], title: str = "") -> None:
    """Grid figure of named panels; missing quantities are simply not passed in."""
    if not HAVE_MATPLOTLIB:
        return
    items = [(name, image) for name, image in panels.items() if image is not None]
    if not items:
        return
    columns = min(3, len(items))
    rows = int(np.ceil(len(items) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.2 * rows), squeeze=False)
    for index, (name, image) in enumerate(items):
        axis = axes[index // columns][index % columns]
        axis.imshow(image)
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    for index in range(len(items), rows * columns):
        axes[index // columns][index % columns].axis("off")
    if title:
        figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
