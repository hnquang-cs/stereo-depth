"""Visualisation helpers for training monitoring and benchmark error maps.

Sparse ground truth is never densified for display: invalid pixels are drawn as
a neutral grey so a sparse LiDAR panel looks sparse.
"""

from __future__ import annotations

import re
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Figures are built as matplotlib Figure objects, never through pyplot. pyplot
# needs a backend, and choosing one here (Agg, as this module once did) switches
# off a notebook's inline plots for every later cell -- checked in a Jupyter
# kernel. A Figure needs no backend to be saved, and display(figure) shows it.
try:
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
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
    figure = Figure(figsize=(6 * columns, 3.2 * rows))
    axes = figure.subplots(rows, columns, squeeze=False)
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


def _series(history: Sequence[Dict[str, float]], key: str) -> Optional[np.ndarray]:
    """One history key per epoch, NaN where an epoch lacks it; None if none has it."""
    values = [record.get(key) for record in history]
    if all(value is None for value in values):
        return None
    return np.array([np.nan if value is None else float(value) for value in values])


def plot_training_curves(history: Sequence[Dict[str, float]], path: Optional[str] = None):
    """A run's curves, one point per epoch, from its ``history.json``.

    Four panels: the training loss and its parts; training and validation EPE,
    with the epoch the best checkpoint came from marked; each dataset's
    validation EPE; the learning rate. A resumed run's history holds its
    earlier sessions too, so the curves cover the whole run.

    Returns the figure (``display(figure)`` shows it in a notebook) and writes it
    to ``path`` when one is given; None without matplotlib or history.
    """
    if not HAVE_MATPLOTLIB or not history:
        return None
    epochs = np.array([record.get("epoch", index) for index, record in enumerate(history)])
    figure = Figure(figsize=(19, 4.4), facecolor="white")
    loss_axis, epe_axis, dataset_axis, lr_axis = figure.subplots(1, 4)

    total = _series(history, "train/total")
    if total is not None:
        loss_axis.plot(epochs, total, marker="o", ms=3, lw=2.2, color="#222222", label="total")
    for part in ("disp_1x", "disp_4x", "nsce", "smooth"):
        values = _series(history, f"train/part/{part}")
        if values is not None:
            loss_axis.plot(epochs, values, lw=1.3, label=part)
    loss_axis.set_yscale("log", nonpositive="mask")
    loss_axis.set_title("training loss and its parts (log scale)")

    for key, label, colour in (("train/epe", "train", "#1a6fb5"), ("val/epe", "validation", "#b5541a")):
        values = _series(history, key)
        if values is not None:
            epe_axis.plot(epochs, values, marker="o", ms=3, lw=2, color=colour, label=label)
    validation = _series(history, "val/epe")
    if validation is not None and np.isfinite(validation).any():
        best = int(np.nanargmin(validation))
        epe_axis.plot([epochs[best]], [validation[best]], marker="*", ms=14, color="#b5541a", ls="none",
                      label=f"best: {validation[best]:.2f} px at epoch {epochs[best]} (best.pt)")
    epe_axis.set_title("EPE (px at the training width)")

    names = sorted({match.group(1) for record in history for key in record
                    for match in [re.fullmatch(r"val/([^/]+)/epe", key)] if match})
    for name in names:
        dataset_axis.plot(epochs, _series(history, f"val/{name}/epe"), marker="o", ms=3, lw=1.5,
                          label=name)
    dataset_axis.set_title("validation EPE by dataset (px)")

    rate = _series(history, "train/lr")
    if rate is not None:
        lr_axis.plot(epochs, rate, lw=2, color="#2e7d32")
    lr_axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    lr_axis.set_title("learning rate (end of epoch)")

    for axis in (loss_axis, epe_axis, dataset_axis, lr_axis):
        axis.set_xlabel("epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
        if not axis.lines:
            axis.text(0.5, 0.5, "not recorded", ha="center", va="center",
                      transform=axis.transAxes, color="#888888")

    seconds, waiting = _series(history, "train/seconds"), _series(history, "train/data_wait")
    note = f"{len(history)} epochs"
    if seconds is not None:
        note += f"   |   about {np.nanmedian(seconds) / 60:.1f} min of training per epoch"
    if waiting is not None:
        note += f"   |   waiting for data {100 * np.nanmedian(waiting):.0f}% of it"
    figure.suptitle(note, fontsize=12, color="#222222")
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    return figure


EvaluationRow = Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]


def _bar_name(label: str) -> str:
    """A table row's label, short enough to sit under its bar."""
    label = label.replace("this model, ", "").replace("paper, published", "paper (published)")
    return textwrap.fill(label, 14)


def _bar_colour(label: str) -> str:
    if label.startswith("paper"):
        return "#9e9e9e"
    if "floor" in label:
        return "#a5d6a7"
    if "int8" in label:
        return "#ef8a3a"
    return "#1a6fb5"


def plot_evaluation(rows: Sequence[EvaluationRow], per_image: Optional[List[Dict[str, float]]] = None,
                    path: Optional[str] = None, title: str = ""):
    """The benchmark result beside the paper's, with speed, size and the per-image spread.

    Args:
        rows: ``(label, epe, bad_1, size_mb, cpu_ms)`` per model, None where a
            quantity was not measured -- the rows of the notebook's comparison
            table.
        per_image: the summary's per-image metrics; their EPE spread is drawn.

    Returns the figure, and writes it to ``path`` when one is given.
    """
    if not HAVE_MATPLOTLIB or not rows:
        return None
    figure = Figure(figsize=(19, 4.6), facecolor="white")
    epe_axis, bad_axis, speed_axis, spread_axis = figure.subplots(1, 4)

    def bars(axis, column, unit, label_format):
        kept = [(row[0], row[column]) for row in rows if row[column] is not None
                and row[column] == row[column]]
        if not kept:
            axis.text(0.5, 0.5, "not measured", ha="center", va="center",
                      transform=axis.transAxes, color="#888888")
            return
        names = [_bar_name(name) for name, _ in kept]
        values = [value for _, value in kept]
        drawn = axis.bar(names, values, color=[_bar_colour(name) for name, _ in kept])
        axis.bar_label(drawn, labels=[label_format.format(value) for value in values], fontsize=9)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", labelsize=8)
        axis.margins(y=0.15)

    bars(epe_axis, 1, "px", "{:.3f}")
    epe_axis.set_title("EPE (lower is better)")
    bars(bad_axis, 2, "% of pixels", "{:.1f}")
    bad_axis.set_title("%bad(1.0): off by more than 1 px")

    timed = [row for row in rows if row[4] is not None]
    if timed:
        names = [_bar_name(row[0]) for row in timed]
        drawn = speed_axis.bar(names, [row[4] for row in timed],
                               color=[_bar_colour(row[0]) for row in timed])
        speed_axis.bar_label(drawn, labels=[f"{row[4]:.1f} ms" + (f"\n{row[3]:.1f} MB" if row[3] else "")
                                            for row in timed], fontsize=9)
        speed_axis.set_ylabel("ms per prediction")
        speed_axis.tick_params(axis="x", labelsize=8)
        speed_axis.margins(y=0.25)
    else:
        speed_axis.text(0.5, 0.5, "not timed", ha="center", va="center",
                        transform=speed_axis.transAxes, color="#888888")
    speed_axis.set_title("CPU latency and model size")

    errors = np.array([image["epe"] for image in per_image or [] if "epe" in image], dtype=float)
    if errors.size:
        spread_axis.hist(errors, bins=min(40, max(5, errors.size // 5)), color="#1a6fb5", alpha=0.85)
        spread_axis.axvline(np.median(errors), color="#222222", ls="--", lw=1)
        spread_axis.text(0.97, 0.95, f"median {np.median(errors):.2f} px\nworst {errors.max():.2f} px",
                         ha="right", va="top", transform=spread_axis.transAxes, fontsize=9)
        spread_axis.set_xlabel("EPE of one image (px)")
        spread_axis.set_ylabel("images")
    else:
        spread_axis.text(0.5, 0.5, "no per-image results", ha="center", va="center",
                         transform=spread_axis.transAxes, color="#888888")
    spread_axis.set_title("EPE per image, this model")

    for axis in (epe_axis, bad_axis, speed_axis, spread_axis):
        axis.grid(axis="y", alpha=0.3)
    if title:
        figure.suptitle(title, fontsize=12, color="#222222")
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    return figure
