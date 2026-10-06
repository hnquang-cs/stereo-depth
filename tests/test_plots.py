"""The run's figures: training curves and the evaluation summary.

And one property that matters more than either: plotting must leave a
notebook's display alone. Choosing a backend on import (Agg, as the plotting
module once did) switched off every later cell's inline plots -- checked in a
Jupyter kernel -- so the notebook showed nothing after training.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")

from stereo.utils.visualization import plot_evaluation, plot_training_curves

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _history(epochs=4, validation=True):
    records = []
    for epoch in range(epochs):
        record = {"epoch": epoch, "train/total": 150.0 / (epoch + 1), "train/epe": 10.0 / (epoch + 1),
                  "train/part/disp_1x": 80.0 / (epoch + 1), "train/part/disp_4x": 70.0 / (epoch + 1),
                  "train/part/nsce": 0.7, "train/part/smooth": 0.2, "train/lr": 1e-3 * (1 - epoch / epochs),
                  "train/seconds": 600.0, "train/data_wait": 0.05}
        if validation:
            record.update({"val/epe": 9.0 / (epoch + 1) + (2.0 if epoch == epochs - 1 else 0.0),
                           "val/sceneflow/epe": 8.0 / (epoch + 1), "val/middlebury/epe": 11.0 / (epoch + 1)})
        records.append(record)
    return records


def test_the_training_curves_are_drawn_and_saved(tmp_path):
    path = tmp_path / "training_curves.png"
    figure = plot_training_curves(_history(), str(path))

    assert path.stat().st_size > 10_000
    titles = [axis.get_title() for axis in figure.axes]
    assert titles == ["training loss and its parts (log scale)", "EPE (px at the training width)",
                      "validation EPE by dataset (px)", "learning rate (end of epoch)"]
    by_dataset = figure.axes[2].get_legend_handles_labels()[1]
    assert by_dataset == ["middlebury", "sceneflow"]
    # The checkpoint kept is the best validation epoch, not the last one.
    labels = figure.axes[1].get_legend_handles_labels()[1]
    assert "best: 3.00 px at epoch 2 (best.pt)" in labels    # 9/3; the last epoch is worse


def test_curves_survive_a_run_without_validation_or_with_gaps(tmp_path):
    history = _history(validation=False)
    del history[1]["train/lr"]
    figure = plot_training_curves(history, str(tmp_path / "curves.png"))
    assert (tmp_path / "curves.png").exists()
    assert "not recorded" in [text.get_text() for text in figure.axes[2].texts]
    assert plot_training_curves([], None) is None


def test_the_evaluation_figure_is_drawn_and_saved(tmp_path):
    rows = [("paper, published", 0.936, 10.0, None, None),
            ("this model, fp32 (cuda)", 1.8, 20.1, 15.7, 47.0),
            ("this model, dynamic int8", 1.8, 20.1, 15.7, 48.0),
            ("this model, static int8", None, None, None, None),        # unavailable here
            ("perfect at 224 px (floor)", 1.01, 8.8, None, None)]
    per_image = [{"sample_id": str(index), "epe": 1.0 + 0.1 * index} for index in range(30)]
    path = tmp_path / "evaluation_results.png"
    figure = plot_evaluation(rows, per_image, str(path), title="sceneflow, 30 images")

    assert path.stat().st_size > 10_000
    epe_bars = [patch.get_height() for patch in figure.axes[0].patches]
    assert epe_bars == pytest.approx([0.936, 1.8, 1.8, 1.01])                   # unmeasured left out
    assert len(figure.axes[2].patches) == 2                                    # only timed rows
    assert "median 2.45 px" in figure.axes[3].texts[0].get_text()


def test_plotting_leaves_a_notebook_s_display_alone(tmp_path):
    """'svg' stands in for a notebook's inline backend: it must still be the
    backend after the plotting code is imported and used."""
    code = (
        "import matplotlib\n"
        "matplotlib.use('svg')\n"
        "import numpy as np\n"
        "from stereo.utils.visualization import plot_training_curves, save_evaluation_figure\n"
        f"plot_training_curves([{{'epoch': 0, 'train/total': 1.0}}], {str(tmp_path / 'c.png')!r})\n"
        f"save_evaluation_figure({str(tmp_path / 'e.png')!r}, {{'map': np.zeros((4, 4, 3))}})\n"
        "print(matplotlib.get_backend())\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                            check=True)
    assert result.stdout.strip() == "svg"
    assert (tmp_path / "c.png").exists() and (tmp_path / "e.png").exists()
