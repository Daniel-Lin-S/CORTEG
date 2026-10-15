"""Render five-finger saved-prediction examples.

Input: one interval of original glove targets, model predictions, model
labels, time coordinates, and shared finger limits. Output is one PNG with
five stacked finger panels and a dedicated legend column.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plotting.fingerflex_comparison import STYLES

FIGURE_SIZE = (18, 13)
DPI = 180
TITLE_SIZE = 22
AXIS_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 17
TRUTH_WIDTH = 2.8
MODEL_WIDTH = 1.8
FINGER_COUNT = 5
PANEL_COLUMNS = 2
LIMIT_COLUMNS = 2
PANEL_WIDTH_RATIOS = (3.3, 1.7)
GRID_ALPHA = 0.18


def plot_prediction_window(
    output: Path,
    subject: str,
    interval_number: int,
    time_seconds: np.ndarray,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    labels: dict[str, str],
    limits: np.ndarray,
) -> None:
    """Save one original-value trace figure for all five fingers.

    Parameters
    ----------
    output : Path
        Destination PNG path.
    subject : str
        Stanford subject code.
    interval_number : int
        One-based interval number within the subject.
    time_seconds : ndarray, shape (rows,)
        Time since interval start.
    truth : ndarray, shape (rows, 5)
        Original glove target values.
    predictions : dict[str, ndarray]
        Original-value model predictions, each shaped ``(rows, 5)``.
    labels : dict[str, str]
        Feature-explicit display name per model.
    limits : ndarray, shape (5, 2)
        Matched y-axis limits across comparison roots.
    """
    if not predictions or set(predictions) != set(labels):
        raise ValueError("Every plotted model needs one prediction and label.")
    if truth.ndim != 2 or truth.shape[1] != FINGER_COUNT or (
        time_seconds.shape != (len(truth),)
    ) or limits.shape != (FINGER_COUNT, LIMIT_COLUMNS):
        raise ValueError("Expected five aligned finger traces and limits.")
    if not np.isfinite(truth).all() or not np.isfinite(limits).all():
        raise ValueError("Truth or plot limits contain nonfinite values.")
    figure = plt.figure(figsize=FIGURE_SIZE, layout="constrained")
    layout = figure.add_gridspec(
        FINGER_COUNT, PANEL_COLUMNS, width_ratios=PANEL_WIDTH_RATIOS,
    )
    axes = []
    legend_axis = figure.add_subplot(layout[:, 1])
    legend_axis.axis("off")
    for finger in range(truth.shape[1]):
        axis = figure.add_subplot(
            layout[finger, 0], sharex=axes[0] if axes else None,
        )
        axes.append(axis)
        axis.plot(
            time_seconds, truth[:, finger], color="black",
            linewidth=TRUTH_WIDTH, label="Ground truth", zorder=10,
        )
        for model, values in predictions.items():
            color, _ = STYLES[model]
            axis.plot(
                time_seconds, values[:, finger], color=color,
                linewidth=MODEL_WIDTH, label=labels[model],
            )
        axis.set_ylim(*limits[finger])
        axis.set_ylabel(f"Finger {finger + 1}", fontsize=AXIS_SIZE)
        axis.tick_params(labelsize=TICK_SIZE)
        axis.grid(alpha=GRID_ALPHA)
        if finger < truth.shape[1] - 1:
            axis.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time within interval (s)", fontsize=AXIS_SIZE)
    figure.supylabel("Original glove value", fontsize=AXIS_SIZE)
    figure.suptitle(
        f"Subject {subject.upper()} — test interval {interval_number}",
        fontsize=TITLE_SIZE,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(
        handles, legend_labels, loc="center left", frameon=False,
        fontsize=LEGEND_SIZE,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
