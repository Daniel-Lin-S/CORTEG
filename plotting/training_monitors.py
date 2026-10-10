"""Aggregate and render epoch-based training monitor series.

Input: named, one-dimensional epoch series grouped by subject and repeat.
Output: Matplotlib figures with subject-mean curves and subject SD bands.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class EpochSummary:
    """Across-subject statistics for one curve at each epoch.

    Attributes
    ----------
    epoch : np.ndarray
        Epoch indices, shape (n_epochs,).
    mean, std : np.ndarray
        Across-subject mean and sample standard deviation, each with shape
        (n_epochs,). Standard deviation is NaN when fewer than two subjects
        contribute.
    n_subjects : np.ndarray
        Number of contributing subjects at each epoch, shape (n_epochs,).
    """

    epoch: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_subjects: np.ndarray


@dataclass(frozen=True)
class MonitorCurve:
    """One labeled monitor curve for a fraction panel."""

    label: str
    color: str
    linestyle: str
    summary: EpochSummary


@dataclass(frozen=True)
class MonitorPanel:
    """Curves and title for one training-fraction panel."""

    title: str
    curves: tuple[MonitorCurve, ...]


@dataclass(frozen=True)
class MonitorPlotStyle:
    """Adjustable figure dimensions, fonts, lines, and uncertainty bands."""

    panel_width: float = 4.2
    panel_height: float = 3.8
    legend_width: float = 2.4
    dpi: int = 180
    title_fontsize: float = 14
    panel_fontsize: float = 11
    label_fontsize: float = 10
    tick_fontsize: float = 9
    legend_fontsize: float = 9
    line_width: float = 1.8
    band_alpha: float = 0.18


def aggregate_subject_runs(
    subject_runs: Mapping[str, Sequence[np.ndarray]],
) -> EpochSummary:
    """Average repeats within subject, then summarize across subjects.

    Parameters
    ----------
    subject_runs : Mapping[str, Sequence[np.ndarray]]
        Each subject maps to one or more finite epoch series. Series may have
        different lengths due to early stopping and have shape (epochs,).

    Returns
    -------
    EpochSummary
        Epoch-wise mean, sample SD, and contributing-subject count.
    """
    if not subject_runs:
        raise ValueError("Cannot aggregate an empty subject mapping.")
    if any(not runs for runs in subject_runs.values()):
        raise ValueError("Every subject must have at least one run series.")

    max_epochs = max(
        len(series)
        for runs in subject_runs.values()
        for series in runs
    )
    if max_epochs < 1:
        raise ValueError("Monitor series must contain at least one epoch.")

    subject_values = np.full(
        (len(subject_runs), max_epochs), np.nan, dtype=float,
    )
    for subject_index, (subject, runs) in enumerate(subject_runs.items()):
        repeated = []
        for series in runs:
            values = np.asarray(series, dtype=float)
            if values.ndim != 1 or values.size == 0:
                raise ValueError(
                    f"Expected nonempty 1D monitor series for {subject}; "
                    f"got shape {values.shape}."
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Monitor series contains nonfinite values for {subject}."
                )
            repeated.append(values)
        for epoch_index in range(max_epochs):
            available = [
                run[epoch_index] for run in repeated
                if epoch_index < len(run)
            ]
            if available:
                subject_values[subject_index, epoch_index] = np.mean(
                    available,
                )

    counts = np.isfinite(subject_values).sum(axis=0)
    means = np.full(max_epochs, np.nan, dtype=float)
    stds = np.full(max_epochs, np.nan, dtype=float)
    for epoch_index in range(max_epochs):
        values = subject_values[:, epoch_index]
        values = values[np.isfinite(values)]
        if values.size:
            means[epoch_index] = values.mean()
        if values.size > 1:
            stds[epoch_index] = values.std(ddof=1)
    if not np.isfinite(means).all():
        raise ValueError("Aggregated monitor means contain nonfinite values.")

    return EpochSummary(
        epoch=np.arange(1, max_epochs + 1, dtype=int),
        mean=means,
        std=stds,
        n_subjects=counts,
    )


def _draw_curve(
    axis: plt.Axes,
    curve: MonitorCurve,
    style: MonitorPlotStyle,
    lower_bound: float | None,
) -> None:
    """Draw an epoch mean curve and its available subject SD band."""
    summary = curve.summary
    line, = axis.plot(
        summary.epoch, summary.mean, color=curve.color,
        linestyle=curve.linestyle, linewidth=style.line_width,
        label=curve.label,
    )
    mask = np.isfinite(summary.std) & (summary.n_subjects > 1)
    if mask.any():
        lower = summary.mean - summary.std
        if lower_bound is not None:
            lower = np.maximum(lower, lower_bound)
        axis.fill_between(
            summary.epoch, lower, summary.mean + summary.std,
            where=mask, color=line.get_color(),
            alpha=style.band_alpha, linewidth=0,
        )


def plot_monitor_panels(
    title: str,
    y_label: str,
    panels: Sequence[MonitorPanel],
    output: Path,
    style: MonitorPlotStyle = MonitorPlotStyle(),
    *,
    yscale: str = "linear",
    lower_bound: float | None = None,
) -> Path:
    """Render fraction panels with a shared legend outside the axes.

    Parameters
    ----------
    title : str
        Figure title for this monitor.
    y_label : str
        Shared vertical-axis label.
    panels : Sequence[MonitorPanel]
        Curves for each fraction or other grouping.
    output : Path
        Destination PNG path.
    style : MonitorPlotStyle, optional
        Figure styling, default ``MonitorPlotStyle()``.

    Returns
    -------
    Path
        Saved figure path.
    """
    if yscale not in ("linear", "log", "symlog"):
        raise ValueError(f"Unsupported monitor y scale: {yscale}.")
    if lower_bound is not None and not np.isfinite(lower_bound):
        raise ValueError("Monitor lower bound must be finite.")
    if not panels:
        raise ValueError("A monitor figure needs at least one panel.")
    if any(not panel.curves for panel in panels):
        raise ValueError("Every monitor panel needs at least one curve.")
    expected_labels = tuple(curve.label for curve in panels[0].curves)
    if any(
        tuple(curve.label for curve in panel.curves) != expected_labels
        for panel in panels[1:]
    ):
        raise ValueError("All panels must use the same curve labels.")

    figure_width = style.panel_width * len(panels) + style.legend_width
    figure, axes = plt.subplots(
        1, len(panels), figsize=(figure_width, style.panel_height),
        sharey=True, squeeze=False,
    )
    flat_axes = axes[0]
    flat_axes[0].set_yscale(yscale)
    handles = []
    labels = []
    for panel_index, (axis, panel) in enumerate(zip(flat_axes, panels)):
        for curve in panel.curves:
            _draw_curve(axis, curve, style, lower_bound)
        axis.set_title(panel.title, fontsize=style.panel_fontsize)
        axis.set_xlabel("Epoch", fontsize=style.label_fontsize)
        if panel_index == 0:
            axis.set_ylabel(y_label, fontsize=style.label_fontsize)
        axis.tick_params(labelsize=style.tick_fontsize)
        axis.grid(True, alpha=0.22, linewidth=0.7)
        contributing = [
            curve.summary.n_subjects
            for curve in panel.curves
        ]
        minimum = min(int(values.min()) for values in contributing)
        maximum = max(int(values.max()) for values in contributing)
        axis.text(
            0.98, 0.98, f"subjects per epoch: {minimum}–{maximum}",
            transform=axis.transAxes, ha="right", va="top",
            fontsize=style.tick_fontsize - 1,
        )
        if panel_index == 0:
            handles, labels = axis.get_legend_handles_labels()

    if lower_bound is not None:
        flat_axes[0].set_ylim(bottom=lower_bound)
    figure.suptitle(title, fontsize=style.title_fontsize)
    figure.legend(
        handles, labels, loc="center left",
        bbox_to_anchor=(
            len(panels) * style.panel_width / figure_width, 0.5,
        ), frameon=False, fontsize=style.legend_fontsize,
    )
    figure.tight_layout(
        rect=(0, 0, len(panels) * style.panel_width / figure_width, 0.93),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=style.dpi, bbox_inches="tight")
    plt.close(figure)
    return output
