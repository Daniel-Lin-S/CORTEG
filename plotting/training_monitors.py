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
    panel_height: float = 4.6
    legend_width: float = 2.4
    dpi: int = 180
    title_fontsize: float = 18
    panel_fontsize: float = 18
    subtitle_fontsize: float = 14
    label_fontsize: float = 16
    tick_fontsize: float = 13
    legend_fontsize: float = 13
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
    y_limits: tuple[float, float] | None,
) -> None:
    """Draw an epoch mean curve, its SD band, and range-crossing arrows."""
    summary = curve.summary
    line, = axis.plot(
        summary.epoch, summary.mean, color=curve.color,
        linestyle=curve.linestyle, linewidth=style.line_width,
        label=curve.label,
    )
    mask = np.isfinite(summary.std) & (summary.n_subjects > 1)
    if mask.any():
        lower = summary.mean - summary.std
        upper = summary.mean + summary.std
        if y_limits is not None:
            lower = np.clip(lower, *y_limits)
            upper = np.clip(upper, *y_limits)
        axis.fill_between(
            summary.epoch, lower, upper,
            where=mask, color=line.get_color(),
            alpha=style.band_alpha, linewidth=0,
        )
    if y_limits is not None:
        draw_offscale_arrows(
            axis, summary.epoch, summary.mean, y_limits,
            line.get_color(),
        )


def _curve_range_values(curves: Sequence[MonitorCurve]) -> np.ndarray:
    """Return finite means and SD-band edges for y-limit calculation."""
    values = []
    for curve in curves:
        mean = curve.summary.mean
        std = curve.summary.std
        values.extend((mean, mean - std, mean + std))
    finite = [array[np.isfinite(array)] for array in values]
    finite = [array for array in finite if array.size]
    if not finite:
        raise ValueError("Monitor curves contain no finite values.")
    return np.concatenate(finite)


def robust_y_limits(
    panels: Sequence[MonitorPanel],
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975,
) -> tuple[float, float]:
    """Calculate shared limits from central quantiles of curves and bands.

    Parameters
    ----------
    panels : Sequence[MonitorPanel]
        Panels whose curves share one vertical scale.
    lower_quantile : float, optional
        Lower quantile, default 0.025.
    upper_quantile : float, optional
        Upper quantile, default 0.975.

    Returns
    -------
    tuple[float, float]
        Finite lower and upper bounds spanning the requested central range.
    """
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "Expected quantiles satisfying 0 <= lower < upper <= 1."
        )
    values = _curve_range_values([
        curve for panel in panels for curve in panel.curves
    ])
    lower, upper = np.quantile(
        values, (lower_quantile, upper_quantile),
    )
    if not np.isfinite((lower, upper)).all() or lower >= upper:
        raise ValueError(
            f"Expected distinct finite plot limits; got ({lower}, {upper})."
        )
    return float(lower), float(upper)


def draw_offscale_arrows(
    axis: plt.Axes,
    epochs: np.ndarray,
    values: np.ndarray,
    y_limits: tuple[float, float],
    color: str,
) -> None:
    """Mark each mean-line crossing of a clipped y boundary with an arrow.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes receiving arrows at the visible plot boundary.
    epochs : np.ndarray
        Epoch indices, shape (n_epochs,).
    values : np.ndarray
        Mean values, shape (n_epochs,).
    y_limits : tuple[float, float]
        Visible lower and upper y limits.
    color : str
        Mean-line color used for the arrow.
    """
    lower, upper = y_limits
    if epochs.ndim != 1 or values.ndim != 1 or epochs.shape != values.shape:
        raise ValueError(
            "Expected matching one-dimensional epoch and monitor arrays; "
            f"got shapes {epochs.shape} and {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Cannot place arrows for nonfinite curve values.")

    crossings: list[tuple[float, float, int]] = []
    for index in range(len(values) - 1):
        x0, x1 = float(epochs[index]), float(epochs[index + 1])
        y0, y1 = float(values[index]), float(values[index + 1])
        for bound, direction in ((lower, -1), (upper, 1)):
            outside0 = y0 < bound if direction < 0 else y0 > bound
            outside1 = y1 < bound if direction < 0 else y1 > bound
            if outside0 != outside1 and y0 != y1:
                fraction = (bound - y0) / (y1 - y0)
                crossings.append((x0 + fraction * (x1 - x0), bound,
                                  direction))

    for index, value in ((0, values[0]), (-1, values[-1])):
        if value < lower:
            crossings.append((float(epochs[index]), lower, -1))
        elif value > upper:
            crossings.append((float(epochs[index]), upper, 1))

    for epoch, bound, direction in crossings:
        offset = 9 * direction
        axis.annotate(
            "", xy=(epoch, bound), xytext=(0, -offset),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-|>", "color": color,
                        "linewidth": 1.3, "mutation_scale": 10},
            annotation_clip=False,
        )


def plot_monitor_panels(
    title: str,
    y_label: str,
    panels: Sequence[MonitorPanel],
    output: Path,
    style: MonitorPlotStyle = MonitorPlotStyle(),
    *,
    subtitle: str,
    x_label: str = "Epoch (1-based)",
    y_limits: tuple[float, float] | None = None,
) -> Path:
    """Render fraction panels and a shared legend when curves are multiple.

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
    subtitle : str
        Figure subtitle, typically the number of contributing subjects.
    x_label : str, optional
        Shared epoch-axis label, default ``Epoch (1-based)``.
    y_limits : tuple[float, float], optional
        Shared lower and upper y limits, default data-driven limits.

    Returns
    -------
    Path
        Saved figure path.
    """
    if not subtitle.strip():
        raise ValueError("Figure subtitle must not be empty.")
    if y_limits is not None and (
        len(y_limits) != 2
        or not np.isfinite(y_limits).all()
        or y_limits[0] >= y_limits[1]
    ):
        raise ValueError(f"Invalid shared y limits: {y_limits}.")
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

    has_legend = len(expected_labels) > 1
    figure_width = style.panel_width * len(panels)
    if has_legend:
        figure_width += style.legend_width
    figure, axes = plt.subplots(
        1, len(panels), figsize=(figure_width, style.panel_height),
        sharey=True, squeeze=False,
    )
    flat_axes = axes[0]
    handles = []
    labels = []
    for panel_index, (axis, panel) in enumerate(zip(flat_axes, panels)):
        for curve in panel.curves:
            _draw_curve(axis, curve, style, y_limits)
        axis.set_title(panel.title, fontsize=style.panel_fontsize)
        axis.set_xlabel(x_label, fontsize=style.label_fontsize)
        if panel_index == 0:
            axis.set_ylabel(y_label, fontsize=style.label_fontsize)
        axis.tick_params(labelsize=style.tick_fontsize)
        axis.grid(True, alpha=0.22, linewidth=0.7)
        if panel_index == 0:
            handles, labels = axis.get_legend_handles_labels()

    if y_limits is not None:
        flat_axes[0].set_ylim(y_limits)
    figure.suptitle(title, fontsize=style.title_fontsize, y=0.99)
    figure.text(
        0.5, 0.94, subtitle, ha="center", va="top",
        fontsize=style.subtitle_fontsize,
    )
    if has_legend:
        figure.legend(
            handles, labels, loc="center left",
            bbox_to_anchor=(
                len(panels) * style.panel_width / figure_width, 0.5,
            ), frameon=False, fontsize=style.legend_fontsize,
        )
    plot_width = len(panels) * style.panel_width / figure_width
    figure.tight_layout(
        rect=(0, 0, plot_width, 0.88),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=style.dpi, bbox_inches="tight")
    plt.close(figure)
    return output
