"""Draw reusable regression curves and paired-comparison annotations.

Input: numeric x coordinates and matching one-dimensional mean and standard
deviation arrays, or two model means and a corrected p-value at one x value.
Output: Matplotlib artists added to the supplied Axes; no files are written.
"""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

SIGNIFICANCE_LEVEL = 0.05
BRACKET_X_OFFSET = 0.025
LABEL_X_OFFSET = 0.04


def draw_line_band(
    axis: Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str,
    lower_bound: float | None = None,
) -> None:
    """Draw a model mean and its standard-deviation band.

    Parameters
    ----------
    axis : Axes
        Matplotlib axis to update.
    x, mean, std : np.ndarray
        Matching one-dimensional arrays, shape (n_points,).
    label : str
        Legend label.
    color : str
        Matplotlib color specification.
    marker : str
        Matplotlib marker specification.
    lower_bound : float, optional
        Minimum visible value for the shaded band, default None.
    """
    if x.ndim != 1 or mean.shape != x.shape or std.shape != x.shape:
        raise ValueError(
            "Expected matching one-dimensional x, mean, and std arrays; "
            f"got {x.shape}, {mean.shape}, and {std.shape}."
        )
    if not all(np.isfinite(values).all() for values in (x, mean, std)):
        raise ValueError("Curve coordinates and uncertainty must be finite.")
    if (std < 0).any():
        raise ValueError("Curve standard deviation cannot be negative.")
    if lower_bound is not None and not np.isfinite(lower_bound):
        raise ValueError("Curve lower bound must be finite.")
    lower = mean - std
    if lower_bound is not None:
        lower = np.maximum(lower, lower_bound)
    axis.plot(x, mean, label=label, color=color, marker=marker)
    axis.fill_between(
        x, lower, mean + std, color=color, alpha=0.17,
    )


def draw_significance_bracket(
    axis: Axes,
    x: float,
    first: float,
    second: float,
    corrected_p: float,
) -> None:
    """Annotate a significant paired comparison at one x coordinate.

    Parameters
    ----------
    axis : Axes
        Matplotlib axis to update.
    x : float
        Shared training proportion of both model points.
    first, second : float
        The two model means at that proportion.
    corrected_p : float
        Holm-adjusted p-value for the plotted metric and proportion.
    """
    values = np.array((x, first, second, corrected_p), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Bracket coordinates and p-value must be finite.")
    if not 0 <= corrected_p <= 1:
        raise ValueError(
            f"Corrected p-value must lie in [0, 1], got {corrected_p}."
        )
    if corrected_p >= SIGNIFICANCE_LEVEL:
        return
    bracket_x = x + BRACKET_X_OFFSET
    axis.plot(
        (x, bracket_x, bracket_x, x),
        (first, first, second, second),
        color="black", linewidth=0.8,
    )
    axis.annotate(
        f"Holm p={corrected_p:.3g}",
        (x + LABEL_X_OFFSET, (first + second) / 2),
        fontsize=8, va="center",
        xytext=(4, 8), textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
              "edgecolor": "none", "alpha": 0.85},
    )
