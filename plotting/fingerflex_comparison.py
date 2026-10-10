"""Plot four-model FingerFlex fits and six paired model comparisons.

Input is reports/summary.json in a complete comparison artifact. Output is
27 subject PNGs, reports/plots/aggregate.png, and paired_matrix.png.
Subject bands show seed SD; aggregate bands show SD among subject seed means.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plotting.regression import draw_line_band

METRICS = ("pearson_r", "mse", "r2")
MODELS = ("corteg", "hilo", "pls", "cnn_gru")
LABELS = {
    "corteg": "CORTEG (ST-EEGFormer-small)",
    "hilo": "HiLoFuseNet",
    "pls": "PLS (multiband Morlet)",
    "cnn_gru": "CNN-GRU (high gamma)",
}
STYLES = {
    "corteg": ("#0072B2", "o"),
    "hilo": ("#D55E00", "s"),
    "pls": ("#009E73", "^"),
    "cnn_gru": ("#CC79A7", "D"),
}
AXIS_LABELS = {
    "pearson_r": "Mean finger Pearson r",
    "mse": "MSE (training-standardized target)",
    "r2": "Mean finger R²",
}
DPI = 180
SIGNIFICANCE = 0.05


def _load(root: Path) -> tuple[dict, dict]:
    """Load and validate complete comparison data."""
    index = json.loads((root / "index.json").read_text())
    summary = json.loads(
        (root / "reports" / "summary.json").read_text()
    )
    if not summary.get("complete"):
        raise ValueError("Four-model plots require complete fits.")
    if set(index["fit_roots"]) != set(MODELS):
        raise ValueError("Comparison fit roots do not cover four models.")
    return index, summary


def _model_arrays(
    index: dict,
    summary: dict,
    subject: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return subject-level model means and stochastic seed SDs."""
    fractions = np.array(sorted(index["fractions"]), dtype=float)
    mean = np.empty((len(MODELS), len(fractions)))
    std = np.empty_like(mean)
    for model_index, model in enumerate(MODELS):
        for fraction_index, fraction in enumerate(fractions):
            key = f"{int(fraction * 100)}pct"
            row = summary["fractions"][key]["subjects"][subject][model]
            mean[model_index, fraction_index] = row["mean"][metric]
            value = row["std"][metric]
            std[model_index, fraction_index] = (
                np.nan if value is None else value
            )
    if not np.isfinite(mean).all():
        raise ValueError(f"Nonfinite subject means for {subject}/{metric}.")
    if not np.isfinite(std[:2]).all() or not np.isfinite(std[3]).all():
        raise ValueError(f"Missing stochastic seed SD for {subject}/{metric}.")
    return fractions, mean, std


def _draw_curves(
    axis: plt.Axes,
    fractions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    metric: str,
) -> None:
    """Draw reusable model lines with available SD bands."""
    for index, model in enumerate(MODELS):
        color, marker = STYLES[model]
        if np.isnan(std[index]).all():
            axis.plot(
                fractions, mean[index], color=color, marker=marker,
                label=LABELS[model],
            )
        else:
            if not np.isfinite(std[index]).all():
                raise ValueError(f"Partial SD band for {model}/{metric}.")
            draw_line_band(
                axis, fractions, mean[index], std[index],
                label=LABELS[model], color=color, marker=marker,
                lower_bound=0 if metric == "mse" else None,
            )
    axis.set_xlabel("Training proportion")
    axis.set_ylabel(AXIS_LABELS[metric])
    axis.set_xticks(
        fractions, [f"{value:.0%}" for value in fractions],
    )
    axis.grid(alpha=0.2)


def plot_subjects(root: Path, index: dict, summary: dict) -> None:
    """Save one line plot per subject and metric."""
    for subject, metric in itertools.product(
        index["subjects"], METRICS,
    ):
        fractions, mean, std = _model_arrays(
            index, summary, subject, metric,
        )
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        _draw_curves(axis, fractions, mean, std, metric)
        axis.set_title(f"{subject.upper()} — {AXIS_LABELS[metric]}")
        axis.legend(fontsize=8)
        figure.tight_layout()
        output = (
            root / "reports" / "plots" / "subjects"
            / subject / f"{metric}.png"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=DPI, bbox_inches="tight")
        plt.close(figure)


def plot_aggregate(root: Path, index: dict, summary: dict) -> None:
    """Save equal-subject-weight curves with between-subject SD bands."""
    figure, axes = plt.subplots(1, len(METRICS), figsize=(18, 5.5))
    for axis, metric in zip(axes, METRICS):
        subject_means = []
        for subject in index["subjects"]:
            fractions, mean, _ = _model_arrays(
                index, summary, subject, metric,
            )
            subject_means.append(mean)
        values = np.stack(subject_means)
        if len(values) < 2:
            raise ValueError("Aggregate plot needs at least two subjects.")
        _draw_curves(
            axis, fractions, values.mean(axis=0),
            values.std(axis=0, ddof=1), metric,
        )
        axis.set_title(AXIS_LABELS[metric])
        if metric == "mse":
            axis.set_yscale("symlog", linthresh=1.0)
            axis.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    figure.text(
        0.5, 0.02,
        "Equal weight per subject; bands are between-subject SD "
        "of subject seed means.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    output = root / "reports" / "plots" / "aggregate.png"
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def plot_pair_matrix(root: Path, index: dict, summary: dict) -> None:
    """Save six-pair effect and Holm p-value matrix for each metric."""
    pairs = list(itertools.combinations(MODELS, 2))
    fractions = sorted(index["fractions"])
    figure, axes = plt.subplots(
        1, len(METRICS), figsize=(17, 6.5),
        constrained_layout=True,
    )
    for axis, metric in zip(axes, METRICS):
        effects = np.empty((len(pairs), len(fractions)))
        pvalues = np.empty_like(effects)
        for row, (first, second) in enumerate(pairs):
            pair_key = f"{first}__{second}"
            for column, fraction in enumerate(fractions):
                key = f"{int(fraction * 100)}pct"
                result = summary["fractions"][key][
                    "comparisons"
                ][pair_key]
                if not result["complete"]:
                    raise ValueError(
                        f"Incomplete comparison {pair_key}/{key}."
                    )
                cell = result["metrics"][metric]
                effects[row, column] = cell["mean_difference"]
                pvalues[row, column] = cell["p_holm"]
        if not np.isfinite(effects).all() or not np.isfinite(pvalues).all():
            raise ValueError(f"Invalid paired matrix for {metric}.")
        limit = max(float(np.abs(effects).max()), 1e-8)
        axis.imshow(
            effects, cmap="coolwarm", vmin=-limit, vmax=limit,
            aspect="auto",
        )
        for row in range(len(pairs)):
            for column in range(len(fractions)):
                marker = "*" if pvalues[row, column] < SIGNIFICANCE else ""
                axis.text(
                    column, row,
                    f"Δ={effects[row, column]:+.3f}\n"
                    f"p={pvalues[row, column]:.3g}{marker}",
                    ha="center", va="center", fontsize=8,
                )
        axis.set_xticks(
            range(len(fractions)),
            [f"{fraction:.0%}" for fraction in fractions],
        )
        axis.set_yticks(
            range(len(pairs)),
            [f"{first} − {second}" for first, second in pairs],
        )
        axis.set_xlabel("Training proportion")
        axis.set_title(AXIS_LABELS[metric])
    figure.suptitle(
        "Paired subject mean differences (first minus second); "
        "exact sign-flip p, Holm across metrics within pair and fraction. "
        "* p < 0.05",
        fontsize=10,
    )
    output = root / "reports" / "plots" / "paired_matrix.png"
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def generate_plots(root: Path) -> None:
    """Generate subject, aggregate, and six-pair comparison plots."""
    index, summary = _load(root)
    plot_subjects(root, index, summary)
    plot_aggregate(root, index, summary)
    plot_pair_matrix(root, index, summary)
