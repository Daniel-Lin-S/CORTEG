"""Plot completed FingerFlex model fits and paired comparisons.

Input is reports/summary.json in a complete comparison artifact. Output is
plots/subjects/<subject>/<metric>.png, plots/aggregate.png, and
plots/paired_matrix.png. Subject bands show seed SD; aggregate bands show
SD among subject seed means. The model list comes from index.json.
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
SUPPORTED_MODELS = ("corteg", "hilo", "pls", "cnn_gru")
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
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 12
CELL_FONT_SIZE = 10
TEST_LABELS = {
    "exact_sign_flip": "exact sign-flip",
    "wilcoxon_signed_rank": "two-sided Wilcoxon",
}


def _models(index: dict) -> tuple[str, ...]:
    """Return the ordered supported model list from an artifact index."""
    models = tuple(index["fit_roots"])
    if len(models) < 2 or set(models) - set(SUPPORTED_MODELS):
        raise ValueError(
            f"Expected at least two supported comparison models: {models}."
        )
    return models


def _load(root: Path) -> tuple[dict, dict]:
    """Load and validate complete comparison data."""
    index = json.loads((root / "index.json").read_text())
    summary = json.loads(
        (root / "reports" / "summary.json").read_text()
    )
    if not summary.get("complete"):
        raise ValueError("Comparison plots require complete fits.")
    models = _models(index)
    if "models" in summary and tuple(summary["models"]) != models:
        raise ValueError("Report models differ from comparison fit roots.")
    return index, summary


def _model_arrays(
    index: dict,
    summary: dict,
    subject: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return subject-level model means and stochastic seed SDs."""
    fractions = np.array(sorted(index["fractions"]), dtype=float)
    models = _models(index)
    mean = np.empty((len(models), len(fractions)))
    std = np.empty_like(mean)
    for model_index, model in enumerate(models):
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
    for model_index, model in enumerate(models):
        if model == "pls":
            if not np.isnan(std[model_index]).all():
                raise ValueError("PLS seed SD must be inapplicable.")
        elif not np.isfinite(std[model_index]).all():
            raise ValueError(
                f"Missing stochastic seed SD for {subject}/{metric}."
            )
    return fractions, mean, std


def _draw_curves(
    axis: plt.Axes,
    fractions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    metric: str,
    models: tuple[str, ...],
) -> None:
    """Draw reusable model lines with available SD bands."""
    for index, model in enumerate(models):
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
    axis.set_xlabel("Training proportion", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel(AXIS_LABELS[metric], fontsize=AXIS_FONT_SIZE)
    axis.tick_params(labelsize=TICK_FONT_SIZE)
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
        figure, axis = plt.subplots(figsize=(8.5, 5.0))
        _draw_curves(
            axis, fractions, mean, std, metric, _models(index),
        )
        axis.set_title(
            f"{subject.upper()} — {AXIS_LABELS[metric]}",
            fontsize=TITLE_FONT_SIZE,
        )
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="center left",
            bbox_to_anchor=(0.74, 0.5), fontsize=LEGEND_FONT_SIZE,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 0.74, 1))
        output = root / "plots" / "subjects" / subject / f"{metric}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=DPI, bbox_inches="tight")
        plt.close(figure)


def plot_aggregate(root: Path, index: dict, summary: dict) -> None:
    """Save equal-subject-weight curves with between-subject SD bands."""
    figure, axes = plt.subplots(1, len(METRICS), figsize=(20, 6.0))
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
            values.std(axis=0, ddof=1), metric, _models(index),
        )
        axis.set_title(AXIS_LABELS[metric], fontsize=TITLE_FONT_SIZE)
        if metric == "mse":
            axis.set_yscale("symlog", linthresh=1.0)
            axis.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="center left",
        bbox_to_anchor=(0.79, 0.5), fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    figure.text(
        0.4, 0.02,
        "Equal weight per subject; bands show SD across subject seed means.",
        ha="center", fontsize=TICK_FONT_SIZE,
    )
    figure.tight_layout(rect=(0, 0.09, 0.79, 1))
    output = root / "plots" / "aggregate.png"
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def plot_pair_matrix(root: Path, index: dict, summary: dict) -> None:
    """Save paired effect and Holm p-value matrices for each metric."""
    pairs = list(itertools.combinations(_models(index), 2))
    fractions = sorted(index["fractions"])
    figure, axes = plt.subplots(
        1, len(METRICS), figsize=(20, 6.5),
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
                    ha="center", va="center", fontsize=CELL_FONT_SIZE,
                )
        axis.set_xticks(
            range(len(fractions)),
            [f"{fraction:.0%}" for fraction in fractions],
        )
        axis.set_yticks(
            range(len(pairs)),
            [f"{first} − {second}" for first, second in pairs],
        )
        axis.set_xlabel(
            "Training proportion", fontsize=AXIS_FONT_SIZE,
        )
        axis.set_title(AXIS_LABELS[metric], fontsize=TITLE_FONT_SIZE)
        axis.tick_params(labelsize=TICK_FONT_SIZE)
    test = index.get("semantics", {}).get(
        "rules", {},
    ).get("test", "wilcoxon_signed_rank")
    if test not in TEST_LABELS:
        raise ValueError(f"Unknown paired comparison test: {test}.")
    figure.suptitle(
        "Paired subject mean differences (first minus second); "
        f"{TEST_LABELS[test]} p, Holm across metrics within pair and "
        "fraction. * p < 0.05",
        fontsize=TICK_FONT_SIZE,
    )
    output = root / "plots" / "paired_matrix.png"
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def generate_plots(root: Path) -> None:
    """Generate subject, aggregate, and paired comparison plots."""
    index, summary = _load(root)
    plot_subjects(root, index, summary)
    plot_aggregate(root, index, summary)
    plot_pair_matrix(root, index, summary)
