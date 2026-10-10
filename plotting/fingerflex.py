"""Plot saved Stanford FingerFlex regression summaries.

Input: an artifact root containing ``reports/summary.json`` and
``provenance/resolved_config.json``; the summary holds each fraction's
per-subject, per-model seed count, overall metric mean and standard deviation,
and paired-test Holm p-values when two models are complete.
Output: ``reports/plots/subjects/<subject>/<metric>.png`` for each subject
and metric, plus ``reports/plots/aggregate.png``. No fit artifacts change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.regression import (
    draw_line_band,
    draw_significance_bracket,
)

METRICS = ("pearson_r", "mse", "r2")
METRIC_LABELS = {
    "pearson_r": "Mean finger Pearson r",
    "mse": "MSE (training-standardized target)",
    "r2": "Mean finger R²",
}
MODEL_LABELS = {
    "corteg": "CORTEG (ST-EEGFormer-small)",
    "hilo": "HiLoFuseNet",
}
MODEL_STYLES = {
    "corteg": ("#0072B2", "o"),
    "hilo": ("#D55E00", "s"),
}
FIGURE_DPI = 180
SUBJECT_SIZE = (6.4, 4.2)
AGGREGATE_SIZE = (17.0, 5.3)
AXIS_LEFT = 0.07
AXIS_RIGHT = 1.27


@dataclass(frozen=True)
class ResultTable:
    """Complete subject/model/metric/fraction score cube.

    Attributes
    ----------
    fractions : np.ndarray
        Training proportions, shape (n_fractions,).
    subjects, models, metrics : tuple[str, ...]
        Axis labels for the score cube.
    means, seed_stds : np.ndarray
        Values shaped (n_subjects, n_models, n_metrics, n_fractions).
    corrected_p : np.ndarray
        Holm-adjusted paired-model p-values, shape
        (n_metrics, n_fractions); NaN only when no paired test exists.
    """

    fractions: np.ndarray
    subjects: tuple[str, ...]
    models: tuple[str, ...]
    metrics: tuple[str, ...]
    means: np.ndarray
    seed_stds: np.ndarray
    corrected_p: np.ndarray


def _fraction_key(fraction: float) -> str:
    """Return the report key for a configured training proportion."""
    percent = fraction * 100
    if not percent.is_integer():
        raise ValueError(
            f"Training proportion has no integer-percent key: {fraction}."
        )
    return f"{int(percent)}pct"


def _read_json(path: Path) -> dict:
    """Load an existing JSON object with a useful path in any error."""
    if not path.is_file():
        raise FileNotFoundError(f"Required results file is missing: {path}.")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _validated_score(row: dict, metric: str, seeds: int) -> tuple[float, float]:
    """Validate one saved seed mean and standard deviation."""
    if row.get("seeds") != seeds:
        raise ValueError(
            f"Expected {seeds} completed seeds, got {row.get('seeds')}."
        )
    mean = row["mean"][metric]
    std = row["std"][metric]
    if std is None or not np.isfinite((mean, std)).all() or std < 0:
        raise ValueError(
            f"Invalid mean or standard deviation for {metric}: "
            f"{mean}, {std}."
        )
    return float(mean), float(std)


def load_results(root: Path) -> ResultTable:
    """Validate and arrange a complete saved campaign summary.

    Parameters
    ----------
    root : Path
        Artifact root with ``reports/`` and ``provenance/`` directories.

    Returns
    -------
    ResultTable
        Finite overall scores and available corrected p-values.
    """
    summary = _read_json(root / "reports" / "summary.json")
    provenance = root / "provenance"
    config = _read_json(provenance / "resolved_config.json")
    selections = provenance / "selections.json"
    if selections.is_file():
        config.update(_read_json(selections))
    fractions = np.array(sorted(config["fractions"]), dtype=float)
    subjects = tuple(config["subjects"])
    models = tuple(config.get("models", ("corteg", "hilo")))
    seeds = len(config["seeds"])
    if seeds < 2 or len(subjects) == 0 or len(models) == 0:
        raise ValueError(
            "Plots require at least two seeds, one subject, and one model."
        )
    if len(set(fractions)) != len(fractions) or (
        len(set(subjects)) != len(subjects)
        or len(set(models)) != len(models)
    ):
        raise ValueError("Plot selectors contain duplicate values.")
    shape = (len(subjects), len(models), len(METRICS), len(fractions))
    means = np.empty(shape, dtype=float)
    seed_stds = np.empty(shape, dtype=float)
    corrected_p = np.full((len(METRICS), len(fractions)), np.nan)
    for f_index, fraction in enumerate(fractions):
        key = _fraction_key(float(fraction))
        entry = summary["fractions"][key]
        for s_index, subject in enumerate(subjects):
            for m_index, model in enumerate(models):
                row = entry["subjects"][subject][model]
                for k_index, metric in enumerate(METRICS):
                    mean, std = _validated_score(row, metric, seeds)
                    means[s_index, m_index, k_index, f_index] = mean
                    seed_stds[s_index, m_index, k_index, f_index] = std
        if entry["paired_complete"]:
            for k_index, metric in enumerate(METRICS):
                value = entry["paired_test"][metric]["p_holm"]
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(
                        f"Invalid Holm p-value for {key}/{metric}: {value}."
                    )
                corrected_p[k_index, f_index] = value
    return ResultTable(
        fractions, subjects, models, METRICS,
        means, seed_stds, corrected_p,
    )


def _format_axis(axis: plt.Axes, fractions: np.ndarray, metric: str) -> None:
    """Give model curves consistent axis labels and proportions."""
    axis.set_xlabel("Training proportion")
    axis.set_ylabel(METRIC_LABELS[metric])
    axis.set_xticks(
        fractions, [f"{value:.0%}" for value in fractions],
    )
    axis.set_xlim(AXIS_LEFT, AXIS_RIGHT)
    axis.grid(alpha=0.2)


def _draw_models(
    axis: plt.Axes,
    fractions: np.ndarray,
    models: tuple[str, ...],
    means: np.ndarray,
    stds: np.ndarray,
    metric: str,
) -> None:
    """Draw all model curves from (model, fraction) arrays."""
    for index, model in enumerate(models):
        color, marker = MODEL_STYLES[model]
        draw_line_band(
            axis, fractions, means[index], stds[index],
            label=MODEL_LABELS[model], color=color, marker=marker,
            lower_bound=0.0 if metric == "mse" else None,
        )


def plot_subject_metric(
    table: ResultTable,
    subject_index: int,
    metric_index: int,
    output: Path,
) -> None:
    """Save one subject/metric plot using across-seed uncertainty."""
    subject = table.subjects[subject_index]
    metric = table.metrics[metric_index]
    figure, axis = plt.subplots(figsize=SUBJECT_SIZE)
    _draw_models(
        axis, table.fractions, table.models,
        table.means[subject_index, :, metric_index],
        table.seed_stds[subject_index, :, metric_index],
        metric,
    )
    _format_axis(axis, table.fractions, metric)
    axis.set_title(f"{subject.upper()} — {METRIC_LABELS[metric]}")
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_aggregate(table: ResultTable, output: Path) -> None:
    """Save equal-subject-weight curves and paired-test markers.

    The shaded area is sample standard deviation across subject seed means,
    not standard deviation over all individual seeds.
    """
    if len(table.subjects) < 2:
        raise ValueError("Aggregate SD needs at least two subjects.")
    means = table.means.mean(axis=0)
    between_subject_stds = table.means.std(axis=0, ddof=1)
    figure, axes = plt.subplots(1, len(METRICS), figsize=AGGREGATE_SIZE)
    for metric_index, axis in enumerate(axes):
        metric = METRICS[metric_index]
        _draw_models(
            axis, table.fractions, table.models,
            means[:, metric_index], between_subject_stds[:, metric_index],
            metric,
        )
        if metric == "mse":
            axis.set_yscale("symlog", linthresh=1.0)
            axis.set_ylim(bottom=0)
        if len(table.models) == 2:
            for fraction_index, fraction in enumerate(table.fractions):
                pvalue = table.corrected_p[metric_index, fraction_index]
                if np.isfinite(pvalue):
                    draw_significance_bracket(
                        axis, float(fraction),
                        float(means[0, metric_index, fraction_index]),
                        float(means[1, metric_index, fraction_index]),
                        float(pvalue),
                    )
        _format_axis(axis, table.fractions, metric)
        axis.set_title(METRIC_LABELS[metric])
    axes[0].legend(loc="best")
    figure.text(
        0.5, 0.03,
        f"Equal weight per subject (n={len(table.subjects)}). "
        "Band: between-subject SD of seed means. "
        "Brackets: paired subject sign-flip test, Holm corrected across "
        "the three metrics within each training proportion. "
        "MSE band clipped at zero; MSE axis uses a symmetric log scale.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def generate_plots(root: Path, output_dir: Path | None = None) -> list[Path]:
    """Generate all subject plots and one aggregate plot.

    Parameters
    ----------
    root : Path
        Existing campaign artifact root.
    output_dir : Path, optional
        Plot directory, default ``<root>/reports/plots``.

    Returns
    -------
    list[Path]
        Absolute PNG paths, one per subject/metric plus aggregate.
    """
    table = load_results(root)
    destination = (
        output_dir if output_dir is not None
        else root / "reports" / "plots"
    )
    paths = []
    for subject_index, subject in enumerate(table.subjects):
        for metric_index, metric in enumerate(table.metrics):
            output = destination / "subjects" / subject / f"{metric}.png"
            plot_subject_metric(table, subject_index, metric_index, output)
            paths.append(output.resolve())
    aggregate = destination / "aggregate.png"
    plot_aggregate(table, aggregate)
    paths.append(aggregate.resolve())
    return paths
