"""Load CORTEG FingerFlex histories and build aggregate monitor figures.

Input: a model fit root with provenance selectors and runs/<fraction>/corteg/
subject/seed*/history.jsonl. Output: one aggregate PNG per available monitor
under reports/plots/training_monitors/.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from plotting.training_monitors import (
    MonitorCurve,
    MonitorPanel,
    MonitorPlotStyle,
    aggregate_subject_runs,
    plot_monitor_panels,
)

FINGER_LABELS = tuple(f"Finger {index}" for index in range(1, 6))
FINGER_COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
)
GROUP_COLORS = {"base": "#0072B2", "adapter": "#D55E00"}
TRAIN_COLOR = "#0072B2"
VALIDATION_COLOR = "#D55E00"
TOTAL_GRADIENT_COLOR = "#009E73"
DEFAULT_STYLE = MonitorPlotStyle()
NONNEGATIVE_MONITORS = ("overall_mse", "finger_mse")

SeriesMap = dict[str, dict[float, dict[str, dict[str, list[np.ndarray]]]]]
MonitorMetadata = tuple[str, str, str]


def _load_object(path: Path) -> dict:
    """Read a required JSON object with a path-specific error."""
    if not path.is_file():
        raise FileNotFoundError(f"Required campaign file is missing: {path}.")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _selectors(root: Path) -> tuple[list[str], list[float], list[int]]:
    """Resolve current subject, fraction, and seed selectors."""
    config = _load_object(root / "provenance" / "resolved_config.json")
    selection_path = root / "provenance" / "selections.json"
    if selection_path.is_file():
        config.update(_load_object(selection_path))
    subjects = config.get("subjects", [])
    fractions = config.get("fractions", [])
    seeds = config.get("seeds", [])
    if not subjects or not fractions or not seeds:
        raise ValueError(
            "Campaign selectors must include subjects, fractions, and seeds."
        )
    if any(len(values) != len(set(values)) for values in (
        subjects, fractions, seeds,
    )):
        raise ValueError("Campaign selectors contain duplicate values.")
    return list(subjects), list(fractions), list(seeds)


def _read_history(path: Path) -> list[dict]:
    """Read one validated, sequential epoch history."""
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON monitor row in {path}: {error}."
        ) from error
    if not rows:
        raise ValueError(f"Training history is empty: {path}.")
    epochs = [row.get("epoch") for row in rows]
    expected = list(range(1, len(rows) + 1))
    if epochs != expected:
        raise ValueError(
            f"Expected sequential epochs 1..{len(rows)} in {path}; "
            f"got {epochs[:4]}...{epochs[-4:]}."
        )
    return rows


def _finite_series(values: object, name: str, path: Path) -> np.ndarray:
    """Convert one scalar or finger monitor into a finite 1D array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"Expected a nonempty 1D {name} series in {path}; "
            f"got shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite values in {name} at {path}.")
    return array


def _record_series(row: dict, path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Extract available scalar series from one epoch record."""
    extracted: dict[str, dict[str, np.ndarray]] = {}
    validation = row.get("validation", {})
    overall = validation.get("overall", {})
    fingers = validation.get("fingers", {})

    mse_curves = {}
    if "train_mse" in row:
        mse_curves["Train MSE"] = _finite_series(
            [row["train_mse"]], "train_mse", path,
        )
    if "mse" in overall:
        mse_curves["Validation MSE"] = _finite_series(
            [overall["mse"]], "validation.mse", path,
        )
    if mse_curves:
        extracted["overall_mse"] = mse_curves

    for metric in ("pearson_r", "r2"):
        if metric in overall:
            extracted[f"validation_{metric}"] = {
                "Overall": _finite_series(
                    [overall[metric]], f"validation.{metric}", path,
                ),
            }

    for metric in ("pearson_r", "mse", "r2"):
        if metric in fingers:
            values = _finite_series(
                fingers[metric], f"validation.fingers.{metric}", path,
            )
            if len(values) != len(FINGER_LABELS):
                raise ValueError(
                    f"Expected {len(FINGER_LABELS)} finger values for "
                    f"{metric} at {path}, got {len(values)}."
                )
            extracted[f"finger_{metric}"] = {
                label: np.asarray([value], dtype=float)
                for label, value in zip(FINGER_LABELS, values)
            }

    for source_name, monitor_name, ylabel in (
        ("learning_rates", "learning_rates", "Learning rate"),
        ("group_gradient_norm_mean", "group_gradient_norm", "Gradient norm"),
    ):
        values = row.get(source_name)
        if isinstance(values, dict) and values:
            extracted[monitor_name] = {
                group.title(): _finite_series(
                    [value], f"{source_name}.{group}", path,
                ) for group, value in values.items()
            }

    if "gradient_norm_mean" in row:
        extracted["gradient_norm"] = {
            "All trainable parameters": _finite_series(
                [row["gradient_norm_mean"]], "gradient_norm_mean", path,
            ),
        }

    parameter_groups = row.get("parameter_groups")
    if isinstance(parameter_groups, dict):
        for metric in ("norm", "change"):
            values = {
                group.title(): _finite_series(
                    [entry[metric]], f"parameter_groups.{group}.{metric}",
                    path,
                )
                for group, entry in parameter_groups.items()
                if metric in entry
            }
            if values:
                extracted[f"parameter_group_{metric}"] = values
    return extracted


def _append_series(
    destination: SeriesMap,
    monitor: str,
    fraction: float,
    curve: str,
    subject: str,
    value: np.ndarray,
) -> None:
    """Append an epoch value to one monitor/fraction/subject group."""
    by_curve = destination.setdefault(monitor, {}).setdefault(
        fraction, {},
    ).setdefault(curve, {})
    by_curve.setdefault(subject, []).append(value)


def _collect_histories(
    root: Path,
    subjects: list[str],
    fractions: list[float],
    seeds: list[int],
) -> tuple[SeriesMap, int, list[str]]:
    """Load completed or partial histories and group available monitors."""
    collected: SeriesMap = {}
    count = 0
    missing = []
    for fraction in fractions:
        percent = fraction * 100
        if not float(percent).is_integer():
            raise ValueError(
                f"Fraction has no integer percent run key: {fraction}."
            )
        for subject in subjects:
            for seed in seeds:
                path = (
                    root / "runs" / f"{int(percent)}pct" / "corteg"
                    / subject / f"seed{seed}" / "history.jsonl"
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                rows = _read_history(path)
                series_by_monitor: dict[str, dict[str, list[float]]] = {}
                for row in rows:
                    extracted = _record_series(row, path)
                    for monitor, curves in extracted.items():
                        for curve, value in curves.items():
                            series_by_monitor.setdefault(
                                monitor, {},
                            ).setdefault(curve, []).append(float(value[0]))
                for monitor, curves in series_by_monitor.items():
                    for curve, values in curves.items():
                        _append_series(
                            collected, monitor, fraction, curve, subject,
                            np.asarray(values, dtype=float),
                        )
                count += 1
    if count == 0:
        raise ValueError(f"No training histories found under {root / 'runs'}.")
    return collected, count, missing


def _monitor_metadata() -> dict[str, MonitorMetadata]:
    """Return plot title, y-axis label, and output filename by monitor."""
    return {
        "overall_mse": (
            "Training and validation MSE",
            "MSE (training-standardized target)", "overall_mse.png",
        ),
        "validation_pearson_r": (
            "Overall validation Pearson r", "Mean finger Pearson r",
            "validation_pearson_r.png",
        ),
        "validation_r2": (
            "Overall validation R²", "Mean finger R²",
            "validation_r2.png",
        ),
        "finger_pearson_r": (
            "Per-finger validation Pearson r", "Pearson r",
            "finger_pearson_r.png",
        ),
        "finger_mse": (
            "Per-finger validation MSE", "MSE", "finger_mse.png",
        ),
        "finger_r2": (
            "Per-finger validation R²", "R²", "finger_r2.png",
        ),
        "learning_rates": (
            "Learning rates by parameter group", "Learning rate",
            "learning_rates.png",
        ),
        "gradient_norm": (
            "Total trainable gradient norm", "Gradient norm",
            "gradient_norm.png",
        ),
        "group_gradient_norm": (
            "Gradient norm by parameter group", "Gradient norm",
            "group_gradient_norm.png",
        ),
        "parameter_group_norm": (
            "Trainable parameter-group norm", "L2 norm",
            "parameter_group_norm.png",
        ),
        "parameter_group_change": (
            "Parameter-group change from initialization",
            "L2 change", "parameter_group_change.png",
        ),
    }


def _curve_style(monitor: str, label: str, index: int) -> tuple[str, str]:
    """Choose stable colors and line styles for a monitor curve."""
    if monitor == "overall_mse":
        if label == "Train MSE":
            return TRAIN_COLOR, "-"
        return VALIDATION_COLOR, "--"
    if monitor == "gradient_norm":
        return TOTAL_GRADIENT_COLOR, "-"
    key = label.lower()
    if key in GROUP_COLORS:
        return GROUP_COLORS[key], "-"
    return FINGER_COLORS[index % len(FINGER_COLORS)], "-"


def _panels_for_monitor(
    monitor: str,
    data: dict[float, dict[str, dict[str, list[np.ndarray]]]],
    fractions: list[float],
) -> tuple[MonitorPanel, ...]:
    """Aggregate all curves and construct one panel per fraction."""
    panels = []
    expected_curves = None
    for fraction in fractions:
        fraction_data = data.get(fraction, {})
        if not fraction_data:
            raise ValueError(
                f"Monitor {monitor} has no curves at {fraction:.0%}."
            )
        labels = tuple(sorted(fraction_data))
        if expected_curves is None:
            expected_curves = labels
        elif labels != expected_curves:
            raise ValueError(
                f"Monitor {monitor} has inconsistent curves across fractions."
            )
        curves = []
        for index, label in enumerate(labels):
            summary = aggregate_subject_runs(fraction_data[label])
            color, linestyle = _curve_style(monitor, label, index)
            curves.append(MonitorCurve(
                label, color, linestyle, summary,
            ))
        panels.append(MonitorPanel(f"{fraction:.0%} training", tuple(curves)))
    return tuple(panels)


def generate_training_monitor_plots(
    root: Path,
    output_dir: Path | None = None,
    style: MonitorPlotStyle = DEFAULT_STYLE,
) -> list[Path]:
    """Generate aggregate training-monitor figures from one fit root.

    Parameters
    ----------
    root : Path
        CORTEG fit root containing selectors and structured run histories.
    output_dir : Path, optional
        Figure directory, default ``reports/plots/training_monitors``.
    style : MonitorPlotStyle, optional
        Reusable Matplotlib style settings, default ``DEFAULT_STYLE``.

    Returns
    -------
    list[Path]
        Generated figure paths, one per monitor available in the histories.
    """
    root = root.expanduser().resolve()
    subjects, fractions, seeds = _selectors(root)
    collected, history_count, missing = _collect_histories(
        root, subjects, fractions, seeds,
    )
    output = output_dir or (
        root / "reports" / "plots" / "training_monitors"
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata = _monitor_metadata()
    paths = []
    for monitor, fraction_data in collected.items():
        if monitor not in metadata:
            raise ValueError(f"No plotting metadata for monitor {monitor}.")
        title, ylabel, filename = metadata[monitor]
        panels = _panels_for_monitor(
            monitor, fraction_data, fractions,
        )
        nonnegative = monitor in NONNEGATIVE_MONITORS
        paths.append(plot_monitor_panels(
            title, ylabel, panels, output / filename, style,
            yscale="symlog" if nonnegative else "linear",
            lower_bound=0.0 if nonnegative else None,
        ))
    if missing:
        print(
            f"Skipped {len(missing)} missing histories; loaded "
            f"{history_count} histories.", file=sys.stderr,
        )
    return paths
