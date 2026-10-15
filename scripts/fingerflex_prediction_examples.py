"""Generate matched FingerFlex prediction examples from completed fits.

Input: a portable comparison JSON and its comparison roots,
each with ``index.json``, completed ``reports/summary.json``, and saved
model fit predictions. Output: eight PNGs per comparison under
``plots/prediction_examples/<subject>/window_<n>.png`` and an identical
``plots/prediction_examples/selection.json`` in each root. Fits and report
metrics remain read only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_campaign import EC0G_HZ, STRIDE
from data.fingerflex.targets import TargetRows, load_target_rows
from data.fingerflex.saved_predictions import (
    load_saved_prediction, validate_target_identity,
)
from evaluation.fingerflex_prediction_examples import (
    matched_limits, restore_original_values, select_intervals,
)
from experiments.fingerflex_campaign import split_fit_rows
from scripts.fingerflex_comparison import DETERMINISTIC_MODELS
from plotting.fingerflex_prediction_examples import plot_prediction_window

MIN_ROOT_COUNT = 1


def _comparison_models(config: dict) -> dict[str, list[str]]:
    """Resolve one direct comparison or several configured groups."""
    if "groups" in config:
        return {
            group["name"]: group["models"]
            for group in config["groups"].values()
        }
    return {config["name"]: config["models"]}


def _load_comparison(root: Path, config: dict) -> dict:
    """Validate a completed comparison against the requested plot design."""
    index = json.loads((root / "index.json").read_text())
    summary = json.loads(
        (root / "reports" / "summary.json").read_text()
    )
    if not summary.get("complete"):
        raise ValueError(f"Comparison is incomplete: {root}.")
    if root.name != f"{index['name']}__{index['semantic_id']}":
        raise ValueError(f"Comparison root identity differs: {root}.")
    allowed = _comparison_models(config)
    models = list(index["fit_roots"])
    if allowed.get(index["name"]) != models:
        raise ValueError(f"Unexpected comparison models in {root}.")
    settings = config["prediction_examples"]
    if settings["training_fraction"] not in index["fractions"]:
        raise ValueError(f"Requested training fraction absent in {root}.")
    if settings["neural_seed"] not in index["seeds"]:
        raise ValueError(f"Requested neural seed absent in {root}.")
    if any(
        index["plot_labels"].get(model) != config["plot_labels"][model]
        for model in models
    ):
        raise ValueError(f"Configured plot labels differ in {root}.")
    return index


def _target_rows(index: dict, subject: str) -> TargetRows:
    """Read shared prepared targets through one existing fit root."""
    first_root = Path(next(iter(index["fit_roots"].values())))
    rows = load_target_rows(
        first_root / "prepared" / subject / "common"
    )
    if not np.all(np.diff(rows.test_indices) == STRIDE):
        raise ValueError(
            f"Expected 25 Hz test target rows for {subject}."
        )
    return rows


def _fit_prediction(
    index: dict,
    model: str,
    subject: str,
    rows: TargetRows,
    fraction: float,
    neural_seed: int,
) -> np.ndarray:
    """Load and restore one model's full held-out prediction matrix."""
    seed = None if model in DETERMINISTIC_MODELS else neural_seed
    repeat = "deterministic" if seed is None else f"seed{seed}"
    fit_root = Path(index["fit_roots"][model])
    path = (
        fit_root / "runs" / f"{int(fraction * 100)}pct"
        / model / subject / repeat
    )
    saved = load_saved_prediction(path)
    expected = {
        "model": model, "subject": subject,
        "fraction": fraction, "seed": seed, "delay_ms": 0,
    }
    if any(saved.result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Fit selectors differ in {path / 'result.json'}.")
    fit, validation = split_fit_rows(len(rows.train_y), fraction)
    validate_target_identity(
        saved, rows, fit, validation, path,
    )
    return restore_original_values(saved, rows, fit, str(path))


def _selection_manifest(
    selected: dict[str, list[tuple[int, int]]],
    rows_by_subject: dict[str, TargetRows],
    settings: dict,
) -> dict:
    """Describe shared test rows and sampling settings for both roots."""
    examples = {}
    for subject, intervals in selected.items():
        indices = rows_by_subject[subject].test_indices
        examples[subject] = [
            {
                "interval": number,
                "start_row": start,
                "stop_row_exclusive": stop,
                "first_target_sample": int(indices[start]),
                "last_target_sample": int(indices[stop - 1]),
            }
            for number, (start, stop) in enumerate(intervals, start=1)
        ]
    return {
        "settings": settings,
        "target_sample_hz": EC0G_HZ,
        "examples": examples,
    }


def generate_examples(roots: tuple[Path, ...], config: dict) -> dict:
    """Render matched random test intervals for completed groups.

    Parameters
    ----------
    roots : tuple[Path, ...]
        Existing comparison roots, one for each configured group.
    config : dict
        Portable plot settings, group names, and model labels.

    Returns
    -------
    dict
        The common selection manifest written to both roots.
    """
    expected_names = set(_comparison_models(config))
    if len(roots) < MIN_ROOT_COUNT or len(set(roots)) != len(roots):
        raise ValueError("Distinct comparison roots are required.")
    indexes = [_load_comparison(root, config) for root in roots]
    if {index["name"] for index in indexes} != expected_names:
        raise ValueError("Comparison roots differ from the configuration.")
    reference = indexes[0]
    if any(
        index["subjects"] != reference["subjects"] or
        index["semantics"]["preprocessing_id"] !=
        reference["semantics"]["preprocessing_id"]
        for index in indexes[1:]
    ):
        raise ValueError("Comparison subjects or preprocessing differ.")
    settings = config["prediction_examples"]
    rows_by_subject = {
        subject: _target_rows(indexes[0], subject)
        for subject in indexes[0]["subjects"]
    }
    selected = select_intervals(
        {subject: len(rows.test_y) for subject, rows in (
            rows_by_subject.items()
        )},
        settings["subject_count"],
        settings["intervals_per_subject"],
        settings["duration_seconds"],
        settings["sampling_seed"],
    )
    manifest = _selection_manifest(selected, rows_by_subject, settings)
    fraction = settings["training_fraction"]
    neural_seed = settings["neural_seed"]
    for subject, intervals in selected.items():
        rows = rows_by_subject[subject]
        group_predictions = [
            {
                model: _fit_prediction(
                    index, model, subject, rows, fraction, neural_seed,
                )
                for model in index["fit_roots"]
            }
            for index in indexes
        ]
        for number, (start, stop) in enumerate(intervals, start=1):
            truth = rows.test_y[start:stop]
            time = (
                rows.test_indices[start:stop]
                - rows.test_indices[start]
            ) / EC0G_HZ
            groups = [
                {model: values[start:stop] for model, values in group.items()}
                for group in group_predictions
            ]
            limits = matched_limits(truth, groups)
            for root, index, predictions in zip(
                roots, indexes, groups
            ):
                output = (
                    root / "plots" / "prediction_examples" / subject
                    / f"window_{number}.png"
                )
                labels = {
                    model: index["plot_labels"][model]
                    for model in predictions
                }
                plot_prediction_window(
                    output, subject, number, time, truth,
                    predictions, labels, limits,
                )
    for root in roots:
        path = root / "plots" / "prediction_examples" / "selection.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, path)
    return manifest


def main() -> None:
    """Parse completed comparison roots and generate example plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs" / "fingerflex_hga_controls.json",
    )
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    roots = tuple(path.resolve() for path in arguments.roots)
    generate_examples(roots, config)
    for root in roots:
        print(root / "plots" / "prediction_examples")


if __name__ == "__main__":
    main()
