"""Read and validate completed FingerFlex test-prediction artifacts.

Input: one fit directory containing ``result.json`` and
``test_predictions.npz``. Output: standardized five-finger arrays and fit
metadata in memory. Dataset loading and model selection happen elsewhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.fingerflex.targets import (
    FINGER_COUNT, TargetRows, target_boundaries,
)


@dataclass(frozen=True)
class SavedPrediction:
    """Saved standardized test arrays and completed fit metadata."""

    truth: np.ndarray
    prediction: np.ndarray
    result: dict


def load_saved_prediction(path: Path) -> SavedPrediction:
    """Read a completed fit's saved standardized test predictions.

    Parameters
    ----------
    path : Path
        Single completed fit directory.

    Returns
    -------
    SavedPrediction
        ``truth`` and ``prediction`` both have shape ``(test rows, 5)``.
    """
    result_path = path / "result.json"
    prediction_path = path / "test_predictions.npz"
    if not result_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(
            f"Completed result and predictions required in {path}."
        )
    result = json.loads(result_path.read_text())
    if not result.get("test_scored_after_restore"):
        raise ValueError(f"Test completion marker absent in {result_path}.")
    with np.load(prediction_path) as saved:
        if set(saved.files) != {"y", "y_hat"}:
            raise ValueError(
                f"Expected y and y_hat arrays in {prediction_path}."
            )
        truth = saved["y"]
        prediction = saved["y_hat"]
    if truth.ndim != 2 or truth.shape[1] != FINGER_COUNT:
        raise ValueError(
            f"Expected prediction shape (rows, {FINGER_COUNT}), "
            f"got {truth.shape} in {prediction_path}."
        )
    if prediction.shape != truth.shape or not np.isfinite(
        truth
    ).all() or not np.isfinite(prediction).all():
        raise ValueError(
            f"Mismatched or nonfinite test predictions in "
            f"{prediction_path}."
        )
    return SavedPrediction(truth, prediction, result)


def validate_target_identity(
    saved: SavedPrediction,
    rows: TargetRows,
    fit: slice,
    validation: slice,
    path: Path,
) -> None:
    """Compare completed fit boundaries with prepared target-row hashes."""
    boundaries = saved.result.get("boundaries", {})
    expected = target_boundaries(rows, fit, validation)
    for key in (
        "fit_indices_sha256", "validation_indices_sha256",
        "test_grid_sha256", "target_values_sha256",
        "test_window_count",
    ):
        if boundaries.get(key) != expected[key]:
            raise ValueError(f"{key} differs in {path}.")
