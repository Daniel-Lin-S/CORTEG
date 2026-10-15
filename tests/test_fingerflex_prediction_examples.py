"""Verify saved FingerFlex prediction sampling and figure inputs.

Input: synthetic prepared targets, completed fit records, and predictions.
Output: assertions and temporary PNG files; no campaign data is changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.fingerflex_campaign import digest_array
from data.fingerflex.saved_predictions import (
    SavedPrediction, load_saved_prediction,
)
from data.fingerflex.targets import TargetRows
from evaluation.fingerflex_prediction_examples import (
    matched_limits, restore_original_values, select_intervals,
)
from experiments.fingerflex_campaign import split_fit_rows
from plotting.fingerflex_prediction_examples import (
    LEGEND_SIZE, TICK_SIZE, plot_prediction_window,
)
from scripts.fingerflex_prediction_examples import _fit_prediction


def _rows() -> TargetRows:
    """Make nonconstant five-finger fit and test targets."""
    train = np.arange(500, dtype=np.float32).reshape(100, 5) / 20
    test = np.arange(1000, 2000, dtype=np.float32).reshape(200, 5) / 30
    return TargetRows(
        train, np.arange(100, dtype=np.int64) * 40,
        test, np.arange(200, dtype=np.int64) * 40 + 5000,
    )


def _save_fit(
    root: Path, model: str, subject: str, seed: int | None,
    rows: TargetRows, shift: float,
) -> Path:
    """Write one synthetic completed fit under a temporary root."""
    repeat = "deterministic" if seed is None else f"seed{seed}"
    path = root / "runs" / "100pct" / model / subject / repeat
    path.mkdir(parents=True)
    fit, validation = split_fit_rows(len(rows.train_y), 1.0)
    mean = rows.train_y[fit].mean(axis=0, keepdims=True)
    std = rows.train_y[fit].std(axis=0, keepdims=True)
    np.savez(
        path / "test_predictions.npz",
        y=((rows.test_y - mean) / std).astype(np.float32),
        y_hat=((rows.test_y + shift - mean) / std).astype(np.float32),
    )
    result = {
        "model": model, "subject": subject, "fraction": 1.0,
        "seed": seed, "delay_ms": 0,
        "test_scored_after_restore": True,
        "boundaries": {
            "fit_indices_sha256": digest_array(rows.train_indices[fit]),
            "validation_indices_sha256": digest_array(
                rows.train_indices[validation]
            ),
            "test_grid_sha256": digest_array(rows.test_indices),
            "target_values_sha256": digest_array(rows.test_y),
            "test_window_count": len(rows.test_y),
        },
    }
    (path / "result.json").write_text(json.dumps(result))
    return path


def test_selection_is_reproducible_and_disjoint() -> None:
    """Sample four subjects and two disjoint 125-row intervals each."""
    counts = {str(index): 500 for index in range(9)}
    first = select_intervals(counts, 4, 2, 5, 20261014)
    second = select_intervals(counts, 4, 2, 5, 20261014)
    assert first == second
    assert len(first) == 4
    for intervals in first.values():
        assert len(intervals) == 2
        assert all(stop - start == 125 for start, stop in intervals)
        assert intervals[0][1] <= intervals[1][0]
    with pytest.raises(ValueError, match="too few rows"):
        select_intervals({"bp": 100}, 1, 2, 5, 20261014)


def test_original_values_and_seed_choice(tmp_path: Path) -> None:
    """Restore each requested seed and ignore it for deterministic ridge."""
    rows = _rows()
    neural_root = tmp_path / "neural"
    ridge_root = tmp_path / "ridge"
    _save_fit(neural_root, "corteg", "bp", 0, rows, 1.0)
    _save_fit(neural_root, "corteg", "bp", 1, rows, 2.0)
    _save_fit(ridge_root, "ridge_late", "bp", None, rows, 3.0)
    index = {"fit_roots": {
        "corteg": str(neural_root),
        "ridge_late": str(ridge_root),
    }}
    np.testing.assert_allclose(
        _fit_prediction(index, "corteg", "bp", rows, 1.0, 0),
        rows.test_y + 1.0, atol=1e-5,
    )
    np.testing.assert_allclose(
        _fit_prediction(index, "corteg", "bp", rows, 1.0, 1),
        rows.test_y + 2.0, atol=1e-5,
    )
    np.testing.assert_allclose(
        _fit_prediction(index, "ridge_late", "bp", rows, 1.0, 1),
        rows.test_y + 3.0, atol=1e-5,
    )
    fit_path = neural_root / "runs/100pct/corteg/bp/seed0"
    result_path = fit_path / "result.json"
    result = json.loads(result_path.read_text())
    result["boundaries"]["test_grid_sha256"] = "wrong"
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="test_grid_sha256"):
        _fit_prediction(index, "corteg", "bp", rows, 1.0, 0)


def test_missing_nonfinite_and_truth_mismatch(tmp_path: Path) -> None:
    """Fail clearly on incomplete or inconsistent saved predictions."""
    with pytest.raises(FileNotFoundError, match="required"):
        load_saved_prediction(tmp_path)
    rows = _rows()
    fit_path = _save_fit(
        tmp_path, "corteg", "bp", 0, rows, 0.0,
    )
    saved = load_saved_prediction(fit_path)
    fit, _ = split_fit_rows(len(rows.train_y), 1.0)
    altered = SavedPrediction(
        saved.truth + 1.0, saved.prediction, saved.result,
    )
    with pytest.raises(ValueError, match="Saved standardized truth"):
        restore_original_values(altered, rows, fit, str(fit_path))
    np.savez(
        fit_path / "test_predictions.npz",
        y=saved.truth, y_hat=np.full_like(saved.truth, np.nan),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        load_saved_prediction(fit_path)


def test_shared_limits_and_readable_trace_figure(tmp_path: Path) -> None:
    """Use common finger limits and render all five panels."""
    time = np.arange(125) / 25
    truth = np.stack([
        np.sin(time + finger) for finger in range(5)
    ], axis=1)
    predictions = {"corteg": truth + 0.2}
    groups = [predictions, {"ridge_late": truth - 0.4}]
    limits = matched_limits(truth, groups)
    assert np.all(limits[:, 0] < (truth - 0.4).min(axis=0))
    assert np.all(limits[:, 1] > (truth + 0.2).max(axis=0))
    assert TICK_SIZE >= 15 and LEGEND_SIZE >= 16
    output = tmp_path / "example.png"
    plot_prediction_window(
        output, "bp", 1, time, truth, predictions,
        {"corteg": "CORTEG (LFS + single-band HGA)"}, limits,
    )
    assert output.stat().st_size > 0
