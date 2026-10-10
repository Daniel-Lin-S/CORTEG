"""Fit one deterministic Stanford wavelet PLS regression.

Input is a shared PLS feature cache and common target arrays. Output under
one fit directory is normalization.npz, component_history.jsonl, best.joblib,
test_predictions.npz, and result.json. Test is scored only after loading the
validation-selected PLS estimator.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from data.fingerflex_campaign import digest_array
from data.fingerflex_wavelet import feature_root
from experiments.fingerflex_campaign import (
    CLIP_Z,
    MIN_STD,
    metric_values,
    split_fit_rows,
)

DEFAULT_MAX_COMPONENTS = 50


def _standardize(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Scale (N,C,10,15) features using selected fit rows only."""
    mean = train.mean(axis=0, keepdims=True)
    std = np.maximum(train.std(axis=0, keepdims=True), MIN_STD)
    arrays = tuple(np.clip(
        (part - mean) / (std + 1e-6), -CLIP_Z, CLIP_Z,
    ).reshape(len(part), -1).astype(np.float32) for part in (
        train, val, test,
    ))
    if any(not np.isfinite(part).all() for part in arrays):
        raise ValueError("PLS normalization produced nonfinite features.")
    return arrays, mean, std


def fit_pls(
    prepared: Path,
    config: dict,
    subject: str,
    fraction: float,
    output: Path,
    max_components: int = DEFAULT_MAX_COMPONENTS,
) -> None:
    """Select PLS components on validation rows and save one complete fit.

    Parameters
    ----------
    prepared : Path
        Shared prepared-cache subjects directory.
    config : dict
        Resolved configuration including artifact location.
    subject : str
        Patient identifier.
    fraction : float
        Proportion of non-validation training windows to fit.
    output : Path
        New model-specific fit directory.
    """
    if (output / "result.json").is_file():
        return
    output.mkdir(parents=True, exist_ok=True)
    common = prepared / subject / "common"
    features = feature_root(config, subject)
    train_y = np.load(common / "train_y.npy")
    test_y = np.load(common / "test_y.npy")
    fit, val = split_fit_rows(len(train_y), fraction)
    train_x = np.load(features / "train_x.npy", mmap_mode="r")
    test_x = np.load(features / "test_x.npy", mmap_mode="r")
    if len(train_x) != len(train_y) or len(test_x) != len(test_y):
        raise ValueError(
            f"PLS features and targets have unequal rows for {subject}."
        )
    (x_fit, x_val, x_test), x_mean, x_std = _standardize(
        train_x[fit], train_x[val], test_x,
    )
    y_mean = train_y[fit].mean(axis=0, keepdims=True)
    y_std = train_y[fit].std(axis=0, keepdims=True)
    if (y_std <= 0).any():
        raise ValueError(
            f"PLS fit targets contain a constant finger: {subject}."
        )
    if max_components < 1:
        raise ValueError("PLS maximum components must be positive.")
    y_fit, y_val, y_test = (
        ((part - y_mean) / y_std).astype(np.float32)
        for part in (train_y[fit], train_y[val], test_y)
    )
    np.savez(
        output / "normalization.npz", x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
    )
    component_limit = min(
        max_components, len(x_fit) - 1, x_fit.shape[1],
    )
    if component_limit < 1:
        raise ValueError("PLS fit has no allowable components.")
    best_score = -np.inf
    best_count = None
    history = output / "component_history.jsonl"
    with history.open("w") as stream:
        for count in range(1, component_limit + 1):
            estimator = PLSRegression(n_components=count)
            estimator.fit(x_fit, y_fit)
            score = metric_values(
                y_val, estimator.predict(x_val),
            )["overall"]["pearson_r"]
            stream.write(json.dumps({
                "components": count,
                "validation_pearson_r": score,
            }) + "\n")
            stream.flush()
            if score > best_score:
                best_score = score
                best_count = count
                joblib.dump(estimator, output / "best.joblib")
    if best_count is None:
        raise RuntimeError("PLS validation selected no component count.")
    estimator = joblib.load(output / "best.joblib")
    truth = y_test
    prediction = estimator.predict(x_test)
    metrics = metric_values(truth, prediction)
    np.savez(
        output / "test_predictions.npz",
        y=truth, y_hat=prediction,
    )
    indices = np.load(common / "train_indices.npy")
    test_indices = np.load(common / "test_indices.npy")
    boundaries = {
        "fit_indices_sha256": digest_array(indices[fit]),
        "validation_indices_sha256": digest_array(indices[val]),
        "test_grid_sha256": digest_array(test_indices),
        "fit_window_count": len(indices[fit]),
        "validation_window_count": len(indices[val]),
        "test_window_count": len(test_indices),
        "fit_first_target": int(indices[fit][0]),
        "fit_last_target": int(indices[fit][-1]),
        "validation_first_target": int(indices[val][0]),
        "test_first_target": int(test_indices[0]),
        "target_values_sha256": digest_array(test_y),
    }
    result = {
        "model": "pls",
        "subject": subject,
        "fraction": fraction,
        "seed": None,
        "delay_ms": 0,
        "selected_components": best_count,
        "validation_selection_metric": "mean_finger_pearson_r",
        "validation_pearson_r": best_score,
        "metrics": metrics,
        "boundaries": boundaries,
        "test_scored_after_restore": True,
        "history_path": str(history),
        "checkpoint_path": str(output / "best.joblib"),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
