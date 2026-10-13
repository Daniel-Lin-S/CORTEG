"""Coordinate one Stanford FingerFlex early or late ridge fit.

Input: prepared/<subject>/common/{train,test}_{y,indices}.npy and
prepared/<subject>/corteg/0/{train,test}_{x,hi}.npy; a fit fraction,
model name, and output directory. Output: cv_history.json,
normalization.npz, best.joblib, test_predictions.npz, and result.json.
The result marker is written last so interrupted fits can be quarantined.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from preprocessing.channel_time_features import transform_streams
from data.fingerflex_linear import load_fingerflex_linear
from experiments.fingerflex_campaign import split_fit_rows
from evaluation.regression import (
    finger_correlations, regression_metrics,
)
from experiments.ridge_cv import (
    ALPHA_PER_ROW, FOLDS, _folds, choose_alpha, convex_weights,
)
from models.linear_fusion import MODES, RidgeFusion, make_ridge

STREAMS = ("broadband", "high_gamma")
SELECTION_METRIC = "five_blocked_fold_mean_finger_pearson_r"


def _cross_validate(
    sources: tuple[np.ndarray, ...],
    target: np.ndarray,
) -> tuple[float, np.ndarray, list[dict]]:
    """Apply the CV strategy to scaled ridge stream designs."""
    predictions = np.empty(
        (len(ALPHA_PER_ROW), len(target), target.shape[1]),
        dtype=np.float32,
    )
    fold_scores = np.empty((len(ALPHA_PER_ROW), FOLDS))
    history = []
    for fold, (training, held_out) in enumerate(_folds(len(target))):
        x_train, x_held, _ = transform_streams(
            tuple(source[training] for source in sources),
            tuple(source[held_out] for source in sources),
        )
        for index, ratio in enumerate(ALPHA_PER_ROW):
            estimator = make_ridge(ratio * len(training))
            estimator.fit(x_train, target[training])
            predicted = estimator.predict(x_held)
            predictions[index, held_out] = predicted
            fold_scores[index, fold] = finger_correlations(
                target[held_out], predicted,
            ).mean()
        history.append({
            "fold": fold,
            "train_rows": len(training),
            "held_out_rows": len(held_out),
            "scores": fold_scores[:, fold].tolist(),
        })
    best = choose_alpha(fold_scores)
    return ALPHA_PER_ROW[best], predictions[best], history


def _refit(
    sources: tuple[np.ndarray, ...],
    targets: np.ndarray,
    test_sources: tuple[np.ndarray, ...],
    ratio: float,
) -> tuple[object, np.ndarray, tuple]:
    """Fit the selected ridge on all fit rows and transform test rows."""
    x_train, x_test, scalers = transform_streams(
        sources, test_sources,
    )
    estimator = make_ridge(ratio * len(targets))
    estimator.fit(x_train, targets)
    return estimator, x_test, scalers


def _fit_early(
    training: tuple[np.ndarray, np.ndarray],
    target: np.ndarray,
    testing: tuple[np.ndarray, np.ndarray],
) -> tuple[RidgeFusion, dict, dict, tuple]:
    """Select and refit one ridge on concatenated feature streams."""
    ratio, _, history = _cross_validate(training, target)
    estimator, design, scalers = _refit(
        training, target, testing, ratio,
    )
    model = RidgeFusion("ridge_early", {"joint": estimator})
    return model, {"joint": design}, {
        "selected": {"joint": ratio},
        "history": {"joint": history},
    }, scalers


def _fit_late(
    training: tuple[np.ndarray, np.ndarray],
    target: np.ndarray,
    testing: tuple[np.ndarray, np.ndarray],
) -> tuple[RidgeFusion, dict, dict, tuple]:
    """Select separate ridge strengths and convex OOF fusion weights."""
    selected, history, oof, estimators, designs = {}, {}, {}, {}, {}
    scalers = []
    for key, source, test_source in zip(
        ("first", "second"), training, testing,
    ):
        ratio, oof[key], history[key] = _cross_validate(
            (source,), target,
        )
        estimator, designs[key], statistics = _refit(
            (source,), target, (test_source,), ratio,
        )
        selected[key] = ratio
        estimators[key] = estimator
        scalers.append(statistics[0])
    weights = convex_weights(
        oof["first"], oof["second"], target,
    )
    model = RidgeFusion("ridge_late", estimators, weights)
    return model, designs, {
        "selected": selected,
        "history": history,
    }, tuple(scalers)


def _save_fit(
    output: Path,
    model: RidgeFusion,
    designs: dict,
    selection: dict,
    scalers: tuple,
    target: np.ndarray,
    test_y: np.ndarray,
    identity: dict,
) -> None:
    """Persist a fitted model, predictions, then its completion marker."""
    y_mean = target.mean(axis=0, keepdims=True)
    y_std = target.std(axis=0, keepdims=True)
    if (y_std <= 0).any():
        raise ValueError("At least one fitted finger target is constant.")
    truth = ((test_y - y_mean) / y_std).astype(np.float32)
    prediction = (
        (model.predict(designs) - y_mean) / y_std
    ).astype(np.float32)
    metrics = regression_metrics(truth, prediction)
    normalization = {"y_mean": y_mean, "y_std": y_std}
    for name, scaler in zip(STREAMS, scalers):
        normalization[f"{name}_mean"] = scaler.mean
        normalization[f"{name}_std"] = scaler.std
    np.savez(output / "normalization.npz", **normalization)
    (output / "cv_history.json").write_text(
        json.dumps(selection["history"], indent=2) + "\n"
    )
    joblib.dump({
        "model": model,
        "scalers": scalers,
        "selected_alpha_per_row": selection["selected"],
        "selected_bin_count": identity.get("selected_bin_count"),
    }, output / "best.joblib")
    np.savez(
        output / "test_predictions.npz",
        y=truth, y_hat=prediction,
    )
    result = {
        **identity,
        "validation_selection_metric": SELECTION_METRIC,
        "selected_alpha_per_row": selection["selected"],
        "fusion_weights": (
            model.weights.tolist() if model.weights is not None else None
        ),
        "metrics": metrics,
        "test_scored_after_restore": True,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def fit_ridge(
    prepared: Path,
    subject: str,
    fraction: float,
    model_name: str,
    output: Path,
) -> None:
    """Fit and save one deterministic early or late ridge model."""
    if model_name not in MODES:
        raise ValueError(f"Unknown ridge model: {model_name}.")
    if (output / "result.json").is_file():
        return
    arrays = load_fingerflex_linear(prepared, subject)
    fit, validation = split_fit_rows(len(arrays.train_y), fraction)
    target = arrays.train_y[fit]
    training = (
        arrays.train_broadband[fit], arrays.train_high_gamma[fit],
    )
    testing = (arrays.test_broadband, arrays.test_high_gamma)
    if model_name == "ridge_early":
        model, designs, selection, scalers = _fit_early(
            training, target, testing,
        )
    else:
        model, designs, selection, scalers = _fit_late(
            training, target, testing,
        )
    identity = {
        "model": model_name,
        "subject": subject,
        "fraction": fraction,
        "seed": None,
        "delay_ms": 0,
        "boundaries": arrays.boundaries(fit, validation),
    }
    output.mkdir(parents=True, exist_ok=True)
    _save_fit(
        output, model, designs, selection, scalers,
        target, arrays.test_y, identity,
    )
