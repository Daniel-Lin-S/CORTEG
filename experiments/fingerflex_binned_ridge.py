"""Fit one FingerFlex ridge after joint temporal-bin and L2 selection.

Input: shared prepared/<subject>/{common,corteg/0} arrays, subject,
training fraction, and resolved binning settings. Output: one structured
fit folder with cv_history.json, normalization.npz, best.joblib,
test_predictions.npz, and result.json written last.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing.channel_time_features import transform_streams
from data.fingerflex_linear import load_fingerflex_linear
from preprocessing.temporal_bins import mean_time_bins
from evaluation.regression import finger_correlations
from experiments.fingerflex_campaign import split_fit_rows
from experiments.fingerflex_ridge import _save_fit
from experiments.ridge_cv import _folds, choose_bin_alpha
from models.linear_fusion import RidgeFusion, make_ridge

MODEL = "ridge_early_binned"
STREAMS = ("broadband", "high_gamma")
WINDOW_MS = 1000


def _binned_streams(
    sources: tuple[np.ndarray, np.ndarray], bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply matching time bins to broadband and HGA windows."""
    return tuple(mean_time_bins(source, bins) for source in sources)


def _cross_validate(
    sources: tuple[np.ndarray, np.ndarray], target: np.ndarray,
    settings: dict,
) -> tuple[int, float, dict]:
    """Search all temporal-bin and alpha pairs on purged folds."""
    bins = tuple(settings["bin_counts"])
    alphas = tuple(settings["alpha_per_training_row"])
    folds = _folds(
        len(target), settings["folds"], settings["purge_rows"],
    )
    scores = np.empty((len(bins), len(alphas), len(folds)))
    for bin_index, count in enumerate(bins):
        binned = _binned_streams(sources, count)
        for fold_index, (training, held_out) in enumerate(folds):
            x_train, x_held, _ = transform_streams(
                tuple(source[training] for source in binned),
                tuple(source[held_out] for source in binned),
            )
            for alpha_index, ratio in enumerate(alphas):
                estimator = make_ridge(ratio * len(training))
                estimator.fit(x_train, target[training])
                prediction = estimator.predict(x_held)
                scores[bin_index, alpha_index, fold_index] = (
                    finger_correlations(
                        target[held_out], prediction,
                    ).mean()
                )
    bin_index, alpha_index = choose_bin_alpha(
        scores, bins, alphas, len(folds),
    )
    history = {
        "bin_counts": bins,
        "alpha_per_training_row": alphas,
        "folds": [
            {
                "train_rows": len(training),
                "held_out_rows": len(held_out),
                "held_out_first": int(held_out[0]),
                "held_out_last": int(held_out[-1]),
            }
            for training, held_out in folds
        ],
        "scores": scores.tolist(),
        "selected_bin_count": bins[bin_index],
        "selected_alpha_per_row": alphas[alpha_index],
    }
    return bins[bin_index], alphas[alpha_index], history


def fit_binned_ridge(
    prepared: Path, subject: str, fraction: float,
    output: Path, settings: dict,
) -> None:
    """Fit, score, and save one deterministic binned early ridge."""
    if (output / "result.json").is_file():
        return
    arrays = load_fingerflex_linear(prepared, subject)
    fit, validation = split_fit_rows(len(arrays.train_y), fraction)
    target = arrays.train_y[fit]
    sources = (
        arrays.train_broadband[fit], arrays.train_high_gamma[fit],
    )
    bins, ratio, history = _cross_validate(sources, target, settings)
    training = _binned_streams(sources, bins)
    testing = _binned_streams(
        (arrays.test_broadband, arrays.test_high_gamma), bins,
    )
    x_train, x_test, scalers = transform_streams(training, testing)
    estimator = make_ridge(ratio * len(target))
    estimator.fit(x_train, target)
    model = RidgeFusion("ridge_early", {"joint": estimator})
    identity = {
        "model": MODEL,
        "subject": subject,
        "fraction": fraction,
        "seed": None,
        "delay_ms": 0,
        "boundaries": arrays.boundaries(fit, validation),
        "selected_bin_count": bins,
        "selected_bin_duration_ms": WINDOW_MS / bins,
    }
    output.mkdir(parents=True, exist_ok=True)
    _save_fit(
        output, model, {"joint": x_test},
        {"selected": {"joint": ratio}, "history": history},
        scalers, target, arrays.test_y, identity,
    )
