"""Coordinate one late-fusion HGA control fit without dataset I/O.

Input: prepared arrays, optional filterbank cache, subject/fraction,
model settings, and output root. Output: independent CV units, selected
branch settings, final model checkpoints, normalized test predictions,
and one result.json completion marker per fit seed.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch

from preprocessing.fingerflex_filterbank_cache import load_filterbank
from data.fingerflex.spectral_streams import load_spectral_streams
from data.fingerflex.targets import target_boundaries
from evaluation.regression import regression_metrics
from experiments.fingerflex_campaign import split_fit_rows
from experiments.fingerflex_ridge import _cross_validate, _refit
from experiments.hga_control_cv import (
    NEURAL_MODELS, SVR_MODEL, candidates, cv_unit_path,
    fit_branch, select_branch,
)
from experiments.ridge_cv import convex_weights

FILTERBANK_MODELS = (
    "ridge_filterbank_average_late",
    "ridge_filterbank_bands_late",
)
NEW_MODELS = FILTERBANK_MODELS + (
    "cnn_gru_late", SVR_MODEL, "mlp_late",
)


def _quarantine_cv_unit(unit: Path, cv_root: Path) -> None:
    """Move an incomplete CV unit aside before resuming."""
    if not unit.exists():
        return
    quarantine = cv_root / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    unit.rename(
        quarantine / f"{unit.parent.name}_"
        f"{unit.name}_{uuid.uuid4().hex}"
    )


def _legacy_cv_resolver(
    model: str, settings: dict, cv_root: Path,
    channels: int, timepoints: int,
) -> tuple[
    Callable[[int, int | None, int], Path] | None,
    list[int] | None,
]:
    """Reuse completed old-grid units and isolate new CV units."""
    previous_grid = settings.get("cnn_gru_legacy_grid")
    if model != "cnn_gru_late" or previous_grid is None:
        return None, None
    current, _ = candidates(model, settings, channels, timepoints)
    previous, _ = candidates(
        model, {"cnn_gru": previous_grid}, channels, timepoints,
    )
    indices = [previous.index(candidate) for candidate in current]

    def resolve(index: int, seed: int | None, fold: int) -> Path:
        current_unit = cv_unit_path(
            cv_root / "reduced_grid", index, seed, fold,
        )
        if (current_unit / "score.json").is_file() and (
            current_unit / "prediction.npy"
        ).is_file():
            return current_unit
        _quarantine_cv_unit(current_unit, cv_root)
        old_unit = cv_unit_path(
            cv_root, indices[index], seed, fold,
        )
        if (old_unit / "score.json").is_file() and (
            old_unit / "prediction.npy"
        ).is_file():
            return old_unit
        _quarantine_cv_unit(old_unit, cv_root)
        return current_unit

    return resolve, indices


def fit_path(root: Path, model: str, fraction: float,
             subject: str, seed: int | None) -> Path:
    """Locate one seed-specific or deterministic final fit."""
    repeat = "deterministic" if seed is None else f"seed{seed}"
    return (
        root / "runs" / f"{int(fraction * 100)}pct"
        / model / subject / repeat
    )


def _hga_arrays(arrays: object, model: str,
                cache: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Select one HGA representation with aligned row counts."""
    if model in FILTERBANK_MODELS:
        if cache is None:
            raise ValueError("Filterbank HGA needs a completed cache.")
        resolved = model == "ridge_filterbank_bands_late"
        training = load_filterbank(cache, "train", resolved)
        testing = load_filterbank(cache, "test", resolved)
    else:
        training = arrays.train_high_gamma
        testing = arrays.test_high_gamma
    if len(training) != len(arrays.targets.train_y) or (
        len(testing) != len(arrays.targets.test_y)
    ):
        raise ValueError("HGA and target row counts differ.")
    return training, testing


def _ridge_hga(
    training: np.ndarray, target: np.ndarray,
    testing: np.ndarray,
    cv_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict, object, object]:
    """Select HGA ridge strength and predict held-out test rows."""
    ratio, oof, history = _cross_validate(
        (training,), target, cv_root,
    )
    estimator, design, scalers = _refit(
        (training,), target, (testing,), ratio,
    )
    prediction = estimator.predict(design)
    return oof, prediction, {
        "selected_alpha_per_row": ratio,
        "history": history,
    }, estimator, scalers[0]


def _save_final(
    output: Path, identity: dict,
    low_model: object, low_scaler: object,
    low_selection: dict, high_selection: dict,
    high_oof: np.ndarray, low_oof: np.ndarray,
    low_test: np.ndarray, high_test: np.ndarray,
    target: np.ndarray, test_y: np.ndarray,
    high_model: object | None,
) -> None:
    """Save complete final predictions and marker after fusion."""
    weights = convex_weights(low_oof, high_oof, target)
    raw_prediction = weights * low_test + (1 - weights) * high_test
    mean = target.mean(axis=0, keepdims=True)
    std = target.std(axis=0, keepdims=True)
    if (std <= 0).any():
        raise ValueError("At least one fitted finger target is constant.")
    truth = ((test_y - mean) / std).astype(np.float32)
    prediction = ((raw_prediction - mean) / std).astype(np.float32)
    metrics = regression_metrics(truth, prediction)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": low_model, "scaler": low_scaler,
        "selected_alpha_per_row": low_selection["selected_alpha_per_row"],
    }, output / "lfs.joblib")
    if high_model is not None:
        joblib.dump(high_model, output / "best.joblib")
    np.savez(
        output / "lfs_normalization.npz",
        mean=low_scaler.mean, std=low_scaler.std,
    )
    np.savez(
        output / "test_predictions.npz", y=truth, y_hat=prediction,
    )
    np.savez(
        output / "branch_predictions.npz",
        lfs=low_test, hga=high_test,
    )
    (output / "cv_history.json").write_text(json.dumps({
        "lfs": low_selection,
        "hga": high_selection,
    }, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps({
        **identity,
        "selected_alpha_per_row": {
            "lfs": low_selection["selected_alpha_per_row"],
            "hga": high_selection.get("selected_alpha_per_row"),
        },
        "selected_hga_candidate": high_selection.get("candidate"),
        "fusion_weights": weights.tolist(),
        "metrics": metrics,
        "test_scored_after_restore": True,
    }, indent=2) + "\n")


def fit_control(
    prepared: Path, model: str, subject: str,
    fraction: float, root: Path, settings: dict,
    seeds: list[int], device: torch.device,
    cache: Path | None = None,
) -> None:
    """Fit one subject/fraction for a configured HGA control.

    Parameters
    ----------
    prepared : Path
        Shared preparation's subjects directory.
    model : str
        One of five new late-fusion model identifiers.
    subject : str
        Stanford subject code.
    fraction : float
        Fraction of chronological nonvalidation training rows.
    root : Path
        Semantic fit root.
    settings : dict
        Portable scientific grid and trainer settings plus local SVR
        cache limit supplied at invocation.
    seeds : list[int]
        Three neural random seeds.
    device : torch.device
        Allocated CPU or CUDA device.
    cache : Path, optional
        Completed subject filterbank directory; default None.
    """
    if model not in NEW_MODELS:
        raise ValueError(f"Unknown HGA control: {model}.")
    repeats = seeds if model in NEURAL_MODELS else [None]
    if all(
        (fit_path(root, model, fraction, subject, seed)
         / "result.json").is_file() for seed in repeats
    ):
        return
    arrays = load_spectral_streams(prepared, subject)
    high_train, high_test = _hga_arrays(arrays, model, cache)
    fit, validation = split_fit_rows(len(arrays.targets.train_y), fraction)
    target = arrays.targets.train_y[fit]
    lfs_fit = arrays.train_broadband[fit]
    high_fit = high_train[fit]
    cv_root = (
        root / "cv" / f"{int(fraction * 100)}pct" / subject
    )
    low_ratio, low_oof, low_history = _cross_validate(
        (lfs_fit,), target, cv_root / "lfs",
    )
    low_model, low_design, low_scalers = _refit(
        (lfs_fit,), target, (arrays.test_broadband,), low_ratio,
    )
    low_test = low_model.predict(low_design)
    low_selection = {
        "selected_alpha_per_row": low_ratio,
        "history": low_history,
    }
    if model in FILTERBANK_MODELS:
        high_oof, high_test_pred, high_selection, high_model, scaler = (
            _ridge_hga(
                high_fit, target, high_test, cv_root / "hga",
            )
        )
        high_selection["feature_scaler"] = {
            "mean_shape": list(scaler.mean.shape),
            "std_shape": list(scaler.std.shape),
        }
        selected_oof = {None: high_oof}
    else:
        validation_x = high_train[validation]
        validation_y = arrays.targets.train_y[validation]
        resolver, legacy_indices = _legacy_cv_resolver(
            model, settings, cv_root / "hga",
            high_fit.shape[1], high_fit.shape[2],
        )
        candidate, selected_oof, history = select_branch(
            model, high_fit, target,
            validation_x, validation_y,
            cv_root / "hga", settings, seeds, device,
            unit_resolver=resolver,
        )
        if legacy_indices is not None:
            history["legacy_candidate_indices"] = legacy_indices
        high_selection = {"candidate": candidate, **history}
    identity = {
        "model": model, "subject": subject,
        "fraction": fraction, "delay_ms": 0,
        "boundaries": target_boundaries(arrays.targets, fit, validation),
    }
    for seed in repeats:
        output = fit_path(root, model, fraction, subject, seed)
        if (output / "result.json").is_file():
            continue
        if model in FILTERBANK_MODELS:
            prediction = high_test_pred
            fitted_high = {
                "model": high_model, "scaler": scaler,
                "selected_alpha_per_row": high_selection[
                    "selected_alpha_per_row"
                ],
            }
            output.mkdir(parents=True, exist_ok=True)
            np.savez(
                output / "normalization.npz",
                x_mean=scaler.mean, x_std=scaler.std,
            )
        else:
            prediction, _ = fit_branch(
                model, candidate, seed,
                high_fit, target,
                high_train[validation], arrays.targets.train_y[validation],
                high_test, output, settings, device,
            )
            fitted_high = (
                joblib.load(output / "best.joblib")
                if model == SVR_MODEL else None
            )
        _save_final(
            output, {**identity, "seed": seed},
            low_model, low_scalers[0], low_selection,
            high_selection, selected_oof[seed], low_oof,
            low_test, prediction,
            target, arrays.targets.test_y, fitted_high,
        )
