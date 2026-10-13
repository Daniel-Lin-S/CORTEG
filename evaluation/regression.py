"""Compute finite multi-output regression metrics from saved predictions.

Input: aligned (N,F) targets and predictions. Output: per-output and
unweighted mean Pearson r, MSE, and R2. A flat prediction has r equal
to zero; a constant target is rejected because its R2 is undefined.
"""
from __future__ import annotations

import numpy as np

MIN_ROWS = 2


def finger_correlations(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Return per-output Pearson r; flat predictions contribute zero."""
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError(
            f"Expected equal (rows, outputs) arrays, got "
            f"{truth.shape} and {prediction.shape}."
        )
    if len(truth) < MIN_ROWS or not np.isfinite(truth).all() or (
        not np.isfinite(prediction).all()
    ):
        raise ValueError("Targets are too short or values are nonfinite.")
    centered = truth.astype(np.float64) - truth.mean(axis=0)
    predicted = prediction.astype(np.float64) - prediction.mean(axis=0)
    target_ss = np.sum(centered ** 2, axis=0)
    predicted_ss = np.sum(predicted ** 2, axis=0)
    if (target_ss <= 0).any():
        raise ValueError("At least one target output is constant.")
    correlation = np.zeros(truth.shape[1], dtype=np.float64)
    valid = predicted_ss > 0
    correlation[valid] = np.sum(
        centered[:, valid] * predicted[:, valid], axis=0
    ) / np.sqrt(target_ss[valid] * predicted_ss[valid])
    if not np.isfinite(correlation).all():
        raise ValueError("Pearson correlations are nonfinite.")
    return correlation


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    """Score aligned outputs and return finger and overall summaries."""
    correlations = finger_correlations(truth, prediction)
    target_ss = np.sum(
        (truth - truth.mean(axis=0)) ** 2, axis=0,
    )
    error = (truth - prediction) ** 2
    values = {
        "pearson_r": correlations,
        "mse": np.mean(error, axis=0),
        "r2": 1 - np.sum(error, axis=0) / target_ss,
    }
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Regression metrics are nonfinite.")
    return {
        "overall": {
            key: float(value.mean()) for key, value in values.items()
        },
        "fingers": {
            key: value.tolist() for key, value in values.items()
        },
    }
