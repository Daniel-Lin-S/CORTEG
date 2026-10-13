"""Identify stable FingerFlex ridge fits by fitting semantics.

Input: resolved source campaign settings and ridge model name. Output:
a semantic index and a fit root named <model>__<semantic hash>.
Invocation choices such as subjects, seeds, workers, and devices are
excluded from the semantic identity.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from preprocessing.channel_time_features import MIN_STD
from data.fingerflex_cache import preparation_id
from experiments.ridge_cv import ALPHA_PER_ROW, FOLDS, PURGE_ROWS
from models.linear_fusion import MODES, RIDGE_MAX_ITER, RIDGE_TOL

IDENTITY_LENGTH = 12
DELAY_MS = 0
BINNED_MODEL = "ridge_early_binned"


def ridge_semantics(base: dict, model: str) -> dict:
    """Describe settings that determine one ridge fit's predictions."""
    if model not in MODES:
        raise ValueError(f"Unknown ridge model: {model}.")
    return {
        "model": model,
        "preprocessing_id": preparation_id(base),
        "delay_ms": DELAY_MS,
        "input_streams": (
            "corteg_broadband", "corteg_high_gamma_channel_zero",
        ),
        "feature_scaling": {
            "axes": ("training_rows", "time"),
            "per_channel": True,
            "min_std": MIN_STD,
            "flatten": "channel_then_time",
        },
        "ridge": {
            "fit_intercept": True,
            "solver": "lsqr",
            "tol": RIDGE_TOL,
            "max_iter": RIDGE_MAX_ITER,
        },
        "selection": {
            "folds": FOLDS,
            "contiguous": True,
            "purge_rows": PURGE_ROWS,
            "alpha_per_training_row": ALPHA_PER_ROW,
            "metric": "mean_finger_pearson_r",
            "tie_break": "largest_alpha",
        },
        "fusion": (
            "concatenate_before_ridge" if model == "ridge_early"
            else "independent_ridge_convex_oof_weight_per_finger"
        ),
        "target_scaling": "fit_rows_for_saved_test_scores",
    }


def ridge_identity(base: dict, model: str) -> str:
    """Hash canonical fitting semantics independently of invocation."""
    canonical = json.dumps(
        ridge_semantics(base, model),
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:IDENTITY_LENGTH]


def ridge_root(base: dict, model: str) -> Path:
    """Locate the stable semantic root for one ridge model."""
    return (
        Path(base["artifact_base"]) / "fits"
        / f"{model}__{ridge_identity(base, model)}"
    )


def ridge_index(base: dict, model: str) -> dict:
    """Build the immutable per-model fit-root index."""
    return {
        "model": model,
        "semantic_id": ridge_identity(base, model),
        "preprocessing_id": preparation_id(base),
        "semantics": ridge_semantics(base, model),
    }


def binned_ridge_semantics(base: dict, settings: dict) -> dict:
    """Describe joint bin/alpha selection for the binned ridge arm."""
    expected = {
        "bin_counts", "folds", "purge_rows",
        "alpha_per_training_row",
    }
    if set(settings) != expected:
        raise ValueError(
            f"Binned ridge settings need {sorted(expected)}, "
            f"got {sorted(settings)}."
        )
    bins = settings["bin_counts"]
    alphas = settings["alpha_per_training_row"]
    if not bins or len(set(bins)) != len(bins) or any(
        type(value) is not int or not 1 <= value <= 128
        for value in bins
    ):
        raise ValueError(f"Invalid temporal bin counts: {bins}.")
    if type(settings["folds"]) is not int or (
        settings["folds"] < 2
    ) or type(settings["purge_rows"]) is not int or (
        settings["purge_rows"] < 0
    ):
        raise ValueError("CV folds or purge rows are invalid.")
    if not alphas or any(
        type(value) not in (int, float) or not math.isfinite(value)
        or value <= 0 for value in alphas
    ):
        raise ValueError(f"Invalid alpha grid: {alphas}.")
    semantics = ridge_semantics(base, "ridge_early")
    semantics["model"] = BINNED_MODEL
    semantics["temporal_binning"] = {
        "recipe": "equal_duration_mean_v1",
        "window_ms": 1000,
        "bin_counts": bins,
        "sample_assignment": "ceil_interval_boundaries",
    }
    semantics["selection"] = {
        "folds": settings["folds"],
        "contiguous": True,
        "purge_rows": settings["purge_rows"],
        "alpha_per_training_row": alphas,
        "metric": "mean_finger_pearson_r",
        "tie_break": "largest_alpha_then_fewest_bins",
    }
    return semantics


def binned_ridge_identity(base: dict, settings: dict) -> str:
    """Hash binned fitting semantics without invocation settings."""
    canonical = json.dumps(
        binned_ridge_semantics(base, settings),
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[
        :IDENTITY_LENGTH
    ]


def binned_ridge_root(base: dict, settings: dict) -> Path:
    """Locate the stable hashed root for binned early ridge fits."""
    return (
        Path(base["artifact_base"]) / "fits"
        / f"{BINNED_MODEL}__{binned_ridge_identity(base, settings)}"
    )
