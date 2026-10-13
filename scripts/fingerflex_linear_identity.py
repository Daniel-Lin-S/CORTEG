"""Identify stable FingerFlex ridge fits by fitting semantics.

Input: resolved source campaign settings and ridge model name. Output:
a semantic index and a fit root named <model>__<semantic hash>.
Invocation choices such as subjects, seeds, workers, and devices are
excluded from the semantic identity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.channel_time_features import MIN_STD
from data.fingerflex_cache import preparation_id
from experiments.ridge_cv import ALPHA_PER_ROW, FOLDS, PURGE_ROWS
from models.linear_fusion import MODES, RIDGE_MAX_ITER, RIDGE_TOL

IDENTITY_LENGTH = 12
DELAY_MS = 0


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
