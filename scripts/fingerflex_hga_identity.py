"""Resolve semantic identities for five Stanford HGA controls.

Input: portable scientific settings and resolved source preprocessing.
Output: stable <readable_name>__<hash> fit/cache roots. Selectors,
source revision, paths, and machine resource limits are excluded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.fingerflex_cache import preparation_id
from experiments.fingerflex_hga_controls import (
    FILTERBANK_MODELS, NEW_MODELS,
)
from experiments.ridge_cv import ALPHA_PER_ROW, FOLDS, PURGE_ROWS
from models.linear_fusion import RIDGE_MAX_ITER, RIDGE_TOL
from models.temporal_spatial_cnn_gru import (
    FILTERS, FILTER_LENGTH, GRU_WIDTH, POOL_LENGTH, POOL_STRIDE,
)
from preprocessing.channel_time_features import MIN_STD

HASH_LENGTH = 12
CV_SEEDS = (0, 1, 2)


def semantic_hash(value: dict) -> str:
    """Hash one canonical fitting or preprocessing recipe."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:HASH_LENGTH]


def validate_settings(settings: dict) -> None:
    """Require configured CV semantics to match reused ridge rules."""
    selection = settings["selection"]
    expected = {
        "folds": FOLDS,
        "purge_rows": PURGE_ROWS,
        "alpha_per_training_row": list(ALPHA_PER_ROW),
        "metric": "mean_five_finger_pearson_r",
        "neural_ties": "candidate_order",
        "ridge_ties": "stronger_alpha",
    }
    if selection != expected:
        raise ValueError(
            f"HGA controls require the existing ridge CV settings; "
            f"expected {expected}, got {selection}."
        )
    if tuple(settings["seeds"]) != CV_SEEDS:
        raise ValueError(
            f"Neural CV selection needs seeds {CV_SEEDS}."
        )
    if settings["filterbank"]["components"] != 11:
        raise ValueError("Filterbank comparison needs 11 components.")


def filterbank_semantics(base: dict, settings: dict) -> dict:
    """Describe the raw-window Gaussian feature recipe."""
    return {
        "preprocessing_id": preparation_id(base),
        "feature": settings["filterbank"],
        "input": "cleaned_one_second_window_per_prepared_target",
        "output": "component_envelopes_and_component_average",
    }


def filterbank_root(base: dict, settings: dict) -> Path:
    """Locate one stable derived-feature cache."""
    identifier = semantic_hash(filterbank_semantics(base, settings))
    return (
        Path(base["artifact_base"]) / "preprocessing"
        / f"filterbank_hga__{identifier}"
    )


def fit_semantics(base: dict, model: str,
                  settings: dict) -> dict:
    """Describe choices that can change one fitted prediction."""
    if model not in NEW_MODELS:
        raise ValueError(f"Unknown HGA control: {model}.")
    result = {
        "model": model,
        "preprocessing_id": preparation_id(base),
        "delay_ms": 0,
        "lfs": "prepared_anti_aliased_128_sample_waveform",
        "hga": (
            "prepared_single_band_70_to_200_hz_envelope"
            if model not in FILTERBANK_MODELS
            else "filterbank_component_average"
            if model == FILTERBANK_MODELS[0]
            else "filterbank_11_component_amplitudes"
        ),
        "feature_scaling": {
            "axes": "fit_rows_and_time_per_channel_component",
            "min_std": MIN_STD,
            "clipping": None,
        },
        "ridge": {
            "fit_intercept": True,
            "solver": "lsqr",
            "tol": RIDGE_TOL,
            "max_iter": RIDGE_MAX_ITER,
        },
        "selection": settings["selection"],
        "fusion": "one_convex_oof_weight_per_finger",
        "target_scaling": "fit_rows_for_saved_test_scores",
    }
    if model in FILTERBANK_MODELS:
        result["filterbank_id"] = semantic_hash(
            filterbank_semantics(base, settings)
        )
    elif model == "svr_late":
        result["svr"] = {
            **settings["svr"],
            "kernel": "rbf",
            "gamma": "factor_times_fold_scaled_sklearn_gamma_scale",
            "target": "per_finger_fold_training_standardized",
        }
    else:
        result["neural"] = {
            "grid": settings["cnn_gru"] if model == "cnn_gru_late"
            else settings["mlp"],
            "training": settings["neural_training"],
            "cv_seed_values": CV_SEEDS,
            "checkpoint": "separate_chronological_validation_tail",
            "architecture": (
                {
                    "filters": FILTERS,
                    "filter_length": FILTER_LENGTH,
                    "pool_length": POOL_LENGTH,
                    "pool_stride": POOL_STRIDE,
                    "gru_width": GRU_WIDTH,
                } if model == "cnn_gru_late"
                else "two_relu_hidden_layers_second_half_width"
            ),
        }
    return result


def fit_root(base: dict, model: str, settings: dict) -> Path:
    """Locate one model's stable semantic fit directory."""
    identifier = semantic_hash(fit_semantics(base, model, settings))
    return Path(base["artifact_base"]) / "fits" / f"{model}__{identifier}"
