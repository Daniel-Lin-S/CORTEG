"""Select HGA branch settings with purged CV and resumable units.

Input: (N,C,T) training windows, five finger targets, external
chronological validation windows, and candidate settings. Output:
selected HGA settings, out-of-fold predictions per seed, per-candidate
scores, and final branch models. Completed fold units live under cv/;
selected final checkpoints live in each run directory.
"""
from __future__ import annotations

import itertools
import json
import uuid
from pathlib import Path

import numpy as np
import torch

from evaluation.regression import finger_correlations
from experiments.hga_branch_fit import (
    NEURAL_MODELS, SVR_MODEL, fit_branch,
)
from experiments.ridge_cv import _folds
from models.temporal_spatial_cnn_gru import TemporalSpatialCNNGRU


def candidates(model: str, settings: dict,
               channels: int, timepoints: int) -> tuple[list[dict], list]:
    """Enumerate configured candidates and record invalid CNN geometry."""
    if model == SVR_MODEL:
        grid = settings["svr"]
        return [
            {"C": c, "gamma_factor": gamma, "epsilon": grid["epsilon"]}
            for c, gamma in itertools.product(
                grid["C"], grid["gamma_factors"],
            )
        ], []
    if model == "mlp_late":
        grid = settings["mlp"]
        return [
            {"hidden_size": width, "dropout": dropout}
            for width, dropout in itertools.product(
                grid["hidden_widths"], grid["dropouts"],
            )
        ], []
    if model != "cnn_gru_late":
        raise ValueError(f"Unknown HGA model: {model}.")
    grid = settings["cnn_gru"]
    valid, excluded = [], []
    for depth, stride, dropout in itertools.product(
        grid["depths"], grid["strides"], grid["dropouts"],
    ):
        candidate = {
            "n_convolution_layers": depth,
            "conv_stride": stride,
            "drop_out": dropout,
        }
        try:
            network = TemporalSpatialCNNGRU(
                channels, timepoints, 5, **candidate,
            )
            network.eval()
            with torch.no_grad():
                network(torch.zeros(1, channels, timepoints))
        except (RuntimeError, ValueError) as error:
            excluded.append({
                "candidate": candidate,
                "reason": f"{type(error).__name__}: {error}",
            })
        else:
            valid.append(candidate)
        finally:
            if "network" in locals():
                del network
    if not valid:
        raise ValueError("CNN-GRU grid has no temporally valid candidate.")
    return valid, excluded


def _unit_path(root: Path, candidate: int, seed: int | None,
               fold: int) -> Path:
    """Locate one independently recoverable CV fit."""
    repeat = "deterministic" if seed is None else f"seed{seed}"
    return root / f"candidate{candidate}" / repeat / f"fold{fold}"


def select_branch(
    model: str, train_x: np.ndarray, train_y: np.ndarray,
    validation_x: np.ndarray, validation_y: np.ndarray,
    cv_root: Path, settings: dict, seeds: list[int],
    device: torch.device,
) -> tuple[dict, dict[int | None, np.ndarray], dict]:
    """Select HGA settings by average fold/finger/seed correlation."""
    grid, excluded = candidates(
        model, settings, train_x.shape[1], train_x.shape[2],
    )
    fold_rows = _folds(
        len(train_y), settings["folds"], settings["purge_rows"],
    )
    repeats = seeds if model in NEURAL_MODELS else [None]
    score_grid = np.empty((len(grid), len(repeats), len(fold_rows)))
    predictions = {}
    for candidate_index, candidate in enumerate(grid):
        selected_oof = {}
        for repeat_index, seed in enumerate(repeats):
            oof = np.empty_like(train_y, dtype=np.float32)
            for fold, (training, held_out) in enumerate(fold_rows):
                unit = _unit_path(cv_root, candidate_index, seed, fold)
                marker = unit / "score.json"
                saved_prediction = unit / "prediction.npy"
                if marker.is_file() and saved_prediction.is_file():
                    record = json.loads(marker.read_text())
                    predicted = np.load(saved_prediction)
                    if predicted.shape != train_y[held_out].shape:
                        raise ValueError(
                            f"CV prediction shape differs: {saved_prediction}."
                        )
                    expected = {
                        "candidate": candidate,
                        "seed": seed,
                        "fold": fold,
                        "training_rows": len(training),
                        "held_out_rows": len(held_out),
                    }
                    if any(
                        record.get(key) != value
                        for key, value in expected.items()
                    ):
                        raise ValueError(
                            f"CV unit identity differs: {marker}."
                        )
                    score = float(finger_correlations(
                        train_y[held_out], predicted,
                    ).mean())
                    if not np.isclose(score, record["score"]):
                        raise ValueError(
                            f"CV score differs from predictions: {marker}."
                        )
                else:
                    if unit.exists():
                        quarantine = cv_root / "quarantine"
                        quarantine.mkdir(parents=True, exist_ok=True)
                        unit.rename(
                            quarantine / f"{unit.parent.name}_"
                            f"{unit.name}_{uuid.uuid4().hex}"
                        )
                    predicted, diagnostics = fit_branch(
                        model, candidate, seed,
                        train_x[training], train_y[training],
                        validation_x, validation_y,
                        train_x[held_out], unit, settings, device,
                        persist_model=model != SVR_MODEL,
                    )
                    score = float(finger_correlations(
                        train_y[held_out], predicted,
                    ).mean())
                    np.save(saved_prediction, predicted)
                    if model != SVR_MODEL:
                        (unit / "best.pt").unlink()
                    marker.write_text(json.dumps({
                        "candidate": candidate,
                        "seed": seed, "fold": fold,
                        "training_rows": len(training),
                        "held_out_rows": len(held_out),
                        "score": score,
                        "diagnostics": diagnostics,
                    }, indent=2) + "\n")
                if not np.isfinite(score):
                    raise ValueError(f"Nonfinite CV score: {marker}.")
                score_grid[candidate_index, repeat_index, fold] = score
                oof[held_out] = predicted
            selected_oof[seed] = oof
        predictions[candidate_index] = selected_oof
    mean = score_grid.mean(axis=(1, 2))
    chosen = int(np.argmax(mean))
    history = {
        "candidates": grid,
        "excluded": excluded,
        "scores": score_grid.tolist(),
        "selected_index": chosen,
        "selection": "mean_five_finger_pearson_over_folds_and_seeds",
        "ties": "configured_candidate_order",
    }
    return grid[chosen], predictions[chosen], history
