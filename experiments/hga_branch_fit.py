"""Fit one HGA branch with training-only normalization.

Input: scaled fold or final training, external chronological checkpoint
validation, prediction rows, candidate settings, and an output folder.
Output: branch normalization, fitted checkpoint, epoch history for neural
models, and raw-unit five-finger predictions. CV selection is separate.
"""
from __future__ import annotations

import joblib
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from evaluation.regression import regression_metrics
from experiments.common import set_seed
from experiments.regression_trainer import predict, train_regressor
from models.hga_regressors import FingerMLP, make_svr
from models.temporal_spatial_cnn_gru import TemporalSpatialCNNGRU
from preprocessing.channel_time_features import ChannelTimeScaler
from train.lr_schedule import WarmupCosineLR

NEURAL_MODELS = ("cnn_gru_late", "mlp_late")
SVR_MODEL = "svr_late"


def _loader(x: np.ndarray, y: np.ndarray, batch: int,
            shuffle: bool, seed: int) -> DataLoader:
    """Build a finite float32 regression loader."""
    if len(x) < 1 or len(x) != len(y):
        raise ValueError("Regression loader needs aligned nonempty rows.")
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        ),
        batch_size=batch, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _forward(network: torch.nn.Module, batch: tuple,
             device: torch.device) -> tuple:
    """Map one high-gamma batch to five standardized finger outputs."""
    x, y = batch
    return network(x.to(device)), y.to(device)


def _network(model: str, channels: int, timepoints: int,
             candidate: dict) -> torch.nn.Module:
    """Construct a candidate network from model-only settings."""
    if model == "mlp_late":
        return FingerMLP(
            channels, timepoints,
            candidate["hidden_size"], candidate["dropout"],
        )
    if model == "cnn_gru_late":
        return TemporalSpatialCNNGRU(
            channels, timepoints, 5, **candidate,
        )
    raise ValueError(f"Unknown neural HGA model: {model}.")


def _standardize_target(target: np.ndarray) -> tuple:
    """Compute five training-only target moments."""
    mean = target.mean(axis=0, keepdims=True)
    std = target.std(axis=0, keepdims=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (
        std <= 0
    ).any():
        raise ValueError("Training finger targets are constant/nonfinite.")
    return mean, std


def fit_branch(
    model: str, candidate: dict, seed: int | None,
    train_x: np.ndarray, train_y: np.ndarray,
    validation_x: np.ndarray | None,
    validation_y: np.ndarray | None,
    prediction_x: np.ndarray, output: Path,
    settings: dict, device: torch.device,
    persist_model: bool = True,
) -> tuple[np.ndarray, dict]:
    """Fit one HGA branch using only supplied training rows for scaling.

    Parameters
    ----------
    train_x, validation_x, prediction_x : ndarray, shape (N,C,T)
        Fold training, chronological checkpoint validation, and either
        CV held-out or final test windows.
    train_y, validation_y : ndarray, shape (N,5)
        Corresponding raw finger targets.
    persist_model : bool, optional
        Save the fitted SVR for final inference. Default True. CV units
        need only predictions and scores.

    Returns
    -------
    tuple[ndarray, dict]
        Raw-unit predictions (prediction rows,5) and fit statistics.
    """
    output.mkdir(parents=True, exist_ok=True)
    scaler = ChannelTimeScaler.fit(train_x)
    design = scaler.transform(train_x)
    predict_design = scaler.transform(prediction_x)
    mean, std = _standardize_target(train_y)
    np.savez(
        output / "normalization.npz",
        x_mean=scaler.mean, x_std=scaler.std,
        y_mean=mean, y_std=std,
    )
    if model == SVR_MODEL:
        estimator = make_svr(
            design, candidate["C"], candidate["gamma_factor"],
            candidate["epsilon"], settings["svr_cache_mib"],
        )
        estimator.fit(design, (train_y - mean) / std)
        prediction = estimator.predict(predict_design) * std + mean
        if persist_model:
            joblib.dump(estimator, output / "best.joblib")
        diagnostics = {
            "checkpoint": "best.joblib" if persist_model else None,
        }
    else:
        if seed is None or validation_x is None or validation_y is None:
            raise ValueError("Neural fit needs seed and validation rows.")
        set_seed(seed)
        network = _network(
            model, train_x.shape[1], train_x.shape[2], candidate,
        ).to(device)
        initial = {
            name: value.detach().cpu().clone()
            for name, value in network.named_parameters()
        }
        names = {"hga": list(initial)}
        training = settings["neural_training"]
        optimizer = torch.optim.AdamW(
            [{"params": network.parameters(), "group_name": "hga"}],
            lr=training["base_lr"],
            weight_decay=training["weight_decay"],
        )
        scheduler = WarmupCosineLR(
            optimizer, warmup_epochs=training["warmup_epochs"],
            max_epochs=training["epochs"],
            min_lr=training["min_lr"],
        )
        batch = training["batch_size"]
        x_fit = design.reshape(train_x.shape)
        x_validation = scaler.transform(
            validation_x,
        ).reshape(validation_x.shape)
        train = _loader(
            x_fit, (train_y - mean) / std, batch, True, seed,
        )
        validation = _loader(
            x_validation, (validation_y - mean) / std,
            batch, False, seed,
        )
        checkpoint, norms = train_regressor(
            network, train, validation, optimizer, scheduler,
            None, device, _forward, regression_metrics,
            names, initial, output,
            training["epochs"], training["patience"],
        )
        inference = _loader(
            predict_design.reshape(prediction_x.shape),
            np.zeros((len(prediction_x), 5), dtype=np.float32),
            batch, False, seed,
        )
        _, standardized = predict(
            network, inference, device, _forward,
        )
        prediction = standardized * std + mean
        diagnostics = {
            "best_epoch": checkpoint["epoch"],
            "validation": checkpoint["validation"],
            "parameter_groups_final": norms,
        }
    if not np.isfinite(prediction).all():
        raise ValueError("High-gamma branch predictions are nonfinite.")
    return prediction.astype(np.float32), diagnostics
