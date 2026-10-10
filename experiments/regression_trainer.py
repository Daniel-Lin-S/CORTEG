"""Train and evaluate a regression model on supplied data loaders.

Input is a model, fit and validation loaders, a forward callback, optimizer,
scheduler, metric callback, and training settings. Output is an epoch history
JSONL and a validation-selected checkpoint in the supplied fit directory.
The caller owns dataset loading, model construction, and test-set scoring.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

MIN_IMPROVEMENT = 1e-3
GRADIENT_CLIP_NORM = 1.0
MIB = 2 ** 20


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device,
    forward: Callable[[nn.Module, tuple, torch.device], tuple],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect finite targets and predictions from a supplied loader.

    Parameters
    ----------
    model : nn.Module
        Regression model.
    loader : DataLoader
        Nonempty data loader with targets in its batches.
    device : torch.device
        Inference device.
    forward : Callable
        Returns prediction and target tensors for a batch.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Target and prediction arrays, shape (N, outputs).
    """
    model.eval()
    truths, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            output, target = forward(model, batch, device)
            truths.append(target.detach().cpu().numpy())
            predictions.append(output.detach().float().cpu().numpy())
    if not truths:
        raise ValueError("Regression loader has no rows.")
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression predictions contain nonfinite values.")
    return truth, prediction


def group_norms(
    model: nn.Module, names: dict[str, list[str]],
    initial: dict[str, torch.Tensor],
) -> dict:
    """Measure parameter norms and changes for named trainable groups."""
    parameters = dict(model.named_parameters())
    result = {}
    for group, labels in names.items():
        norm = sum(float(parameters[key].detach().float().square().sum())
                   for key in labels) ** 0.5
        change = sum(float((parameters[key].detach().cpu()
                            - initial[key]).float().square().sum())
                     for key in labels) ** 0.5
        result[group] = {"norm": norm, "change": change}
    return result


def train_regressor(
    model: nn.Module, train: DataLoader, validation: DataLoader,
    optimizer: torch.optim.Optimizer, scheduler: object,
    scheduler_metric: str | None, device: torch.device,
    forward: Callable[[nn.Module, tuple, torch.device], tuple],
    metrics: Callable[[np.ndarray, np.ndarray], dict],
    names: dict[str, list[str]], initial: dict[str, torch.Tensor],
    output: Path, epochs: int, patience: int,
) -> tuple[dict, dict]:
    """Save validation-selected checkpoint and return training diagnostics.

    Parameters
    ----------
    model : nn.Module
        Initialized regression model.
    train, validation : DataLoader
        Selected fit and held-out validation loaders.
    optimizer : torch.optim.Optimizer
        Configured optimizer.
    scheduler : object
        Configured scheduler with a ``step`` method.
    scheduler_metric : str or None
        Overall validation metric supplied to a metric-driven scheduler.
    device : torch.device
        Training device.
    forward : Callable
        Returns prediction and target tensors for a batch.
    metrics : Callable
        Returns ``overall.pearson_r`` and other metrics.
    names : dict[str, list[str]]
        Named trainable parameter groups.
    initial : dict[str, torch.Tensor]
        Initial CPU parameter copies.
    output : Path
        Fit directory for history and checkpoint.
    epochs, patience : int
        Maximum epochs and early-stopping patience.

    Returns
    -------
    tuple[dict, dict]
        Loaded best checkpoint and final group norms.
    """
    if epochs < 1 or patience < 1:
        raise ValueError("Epoch count and patience must be positive.")
    best_score = -np.inf
    bad_epochs = 0
    history = output / "history.jsonl"
    with history.open("w") as stream:
        for epoch in range(epochs):
            model.train()
            loss_total = 0.0
            seen = 0
            grad_norms = []
            group_grad_norms = {key: [] for key in names}
            for batch in train:
                prediction, truth = forward(model, batch, device)
                loss = torch.mean((prediction - truth) ** 2)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRADIENT_CLIP_NORM,
                )
                if not torch.isfinite(grad):
                    raise ValueError("Nonfinite gradient norm.")
                grad_norms.append(float(grad))
                parameters = dict(model.named_parameters())
                for key, labels in names.items():
                    value = sum(
                        float(parameters[label].grad.detach().float()
                              .square().sum())
                        for label in labels
                        if parameters[label].grad is not None
                    ) ** 0.5
                    group_grad_norms[key].append(value)
                optimizer.step()
                loss_total += float(loss.detach()) * len(truth)
                seen += len(truth)
            if seen == 0:
                raise ValueError("Training loader has no rows.")
            val_true, val_pred = predict(
                model, validation, device, forward,
            )
            values = metrics(val_true, val_pred)
            val_r = values["overall"]["pearson_r"]
            if scheduler_metric is None:
                scheduler.step()
            else:
                scheduler.step(values["overall"][scheduler_metric])
            record = {
                "epoch": epoch + 1,
                "train_mse": loss_total / seen,
                "validation": values,
                "learning_rates": {
                    group["group_name"]: group["lr"]
                    for group in optimizer.param_groups
                },
                "gradient_norm_mean": float(np.mean(grad_norms)),
                "group_gradient_norm_mean": {
                    key: float(np.mean(values))
                    for key, values in group_grad_norms.items()
                },
                "parameter_groups": group_norms(model, names, initial),
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / MIB
                ) if device.type == "cuda" else 0.0,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if val_r > best_score + MIN_IMPROVEMENT:
                best_score = val_r
                bad_epochs = 0
                torch.save({
                    "model": model.state_dict(), "epoch": epoch + 1,
                    "validation": values,
                }, output / "best.pt")
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break
    checkpoint = torch.load(
        output / "best.pt", map_location=device, weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    return checkpoint, group_norms(model, names, initial)
