"""Fit one Stanford FingerFlex model on one subject, fraction, and seed.

Input: campaign config/provenance and prepared NumPy feature/target arrays.
Output: ``runs/<fraction>/<model>/<subject>/seed<seed>/`` with epoch
history JSONL, normalization statistics NPZ, validation-selected checkpoint,
test predictions NPZ, and result JSON. Test evaluation occurs only after
restoring the validation-selected state. No raw dataset files are written.
"""
from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.fingerflex_campaign import digest_array
from experiments.common import set_seed
from experiments.run_regression_hilo_clean import (
    build_model,
    unfreeze_merge_params,
)
from models.steegformer.probe import configure_lora_lastn_probe
from train.lr_schedule import WarmupCosineLR

MIN_STD = 1e-3
CLIP_Z = 5.0
HILO_SOURCE = Path(
    "/home/lyh/codes/HiLoFuseNet/finger_regression/models/nn_regressors.py"
)
FINGER_COUNT = 5


def split_fit_rows(n_train: int, fraction: float) -> tuple[slice, slice]:
    """Split chronological training windows into fit prefix and val tail.

    Parameters
    ----------
    n_train : int
        Number of training-segment windows.
    fraction : float
        Fraction of the non-validation windows to fit.

    Returns
    -------
    tuple[slice, slice]
        Fit and validation row selections.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"Invalid fit fraction {fraction}.")
    val_count = int(np.floor(n_train * 0.1))
    available = n_train - val_count
    selected = int(np.floor(available * fraction))
    if selected < 2 or val_count < 2:
        raise ValueError(
            f"Insufficient windows: fit={selected}, val={val_count}."
        )
    return slice(0, selected), slice(available, n_train)


def normalize(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    model: str,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Fit feature statistics on selected fit rows only.

    Parameters
    ----------
    train, val, test : np.ndarray
        Feature arrays with sample axis first.
    model : str
        ``corteg`` or ``hilo``.

    Returns
    -------
    tuple
        Normalized arrays, mean, and std.
    """
    if len(train) == 0:
        raise ValueError("Cannot normalize an empty fit set.")
    axes = (0, 2) if model == "corteg" else (0,)
    mean = train.mean(axis=axes, keepdims=True)
    std = train.std(axis=axes, keepdims=True)
    std = np.maximum(std, MIN_STD if model == "hilo" else 1e-6)
    arrays = tuple(((x - mean) / std).astype(np.float32)
                   for x in (train, val, test))
    if model == "hilo":
        arrays = tuple(np.clip(x, -CLIP_Z, CLIP_Z) for x in arrays)
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("Feature normalization produced nonfinite values.")
    return arrays, mean, std


def metric_values(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Compute five-finger and overall r, MSE, and R² on one scale.

    Parameters
    ----------
    truth, pred : np.ndarray
        Standardized trajectories, shape (N, 5).

    Returns
    -------
    dict
        Per-finger arrays and unweighted five-finger means.
    """
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError(
            f"Expected equal 2D arrays, got {truth.shape}, {pred.shape}."
        )
    if truth.shape[1] != FINGER_COUNT or len(truth) < 2:
        raise ValueError(f"Expected (N>=2, 5), got {truth.shape}.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Prediction or target contains nonfinite values.")
    centered = truth - truth.mean(axis=0)
    predicted = pred - pred.mean(axis=0)
    target_ss = np.sum(centered ** 2, axis=0)
    pred_ss = np.sum(predicted ** 2, axis=0)
    if (target_ss <= 0).any():
        raise ValueError("Constant test target makes R² undefined.")
    if (pred_ss <= 0).any():
        raise ValueError("Constant predictions make Pearson r undefined.")
    corr = np.sum(centered * predicted, axis=0) / np.sqrt(
        target_ss * pred_ss
    )
    mse = np.mean((truth - pred) ** 2, axis=0)
    r2 = 1 - np.sum((truth - pred) ** 2, axis=0) / target_ss
    values = {"pearson_r": corr, "mse": mse, "r2": r2}
    if any(not np.isfinite(x).all() for x in values.values()):
        raise ValueError("Nonfinite regression metric.")
    return {
        "overall": {k: float(v.mean()) for k, v in values.items()},
        "fingers": {k: v.tolist() for k, v in values.items()},
    }


def _hilo_model(channels: int) -> nn.Module:
    """Import the pinned local upstream HiLoFuseNet architecture."""
    spec = importlib.util.spec_from_file_location(
        "fingerflex_hilo_upstream", HILO_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import HiLoFuseNet from {HILO_SOURCE}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HiLoFuseNet(
        channels, 2, lstm_hidden=256, D=20,
        output_size=FINGER_COUNT, dropout_prob=0.5,
    )


def _corteg_model(
    channels: int,
    xyz_mm: np.ndarray,
    checkpoint: Path,
    training: dict,
) -> nn.Module:
    """Build pretrained small CORTEG with requested trainable groups."""
    model_kwargs = {
        "pretrained": {
            "path": str(checkpoint),
            "ckpt_key": "model",
            "strict": False,
            "strip_prefix": "",
        },
        "backbone_kwargs": {
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    }
    args = Namespace(
        model_kwargs_json=json.dumps(model_kwargs),
        steegformer_variant="small",
        hi_patch_size=25,
        hi_inject_last_n=4,
        merge_strategy="average",
        channel_adapter="knn_soft_fourier",
        knn_k=8,
        adapter_branch="both",
        head_dropout=0.0,
        stream="both",
        no_pretrained=False,
        freeze_readout=False,
    )
    model = build_model(args, channels, 128, xyz_mm, FINGER_COUNT)
    configure_lora_lastn_probe(
        model,
        n_last=training["lora_last_n"],
        r=training["lora_rank"],
        alpha=training["lora_alpha"],
        dropout=training["lora_dropout"],
        targets=("qkv", "proj", "fc1", "fc2"),
    )
    unfreeze_merge_params(model)
    return model


def parameter_groups(
    model: nn.Module,
    model_name: str,
    fraction: float,
    config: dict,
) -> tuple[list[dict], dict]:
    """Assign adapter-only LR boost while keeping LoRA at base LR."""
    base = config["base_lr"]
    adapter_lr = (
        config["adapter_lr_low"] if fraction == 0.1
        else config["adapter_lr_high"]
    )
    grouped: dict[str, list[nn.Parameter]] = {}
    names: dict[str, list[str]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = (
            "adapter" if model_name == "corteg"
            and "channel_adapter" in name else "base"
        )
        grouped.setdefault(group, []).append(param)
        names.setdefault(group, []).append(name)
    if "base" not in grouped or (
        model_name == "corteg" and "adapter" not in grouped
    ):
        raise ValueError("Expected base and CORTEG adapter trainables.")
    groups = [
        {"params": params, "lr": adapter_lr if key == "adapter" else base,
         "group_name": key}
        for key, params in grouped.items()
    ]
    return groups, names


def _load_arrays(
    root: Path,
    subject: str,
    model: str,
    delay: int,
    fraction: float,
    output: Path,
) -> tuple[dict, dict]:
    """Load common rows and normalize features using selected fit rows."""
    common = root / subject / "common"
    feature_root = root / subject / model / str(delay)
    train_y = np.load(common / "train_y.npy")
    test_y = np.load(common / "test_y.npy")
    fit, val = split_fit_rows(len(train_y), fraction)
    y_mean = train_y[fit].mean(axis=0, keepdims=True)
    y_std = train_y[fit].std(axis=0, keepdims=True)
    if (y_std <= 0).any():
        raise ValueError("Selected fit targets contain a constant finger.")
    y_arrays = tuple(((x - y_mean) / y_std).astype(np.float32)
                     for x in (train_y[fit], train_y[val], test_y))
    train_x = np.load(feature_root / "train_x.npy", mmap_mode="r")
    test_x = np.load(feature_root / "test_x.npy", mmap_mode="r")
    x_arrays, x_mean, x_std = normalize(
        train_x[fit], train_x[val], test_x, model,
    )
    arrays = {"x": x_arrays, "y": y_arrays}
    stats = {"x_mean": x_mean, "x_std": x_std,
             "y_mean": y_mean, "y_std": y_std}
    if model == "corteg":
        train_hi = np.load(feature_root / "train_hi.npy", mmap_mode="r")
        test_hi = np.load(feature_root / "test_hi.npy", mmap_mode="r")
        hi_arrays, hi_mean, hi_std = normalize(
            train_hi[fit, ..., 0], train_hi[val, ..., 0],
            test_hi[..., 0], model,
        )
        arrays["hi"] = hi_arrays
        stats.update({"hi_mean": hi_mean, "hi_std": hi_std})
    np.savez(output / "normalization.npz", **stats)
    indices = np.load(common / "train_indices.npy")
    test_indices = np.load(common / "test_indices.npy")
    metadata = {
        "fit_indices_sha256": digest_array(indices[fit]),
        "validation_indices_sha256": digest_array(indices[val]),
        "test_grid_sha256": digest_array(test_indices),
        "fit_window_count": len(indices[fit]),
        "validation_window_count": len(indices[val]),
        "test_window_count": len(test_indices),
        "fit_first_target": int(indices[fit][0]),
        "fit_last_target": int(indices[fit][-1]),
        "validation_first_target": int(indices[val][0]),
        "test_first_target": int(test_indices[0]),
        "target_values_sha256": digest_array(test_y),
    }
    return arrays, metadata


def _loaders(
    arrays: dict,
    model_name: str,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create deterministic fit, validation, and test data loaders."""
    loaders = []
    for index in range(3):
        tensors = [torch.from_numpy(arrays["x"][index])]
        if model_name == "corteg":
            tensors.append(torch.from_numpy(arrays["hi"][index]))
        tensors.append(torch.from_numpy(arrays["y"][index]))
        loaders.append(DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=index == 0,
            generator=torch.Generator().manual_seed(seed),
        ))
    return tuple(loaders)


def _predict(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
    xyz: torch.Tensor | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict standardized finger targets for one data loader."""
    model.eval()
    predicted = []
    target = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if model_name == "corteg":
                high = batch[1].to(device)
                truth = batch[2]
                output = model(
                    x, x_hi=high,
                    ecog_xyz=xyz.expand(len(x), -1, -1),
                )
            else:
                truth = batch[1]
                output = model(x)
            predicted.append(output.float().cpu().numpy())
            target.append(truth.numpy())
    return np.concatenate(target), np.concatenate(predicted)


def _group_norms(
    model: nn.Module,
    names: dict,
    initial: dict[str, torch.Tensor],
) -> dict:
    """Return L2 parameter norms and changes since initialization."""
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


def fit_job(
    root: Path,
    config: dict,
    model_name: str,
    subject: str,
    fraction: float,
    seed: int,
    delay: int,
) -> None:
    """Run one CUDA fit and save validation-selected test result."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fingerflex campaign.")
    set_seed(seed)
    device = torch.device("cuda:0")
    fraction_name = f"{int(fraction * 100)}pct"
    if delay:
        fraction_name = f"{fraction_name}_delay{delay}ms"
    output = (
        root / "runs" / fraction_name / model_name / subject / f"seed{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        return
    arrays, boundaries = _load_arrays(
        root / "prepared", subject, model_name, delay, fraction, output,
    )
    common = root / "prepared" / subject / "common"
    xyz_np = np.load(common / "xyz_mm.npy")
    xyz = torch.tensor(xyz_np / 1000, device=device)[None] if (
        model_name == "corteg"
    ) else None
    training = config["training"]
    model = (
        _corteg_model(
            arrays["x"][0].shape[1], xyz_np,
            Path(config["checkpoint"]), training,
        ) if model_name == "corteg"
        else _hilo_model(arrays["x"][0].shape[1])
    ).to(device)
    groups, names = parameter_groups(model, model_name, fraction, training)
    initial = {name: param.detach().cpu().clone()
               for name, param in model.named_parameters()
               if param.requires_grad}
    optimizer = (
        torch.optim.Adam(groups, lr=training["base_lr"])
        if model_name == "hilo"
        else torch.optim.AdamW(
            groups, lr=training["base_lr"],
            weight_decay=training["weight_decay"],
        )
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5,
        ) if model_name == "hilo"
        else WarmupCosineLR(
            optimizer, warmup_epochs=training["warmup_epochs"],
            max_epochs=training["epochs"],
            min_lr=training["min_lr"],
        )
    )
    train, validation, test = _loaders(
        arrays, model_name, training["batch_size"], seed,
    )
    best_score = -np.inf
    best_epoch = -1
    bad_epochs = 0
    history = output / "history.jsonl"
    with history.open("w") as stream:
        for epoch in range(training["epochs"]):
            model.train()
            loss_total = 0.0
            seen = 0
            grad_norms = []
            group_grad_norms = {key: [] for key in names}
            for batch in train:
                x = batch[0].to(device)
                if model_name == "corteg":
                    high = batch[1].to(device)
                    truth = batch[2].to(device)
                    prediction = model(
                        x, x_hi=high,
                        ecog_xyz=xyz.expand(len(x), -1, -1),
                    )
                else:
                    truth = batch[1].to(device)
                    prediction = model(x)
                loss = torch.mean((prediction - truth) ** 2)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0,
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
                loss_total += float(loss.detach()) * len(x)
                seen += len(x)
            val_true, val_pred = _predict(
                model, validation, model_name, xyz, device,
            )
            metrics = metric_values(val_true, val_pred)
            val_r = metrics["overall"]["pearson_r"]
            val_mse = metrics["overall"]["mse"]
            if model_name == "hilo":
                scheduler.step(val_mse)
            else:
                scheduler.step()
            record = {
                "epoch": epoch + 1,
                "train_mse": loss_total / seen,
                "validation": metrics,
                "learning_rates": {
                    group["group_name"]: group["lr"]
                    for group in optimizer.param_groups
                },
                "gradient_norm_mean": float(np.mean(grad_norms)),
                "group_gradient_norm_mean": {
                    key: float(np.mean(values))
                    for key, values in group_grad_norms.items()
                },
                "parameter_groups": _group_norms(model, names, initial),
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / 2 ** 20
                ),
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if val_r > best_score + 1e-3:
                best_score = val_r
                best_epoch = epoch + 1
                bad_epochs = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": best_epoch,
                     "validation": metrics},
                    output / "best.pt",
                )
            else:
                bad_epochs += 1
            if bad_epochs >= training["patience"]:
                break
    checkpoint = torch.load(
        output / "best.pt", map_location=device, weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    truth, prediction = _predict(model, test, model_name, xyz, device)
    metrics = metric_values(truth, prediction)
    np.savez(output / "test_predictions.npz", y=truth, y_hat=prediction)
    result = {
        "model": model_name,
        "subject": subject,
        "fraction": fraction,
        "seed": seed,
        "delay_ms": delay,
        "best_epoch": best_epoch,
        "validation_selection_metric": "mean_finger_pearson_r",
        "metrics": metrics,
        "boundaries": boundaries,
        "learning_rate_groups": {
            group["group_name"]: (
                training["adapter_lr_low"] if fraction == 0.1
                else training["adapter_lr_high"]
            ) if group["group_name"] == "adapter"
            else training["base_lr"]
            for group in optimizer.param_groups
        },
        "trainable_parameter_names": names,
        "parameter_groups_final": _group_norms(model, names, initial),
        "test_scored_after_restore": True,
        "history_path": str(history),
        "checkpoint_path": str(output / "best.pt"),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
