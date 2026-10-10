"""Fit one Stanford FingerFlex model on one subject, fraction, and seed.

Input: resolved campaign provenance with checkpoint and HiLo source paths,
plus prepared NumPy feature and target arrays.
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
from experiments.regression_trainer import (
    predict, train_regressor,
)
from experiments.run_regression_hilo_clean import (
    build_model,
    unfreeze_merge_params,
)
from models.steegformer.probe import configure_lora_lastn_probe
from train.lr_schedule import WarmupCosineLR
from models.temporal_spatial_cnn_gru import (
    TemporalSpatialCNNGRU,
)

MIN_STD = 1e-3
CLIP_Z = 5.0
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
        ``corteg``, ``hilo``, or ``cnn_gru``.

    Returns
    -------
    tuple
        Normalized arrays, mean, and std.
    """
    if len(train) == 0:
        raise ValueError("Cannot normalize an empty fit set.")
    axes = (0, 2) if model in ("corteg", "cnn_gru") else (0,)
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


def _hilo_model(channels: int, source: Path) -> nn.Module:
    """Import the configured upstream HiLoFuseNet architecture."""
    spec = importlib.util.spec_from_file_location(
        "fingerflex_hilo_upstream", source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import HiLoFuseNet from {source}.")
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
    adapter_lr = base
    if model_name == "corteg":
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
    include_test: bool = True,
) -> tuple[dict, dict]:
    """Load fit rows and optionally test rows with fit-only normalization."""
    common = root / subject / "common"
    feature_root = root / subject / model / str(delay)
    train_y = np.load(common / "train_y.npy")
    fit, val = split_fit_rows(len(train_y), fraction)
    y_mean = train_y[fit].mean(axis=0, keepdims=True)
    y_std = train_y[fit].std(axis=0, keepdims=True)
    if (y_std <= 0).any():
        raise ValueError("Selected fit targets contain a constant finger.")
    y_sources = [train_y[fit], train_y[val]]
    if include_test:
        test_y = np.load(common / "test_y.npy")
        y_sources.append(test_y)
    y_arrays = tuple(
        ((values - y_mean) / y_std).astype(np.float32)
        for values in y_sources
    )
    if model == "cnn_gru":
        feature_root = root / subject / "corteg" / str(delay)
        train_x = np.load(feature_root / "train_hi.npy", mmap_mode="r"
                          )[..., 0]
        test_file = "test_hi.npy"
    else:
        train_x = np.load(feature_root / "train_x.npy", mmap_mode="r")
        test_file = "test_x.npy"
    if include_test:
        test_x = np.load(feature_root / test_file, mmap_mode="r")
        if model == "cnn_gru":
            test_x = test_x[..., 0]
    else:
        test_x = train_x[val]
    normalized, x_mean, x_std = normalize(
        train_x[fit], train_x[val], test_x, model,
    )
    x_arrays = normalized if include_test else normalized[:2]
    arrays = {"x": x_arrays, "y": y_arrays}
    stats = {
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
    }
    if model == "corteg":
        train_hi = np.load(feature_root / "train_hi.npy", mmap_mode="r")
        high_test = (
            np.load(feature_root / "test_hi.npy", mmap_mode="r")[..., 0]
            if include_test else train_hi[val, ..., 0]
        )
        high, hi_mean, hi_std = normalize(
            train_hi[fit, ..., 0], train_hi[val, ..., 0],
            high_test, model,
        )
        arrays["hi"] = high if include_test else high[:2]
        stats.update({"hi_mean": hi_mean, "hi_std": hi_std})
    np.savez(output / "normalization.npz", **stats)
    indices = np.load(common / "train_indices.npy")
    metadata = {
        "fit_indices_sha256": digest_array(indices[fit]),
        "validation_indices_sha256": digest_array(indices[val]),
        "fit_window_count": len(indices[fit]),
        "validation_window_count": len(indices[val]),
        "fit_first_target": int(indices[fit][0]),
        "fit_last_target": int(indices[fit][-1]),
        "validation_first_target": int(indices[val][0]),
    }
    if include_test:
        test_indices = np.load(common / "test_indices.npy")
        metadata.update({
            "test_grid_sha256": digest_array(test_indices),
            "test_window_count": len(test_indices),
            "test_first_target": int(test_indices[0]),
            "target_values_sha256": digest_array(test_y),
        })
    return arrays, metadata


def _loaders(
    arrays: dict,
    model_name: str,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create deterministic fit, validation, and test data loaders."""
    loaders = []
    for index in range(len(arrays["x"])):
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


def fit_job(
    root: Path,
    config: dict,
    model_name: str,
    subject: str,
    fraction: float,
    seed: int,
    delay: int,
    architecture: dict | None = None,
    validation_only: bool = False,
) -> None:
    """Train one fit and optionally defer test scoring until selection."""
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
    marker = "validation_result.json" if validation_only else "result.json"
    if (output / marker).exists():
        return
    arrays, boundaries = _load_arrays(
        root / "prepared", subject, model_name, delay, fraction, output,
        include_test=not validation_only,
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
        else TemporalSpatialCNNGRU(
            input_channels=arrays["x"][0].shape[1],
            input_length=arrays["x"][0].shape[2],
            outputs=FINGER_COUNT,
            **(architecture or {}),
        ) if model_name == "cnn_gru" else _hilo_model(
            arrays["x"][0].shape[1], Path(config["hilo_source"])
        )
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
    loaders = _loaders(
        arrays, model_name, training["batch_size"], seed,
    )
    train, validation = loaders[:2]
    def forward(
        network: nn.Module, batch: tuple,
        target_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply this dataset's high-gamma and spatial inputs."""
        x = batch[0].to(target_device)
        if model_name == "corteg":
            high = batch[1].to(target_device)
            truth = batch[2].to(target_device)
            prediction = network(
                x, x_hi=high,
                ecog_xyz=xyz.expand(len(x), -1, -1),
            )
        else:
            truth = batch[1].to(target_device)
            prediction = network(x)
        return prediction, truth

    checkpoint, final_norms = train_regressor(
        model, train, validation, optimizer, scheduler,
        "mse" if model_name == "hilo" else None,
        device, forward, metric_values, names, initial, output,
        training["epochs"], training["patience"],
    )
    best_epoch = checkpoint["epoch"]
    history = output / "history.jsonl"
    result = {
        "model": model_name,
        "subject": subject,
        "fraction": fraction,
        "seed": seed,
        "delay_ms": delay,
        "best_epoch": best_epoch,
        "validation_selection_metric": "mean_finger_pearson_r",
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
        "parameter_groups_final": final_norms,
        "history_path": str(history),
        "checkpoint_path": str(output / "best.pt"),
    }
    if validation_only:
        result["architecture"] = architecture
        result["validation"] = checkpoint["validation"]
        (output / marker).write_text(json.dumps(result, indent=2))
        return
    truth, prediction = predict(
        model, loaders[2], device, forward,
    )
    result["metrics"] = metric_values(truth, prediction)
    result["test_scored_after_restore"] = True
    np.savez(output / "test_predictions.npz", y=truth, y_hat=prediction)
    (output / marker).write_text(json.dumps(result, indent=2))



def score_validation_fit(
    root: Path,
    config: dict,
    subject: str,
    fraction: float,
    seed: int,
    architecture: dict,
) -> None:
    """Score a selected CNN-GRU checkpoint once on held-out test rows.

    Parameters
    ----------
    root : Path
        Model-specific fit artifact root.
    config : dict
        Resolved training configuration.
    subject : str
        Subject identifier.
    fraction : float
        Fit-row fraction.
    seed : int
        Training random seed.
    architecture : dict
        Explicit CNN-GRU constructor hyperparameters.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for test scoring.")
    output = (
        root / "runs" / f"{int(fraction * 100)}pct"
        / "cnn_gru" / subject / f"seed{seed}"
    )
    if (output / "result.json").is_file():
        return
    validation_path = output / "validation_result.json"
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Validation fit is missing: {validation_path}."
        )
    result = json.loads(validation_path.read_text())
    if result.get("architecture") != architecture:
        raise ValueError(
            f"Validation architecture differs: {validation_path}."
        )
    common = root / "prepared" / subject / "common"
    train_y = np.load(common / "train_y.npy")
    fit, validation = split_fit_rows(len(train_y), fraction)
    train_indices = np.load(common / "train_indices.npy")
    boundaries = result["boundaries"]
    if boundaries["fit_indices_sha256"] != digest_array(
        train_indices[fit]
    ) or boundaries["validation_indices_sha256"] != digest_array(
        train_indices[validation]
    ):
        raise ValueError("Validation fit rows differ from prepared rows.")
    target = np.load(common / "test_y.npy")
    grid = np.load(common / "test_indices.npy")
    features = np.load(
        root / "prepared" / subject / "corteg" / "0"
        / "test_hi.npy", mmap_mode="r",
    )[..., 0]
    with np.load(output / "normalization.npz") as saved:
        x = ((features - saved["x_mean"]) / saved["x_std"]
             ).astype(np.float32)
        y = ((target - saved["y_mean"]) / saved["y_std"]
             ).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Normalized test features or targets are nonfinite.")
    device = torch.device("cuda:0")
    model = TemporalSpatialCNNGRU(
        input_channels=x.shape[1], input_length=x.shape[2],
        outputs=FINGER_COUNT, **architecture,
    ).to(device)
    checkpoint = torch.load(
        output / "best.pt", map_location=device, weights_only=True,
    )
    if checkpoint["epoch"] != result["best_epoch"]:
        raise ValueError("Best epoch differs from validation result.")
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=config["training"]["batch_size"], shuffle=False,
    )
    truth, prediction = _predict(
        model, loader, "cnn_gru", None, device,
    )
    result["metrics"] = metric_values(truth, prediction)
    result["test_scored_after_restore"] = True
    boundaries.update({
        "test_grid_sha256": digest_array(grid),
        "test_window_count": len(grid),
        "test_first_target": int(grid[0]),
        "target_values_sha256": digest_array(target),
    })
    np.savez(output / "test_predictions.npz", y=truth, y_hat=prediction)
    (output / "result.json").write_text(json.dumps(result, indent=2))
