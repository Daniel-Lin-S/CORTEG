"""Audit one completed Stanford FingerFlex subject fit.

Input: a campaign artifact root and one structured ``runs/`` result folder.
Output: ``reports/first_fit_audit.json`` with independently checked target
boundaries, metrics, checkpoint selection, learning rates, and parameter
changes. Raises on any mismatch or nonfinite quantity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_campaign import digest_array
from experiments.fingerflex_campaign import metric_values
from scripts.fingerflex_campaign import job_path, sha256_file


def audit(root: Path, job: dict) -> dict:
    """Verify saved test metrics and validation checkpoint independently.

    Parameters
    ----------
    root : Path
        Campaign artifact root.
    job : dict
        Subject, fraction, model, seed, and delay identifiers.

    Returns
    -------
    dict
        Verified evidence and artifact hashes.
    """
    folder = job_path(root, job)
    result = json.loads((folder / "result.json").read_text())
    history = [json.loads(line) for line in
               (folder / "history.jsonl").read_text().splitlines()]
    if not history:
        raise ValueError(f"Training history is empty: {folder}.")
    checkpoint = torch.load(
        folder / "best.pt", map_location="cpu", weights_only=True,
    )
    if checkpoint["epoch"] != result["best_epoch"]:
        raise ValueError("Saved best epoch disagrees with result JSON.")
    selected = history[result["best_epoch"] - 1]
    if selected["validation"] != checkpoint["validation"]:
        raise ValueError("Checkpoint validation metrics disagree with history.")
    best = -np.inf
    selected_epoch = -1
    for record in history:
        score = record["validation"]["overall"]["pearson_r"]
        if score > best + 1e-3:
            best = score
            selected_epoch = record["epoch"]
    if selected_epoch != result["best_epoch"]:
        raise ValueError("Checkpoint does not match validation selection.")
    common = root / "prepared" / job["subject"] / "common"
    train_indices = np.load(common / "train_indices.npy")
    test_indices = np.load(common / "test_indices.npy")
    test_raw = np.load(common / "test_y.npy")
    boundary = result["boundaries"]
    if digest_array(test_indices) != boundary["test_grid_sha256"]:
        raise ValueError("Test target-grid hash mismatch.")
    if digest_array(test_raw) != boundary["target_values_sha256"]:
        raise ValueError("Test target-value hash mismatch.")
    if not (
        boundary["fit_last_target"]
        < boundary["validation_first_target"]
        < boundary["test_first_target"]
    ):
        raise ValueError("Fit/validation/test target rows overlap.")
    if boundary["test_first_target"] != int(test_indices[0]):
        raise ValueError("Reported test start differs from saved target grid.")
    if boundary["fit_first_target"] != int(train_indices[0]):
        raise ValueError("Reported fit start differs from saved target grid.")
    normalization = np.load(folder / "normalization.npz")
    expected_y = (
        test_raw - normalization["y_mean"]
    ) / normalization["y_std"]
    predictions = np.load(folder / "test_predictions.npz")
    np.testing.assert_allclose(
        predictions["y"], expected_y, rtol=1e-5, atol=1e-5,
    )
    calculated = metric_values(predictions["y"], predictions["y_hat"])
    for metric, value in calculated["overall"].items():
        if not np.isclose(value, result["metrics"]["overall"][metric]):
            raise ValueError(f"Saved {metric} differs from predictions.")
    initial_lr = result["learning_rate_groups"]
    if job["model"] == "corteg":
        expected_adapter = 0.002 if job["fraction"] == 0.1 else 0.01
        if initial_lr != {"base": 0.001, "adapter": expected_adapter}:
            raise ValueError("CORTEG optimizer learning rates are wrong.")
        changes = result["parameter_groups_final"]
        if any(changes[key]["change"] <= 0
               for key in ("base", "adapter")):
            raise ValueError("A CORTEG trainable group did not change.")
    for record in history:
        numbers = [
            record["train_mse"], record["gradient_norm_mean"],
            record["cuda_peak_allocated_mib"],
            *record["validation"]["overall"].values(),
        ]
        if not np.isfinite(numbers).all():
            raise ValueError("Training history has nonfinite monitors.")
    if not result["test_scored_after_restore"]:
        raise ValueError("Result lacks restored-checkpoint test marker.")
    output = {
        "job": job,
        "epochs": len(history),
        "best_epoch": result["best_epoch"],
        "fit_window_count": boundary["fit_window_count"],
        "validation_window_count": boundary["validation_window_count"],
        "test_window_count": boundary["test_window_count"],
        "test_grid_sha256": boundary["test_grid_sha256"],
        "metrics": calculated,
        "learning_rate_groups": initial_lr,
        "parameter_groups_final": result["parameter_groups_final"],
        "checkpoint_sha256": sha256_file(folder / "best.pt"),
        "history_sha256": sha256_file(folder / "history.jsonl"),
        "validated": True,
    }
    return output


def main() -> None:
    """Audit one supplied job and save its report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fraction", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--delay", type=int, default=0)
    args = parser.parse_args()
    job = {
        "subject": args.subject, "model": args.model,
        "fraction": args.fraction, "seed": args.seed,
        "delay": args.delay,
    }
    result = audit(args.root, job)
    path = args.root / "reports" / "first_fit_audit.json"
    path.write_text(json.dumps(result, indent=2))
    print(path)


if __name__ == "__main__":
    main()
