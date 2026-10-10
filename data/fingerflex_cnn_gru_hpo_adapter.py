"""Adapt FingerFlex CNN-GRU fits to the generic HPO coordinator.

Input is a portable HPO JSON and resolved local campaign settings. Output is
searches/<name>__<id>/ with index.json, selection.json, provenance/, status/;
candidate fits remain in fits/cnn_gru_hga__<fit-id>/ with validation_result.json
and best.pt per subject. Only the selected fit obtains result.json and
test_predictions.npz. The comparison stays in comparisons/<name>__<id>/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_cache import (
    ensure_cache, preparation_id, prepare_cached_subject,
    sha256_file,
)
from data.fingerflex_campaign import digest_array
from experiments.hpo import (
    dispatch_jobs, enumerate_grid, pending_jobs as generic_pending_jobs,
    select_candidate,
)
from experiments.fingerflex_campaign import (
    _predict, fit_job, metric_values, score_validation_fit,
    split_fit_rows,
)
from models.temporal_spatial_cnn_gru import TemporalSpatialCNNGRU
from scripts.fingerflex_comparison import (
    ensure_comparison, find_fit_roots, load_comparison_study, report,
)
from scripts.fingerflex_fit_campaigns import _digest, load_study

DEFAULT_ARCHITECTURE = {
    "n_convolution_layers": 3,
    "conv_stride": 3,
    "drop_out": 0.3,
}
SOURCE_FILES = (
    "models/temporal_spatial_cnn_gru.py",
    "experiments/fingerflex_campaign.py",
    "experiments/regression_trainer.py",
    "train/lr_schedule.py",
)
METRICS = ("pearson_r", "mse", "r2")


def architecture_grid(spec: dict) -> list[dict]:
    """Enumerate and validate unique architecture configurations."""
    architectures = enumerate_grid(spec["grid"])
    for architecture in architectures:
        model = TemporalSpatialCNNGRU(
            input_channels=1, input_length=200, outputs=5,
            **architecture,
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 200))
        if output.shape != (1, 5) or not torch.isfinite(output).all():
            raise ValueError(
                f"Invalid CNN-GRU output for {architecture}."
            )
    return architectures


def architecture_key(architecture: dict) -> str:
    """Name one grid point without embedding machine paths."""
    return (
        f"depth{architecture['n_convolution_layers']}_"
        f"stride{architecture['conv_stride']}_"
        f"drop{architecture['drop_out']:.1f}"
    )


def load_spec(path: Path) -> tuple[dict, dict, dict, dict]:
    """Resolve portable HPO settings and prior model fit roots."""
    spec = json.loads(path.read_text())
    architectures = architecture_grid(spec)
    fit_study, base = load_study(REPO / spec["source_fits"])
    reference, reference_base = load_comparison_study(
        REPO / spec["reference_comparison"]
    )
    if preparation_id(base) != preparation_id(reference_base):
        raise ValueError("Reference fits have another preparation ID.")
    if set(spec["subjects"]) != set(base["subjects"]) or (
        len(spec["subjects"]) != len(base["subjects"])
    ):
        raise ValueError("HPO must cover all source-campaign subjects.")
    selection = spec["selection"]
    if selection != {
        "fraction": 1.0, "seed": 0,
        "metric": "mean_finger_pearson_r",
        "aggregation": "equal_subject_mean",
        "tie_break": "ascending_architecture",
    }:
        raise ValueError("Unexpected HPO selection policy.")
    if spec["final"] != {
        "fractions": [1.0, 0.25, 0.1], "seeds": [0, 1, 2],
    }:
        raise ValueError("Unexpected final fit selectors.")
    runtime = spec["runtime"]
    if runtime["gpu_workers"] != 4 or (
        type(runtime["minimum_gpu_free_mib"]) is not int
        or runtime["minimum_gpu_free_mib"] < 1
    ):
        raise ValueError("Invalid GPU runtime capacity settings.")
    roots = find_fit_roots(reference, reference_base)
    return spec, fit_study, base, roots


def candidate_semantics(
    architecture: dict, training: dict, base: dict,
) -> dict:
    """Identify one model fit without search or GPU selectors."""
    return {
        "model": "cnn_gru",
        "architecture": architecture,
        "training": training,
        "preprocessing_id": preparation_id(base),
        "delay_ms": 0,
        "feature": "70-200Hz_Hilbert_amplitude",
        "source_sha256": {
            name: sha256_file(REPO / name)
            for name in SOURCE_FILES
        },
    }


def ensure_search(
    spec: dict, fit_study: dict, base: dict,
    reference_roots: dict,
) -> Path:
    """Create a resumable search and its model-specific fit roots."""
    prepared = ensure_cache(base)
    for subject in spec["subjects"]:
        prepare_cached_subject(subject, base, (0,))
    candidates = {}
    training = fit_study["cnn_gru"]
    for architecture in architecture_grid(spec):
        key = architecture_key(architecture)
        if architecture == DEFAULT_ARCHITECTURE:
            candidates[key] = str(reference_roots["cnn_gru"])
            continue
        semantics = candidate_semantics(architecture, training, base)
        fit_id = _digest(semantics)
        fit_root = (
            Path(base["artifact_base"]) / "fits"
            / f"cnn_gru_hga__{fit_id}"
        )
        for directory in ("runs", "status", "provenance"):
            (fit_root / directory).mkdir(parents=True, exist_ok=True)
        fit_index = {
            "model": "cnn_gru", "semantic_id": fit_id,
            "preprocessing_id": preparation_id(base),
            "semantics": semantics,
        }
        fit_index_path = fit_root / "index.json"
        if fit_index_path.exists():
            if json.loads(fit_index_path.read_text()) != fit_index:
                raise ValueError(
                    f"Candidate identity conflict: {fit_index_path}."
                )
        else:
            fit_index_path.write_text(
                json.dumps(fit_index, indent=2) + "\n"
            )
        resolved = dict(base)
        resolved["training"] = training
        (fit_root / "provenance" / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n"
        )
        for subject in spec["subjects"]:
            target = fit_root / "prepared" / subject
            source = prepared / subject
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError(f"Prepared link differs: {target}.")
            elif target.exists():
                raise ValueError(f"Prepared path is not a link: {target}.")
            else:
                target.symlink_to(source)
        candidates[key] = str(fit_root)
    semantics = {
        "version": "cnn_gru_validation_grid_v1",
        "candidates": {
            key: json.loads(
                (Path(path) / "index.json").read_text()
            )["semantic_id"] for key, path in candidates.items()
        },
        "subjects": spec["subjects"],
        "selection": spec["selection"],
        "training": training,
        "preprocessing_id": preparation_id(base),
    }
    search_id = _digest(semantics)
    search = (
        Path(base["artifact_base"]) / "searches"
        / f"{spec['name']}__{search_id}"
    )
    for directory in ("status", "provenance"):
        (search / directory).mkdir(parents=True, exist_ok=True)
    index = {
        "name": spec["name"], "semantic_id": search_id,
        "semantics": semantics, "candidates": candidates,
        "architectures": {
            architecture_key(value): value
            for value in architecture_grid(spec)
        },
        "reference_roots": {
            key: str(path) for key, path in reference_roots.items()
        },
    }
    path = search / "index.json"
    if path.exists() and json.loads(path.read_text()) != index:
        raise ValueError(f"Search identity conflict: {path}.")
    path.write_text(json.dumps(index, indent=2) + "\n")
    (search / "provenance" / "resolved_search.json").write_text(
        json.dumps(spec, indent=2) + "\n"
    )
    return search


def fit_path(root: Path, fraction: float, subject: str,
             seed: int) -> Path:
    """Locate a CNN-GRU subject fit directory."""
    return (
        root / "runs" / f"{int(fraction * 100)}pct"
        / "cnn_gru" / subject / f"seed{seed}"
    )


def pending_jobs(search: Path, spec: dict, mode: str) -> list[dict]:
    """Map search completion state to generic HPO job enumeration."""
    index = json.loads((search / "index.json").read_text())
    selected = None
    if mode == "final":
        selected = json.loads(
            (search / "selection.json").read_text()
        )["candidate"]
    elif mode != "screen":
        raise ValueError(f"Unknown job phase {mode}.")
    candidates = {
        key: {
            "parameters": index["architectures"][key],
            "fit_root": path,
            "historical": (
                index["architectures"][key] == DEFAULT_ARCHITECTURE
            ),
        } for key, path in index["candidates"].items()
    }

    def complete(job: dict) -> bool:
        root = Path(index["candidates"][job["candidate"]])
        output = fit_path(
            root, job["fraction"], job["unit"], job["seed"],
        )
        marker = (
            "validation_result.json" if job["mode"] == "screen"
            else "result.json"
        )
        return (output / marker).is_file()

    jobs = generic_pending_jobs(
        candidates, spec["subjects"], spec.get("final", {}),
        selected, spec.get("selection", {
            "fraction": 1.0, "seed": 0,
        }), complete,
    )
    return [
        {**{key: value for key, value in job.items() if key != "unit"},
         "subject": job["unit"]}
        for job in jobs
    ]


def _quarantine(output: Path, search: Path, marker: str) -> None:
    """Move an interrupted fit aside before a fresh attempt."""
    if not output.exists() or (output / marker).is_file():
        return
    destination = (
        search / "status" / "quarantine"
        / f"{output.parent.name}_{output.name}_{time.time_ns()}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.rename(destination)


def dispatch(search: Path, spec: dict, phase: str,
             gpus: list[int]) -> None:
    """Run task-specific fits through the dataset-agnostic scheduler."""
    pending = pending_jobs(search, spec, phase)
    index = json.loads((search / "index.json").read_text())

    def execute(job: dict, gpu: int) -> tuple[str, int]:
        root = Path(index["candidates"][job["candidate"]])
        output = fit_path(
            root, job["fraction"], job["subject"], job["seed"],
        )
        marker = (
            "validation_result.json" if job["mode"] == "screen"
            else "result.json"
        )
        if job["mode"] != "score":
            _quarantine(output, search, marker)
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(REPO / "scripts" / "hpo_campaign.py"),
            "fit", "--search-root", str(search),
            "--candidate", job["candidate"],
            "--subject", job["subject"],
            "--fraction", str(job["fraction"]),
            "--seed", str(job["seed"]),
            "--mode", job["mode"],
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with (output / "worker.log").open("a") as stream:
            completed = subprocess.run(
                command, cwd=REPO, env=env, stdout=stream,
                stderr=subprocess.STDOUT, check=False,
            )
        state = (
            "complete" if completed.returncode == 0
            and (output / marker).is_file() else "failed"
        )
        return state, completed.returncode

    dispatch_jobs(search, pending, gpus, execute)
    remaining = pending_jobs(search, spec, phase)
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} {phase} jobs remain; inspect status/jobs.jsonl."
        )


def _baseline_validation(
    root: Path, subject: str,
) -> float:
    """Verify old default checkpoint against current validation rows."""
    output = fit_path(root, 1.0, subject, 0)
    checkpoint = torch.load(
        output / "best.pt", map_location="cpu", weights_only=True,
    )
    common = root / "prepared" / subject / "common"
    train_y = np.load(common / "train_y.npy")
    _, validation = split_fit_rows(len(train_y), 1.0)
    high = np.load(
        root / "prepared" / subject / "corteg" / "0"
        / "train_hi.npy", mmap_mode="r",
    )[validation, ..., 0]
    with np.load(output / "normalization.npz") as saved:
        x = ((high - saved["x_mean"]) / saved["x_std"]
             ).astype(np.float32)
        y = ((train_y[validation] - saved["y_mean"])
             / saved["y_std"]).astype(np.float32)
    model = TemporalSpatialCNNGRU(
        input_channels=x.shape[1], input_length=x.shape[2],
        outputs=5, **DEFAULT_ARCHITECTURE,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=128, shuffle=False,
    )
    truth, prediction = _predict(
        model, loader, "cnn_gru", None, torch.device("cpu"),
    )
    actual = metric_values(truth, prediction)["overall"]
    expected = checkpoint["validation"]["overall"]
    if any(not np.isclose(
        actual[key], expected[key], rtol=1e-3, atol=1e-3,
    ) for key in METRICS):
        raise ValueError(
            f"Default model differs from saved validation: {output}."
        )
    return float(expected["pearson_r"])


def selection_scores(search: Path, spec: dict) -> dict[str, float]:
    """Collect only best-validation Pearson scores for every grid point."""
    index = json.loads((search / "index.json").read_text())
    scores = {}
    for key, path in index["candidates"].items():
        values = []
        for subject in spec["subjects"]:
            root = Path(path)
            if index["architectures"][key] == DEFAULT_ARCHITECTURE:
                value = _baseline_validation(root, subject)
            else:
                output = fit_path(root, 1.0, subject, 0)
                record = json.loads(
                    (output / "validation_result.json").read_text()
                )
                train_indices = np.load(
                    root / "prepared" / subject / "common"
                    / "train_indices.npy"
                )
                fit, validation = split_fit_rows(
                    len(train_indices), 1.0,
                )
                expected_rows = {
                    "fit_indices_sha256": digest_array(
                        train_indices[fit]
                    ),
                    "validation_indices_sha256": digest_array(
                        train_indices[validation]
                    ),
                }
                if any(
                    record["boundaries"].get(name) != digest
                    for name, digest in expected_rows.items()
                ):
                    raise ValueError(
                        f"Candidate rows differ from preparation: {output}."
                    )
                checkpoint = torch.load(
                    output / "best.pt", map_location="cpu",
                    weights_only=True,
                )
                value = float(
                    checkpoint["validation"]["overall"]["pearson_r"]
                )
                if record["best_epoch"] != checkpoint["epoch"] or (
                    not np.isclose(
                        value,
                        record["validation"]["overall"]["pearson_r"],
                    )
                ):
                    raise ValueError(
                        f"Candidate validation mismatch: {output}."
                    )
            if not np.isfinite(value):
                raise ValueError(
                    f"Nonfinite validation Pearson for {key}/{subject}."
                )
            values.append(value)
        scores[key] = float(np.mean(values))
    return scores


def select(search: Path, spec: dict) -> dict:
    """Freeze one global winner before any new test scoring."""
    path = search / "selection.json"
    if path.is_file():
        return json.loads(path.read_text())
    scores = selection_scores(search, spec)
    index = json.loads((search / "index.json").read_text())
    candidates = {
        key: {
            "parameters": index["architectures"][key],
            "fit_root": root,
        } for key, root in index["candidates"].items()
    }
    return select_candidate(
        search, candidates, scores,
        tuple(DEFAULT_ARCHITECTURE), spec["selection"],
    )


def finish_comparison(search: Path, spec: dict,
                      base: dict) -> Path:
    """Create the three-model comparison after selected fits complete."""
    selection = json.loads((search / "selection.json").read_text())
    index = json.loads((search / "index.json").read_text())
    roots = {
        "corteg": Path(index["reference_roots"]["corteg"]),
        "hilo": Path(index["reference_roots"]["hilo"]),
        "cnn_gru": Path(selection["fit_root"]),
    }
    reference, _ = load_comparison_study(
        REPO / spec["reference_comparison"]
    )
    study = dict(reference)
    study["name"] = spec["comparison_name"]
    study["subjects"] = spec["subjects"]
    study["fractions"] = spec["final"]["fractions"]
    study["seeds"] = spec["final"]["seeds"]
    study["fit_ids"] = {
        key: json.loads((root / "index.json").read_text())["semantic_id"]
        for key, root in roots.items()
    }
    comparison = ensure_comparison(study, base, roots)
    (comparison / "provenance" / "hpo_selection.json").write_text(
        json.dumps({
            "search_id": index["semantic_id"],
            "selection": selection,
        }, indent=2) + "\n"
    )
    summary = report(comparison)
    if not summary["complete"]:
        raise RuntimeError("Selected comparison has incomplete fits.")
    return comparison


def run_fit(args: argparse.Namespace) -> None:
    """Execute one explicitly configured fit or checkpoint scoring."""
    search = args.search_root
    index = json.loads((search / "index.json").read_text())
    root = Path(index["candidates"][args.candidate])
    architecture = index["architectures"][args.candidate]
    config = json.loads(
        (root / "provenance" / "resolved_config.json").read_text()
    )
    if args.mode == "screen":
        if args.fraction != 1.0 or args.seed != 0 or (
            architecture == DEFAULT_ARCHITECTURE
        ):
            raise ValueError(
                "Screening requires a new candidate at 100%, seed 0."
            )
        fit_job(
            root, config, "cnn_gru", args.subject, args.fraction,
            args.seed, 0, architecture=architecture,
            validation_only=True,
        )
    elif args.mode == "score":
        selection_path = search / "selection.json"
        if not selection_path.is_file():
            raise ValueError("Cannot score test before selection.")
        selection = json.loads(selection_path.read_text())
        if args.candidate != selection["candidate"] or (
            args.fraction != 1.0 or args.seed != 0
        ):
            raise ValueError(
                "Test scoring must use the selected seed-0 100% fit."
            )
        score_validation_fit(
            root, config, args.subject, args.fraction,
            args.seed, architecture,
        )
    elif args.mode == "final":
        selection_path = search / "selection.json"
        if not selection_path.is_file():
            raise ValueError("Cannot fit final campaign before selection.")
        selection = json.loads(selection_path.read_text())
        if args.candidate != selection["candidate"]:
            raise ValueError(
                "Final fitting must use the selected architecture."
            )
        fit_job(
            root, config, "cnn_gru", args.subject, args.fraction,
            args.seed, 0, architecture=architecture,
        )
    else:
        raise ValueError(f"Unknown fit mode {args.mode}.")
