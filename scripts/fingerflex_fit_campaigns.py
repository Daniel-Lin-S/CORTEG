"""Coordinate independent FingerFlex model-fit campaigns.

Input is a portable four-model JSON and ignored local settings from its
source campaign. Output under artifact_base/fits is one model-specific root
per fit identity with index.json, provenance/, prepared/, runs/, status/.
Old CORTEG/HiLo fit directories are verified and linked, never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_cache import (
    cache_root,
    ensure_cache,
    preparation_id,
    prepare_cached_subject,
    resolve_campaign,
    sha256_file,
)
from data.fingerflex_campaign import digest_array
from data.fingerflex_wavelet import prepare_wavelet, recipe, recipe_id
from experiments.fingerflex_campaign import (
    fit_job,
    metric_values,
    split_fit_rows,
)
from experiments.fingerflex_pls import fit_pls
from scripts.fingerflex_campaign import rank_gpus

DEFAULT_CONFIG = REPO / "configs" / "fingerflex_four_models.json"
METRICS = ("pearson_r", "mse", "r2")
LEGACY_MODELS = ("corteg", "hilo")
NEW_MODELS = ("pls", "cnn_gru")
MODEL_NAMES = {
    "corteg": "corteg_small",
    "hilo": "hilo",
    "pls": "pls_wavelet",
    "cnn_gru": "cnn_gru_hga",
}
CNN_TRAINING_KEYS = (
    "epochs", "patience", "batch_size", "base_lr",
    "weight_decay", "warmup_epochs", "min_lr",
)


def _digest(value: dict) -> str:
    """Hash a JSON semantic object to a short, stable identifier."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def load_study(path: Path) -> tuple[dict, dict]:
    """Load portable selectors and resolve the source campaign locally."""
    source = json.loads(path.read_text())
    campaign = REPO / source["source_campaign"]
    base = resolve_campaign(campaign)
    if base["delay_ms"] != 0:
        raise ValueError("Four-model fits require the 0 ms source campaign.")
    if set(source["models"]) != set((*LEGACY_MODELS, *NEW_MODELS)):
        raise ValueError("Expected CORTEG, HiLo, PLS, and CNN-GRU models.")
    for key in ("subjects", "fractions", "seeds"):
        if not source[key] or len(set(source[key])) != len(source[key]):
            raise ValueError(f"Invalid {key} selectors: {source[key]}.")
        if not set(source[key]).issubset(base[key]):
            raise ValueError(f"{key} selectors exceed the source campaign.")
    if set(source["cnn_gru"]) != set(CNN_TRAINING_KEYS):
        raise ValueError("CNN-GRU training keys are incomplete or unexpected.")
    if source["pls"]["max_components"] < 1:
        raise ValueError("PLS maximum components must be positive.")
    return source, base


def find_legacy(base: dict, study: dict) -> Path:
    """Find the newest complete legacy artifact with matching semantics."""
    parent = Path(base["artifact_base"])
    candidates = []
    for root in parent.glob(f"{base['name']}__*"):
        semantics_path = root / "provenance" / "semantics.json"
        if not semantics_path.is_file():
            continue
        semantics = json.loads(semantics_path.read_text())
        if semantics.get("preprocessing_id") != preparation_id(base):
            continue
        if semantics.get("training") != base["training"]:
            continue
        if semantics.get("checkpoint_sha256") != (
            base["checkpoint_sha256"]
        ):
            continue
        if not semantics.get("model_source_sha256"):
            continue
        if all(
            (root / "runs" / f"{int(fraction * 100)}pct"
             / model / subject / f"seed{seed}" / "result.json").is_file()
            for subject, fraction, model, seed in itertools.product(
                study["subjects"], study["fractions"],
                LEGACY_MODELS, study["seeds"],
            )
        ):
            candidates.append(root)
    if not candidates:
        raise FileNotFoundError(
            "No complete, semantically matching historical fit artifact."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def fit_semantics(
    model: str,
    study: dict,
    base: dict,
    legacy: Path,
) -> dict:
    """Identify one model's fit semantics independently of comparisons."""
    semantics = {
        "model": model,
        "preprocessing_id": preparation_id(base),
        "delay_ms": 0,
    }
    if model in LEGACY_MODELS:
        previous = json.loads(
            (legacy / "provenance" / "semantics.json").read_text()
        )
        hashes = previous["model_source_sha256"]
        shared = (
            "experiments/fingerflex_campaign.py",
            "experiments/run_regression_hilo_clean.py",
            "train/lr_schedule.py",
        )
        selected = list(shared)
        if model == "corteg":
            selected.extend(
                key for key in hashes if key.startswith("models/steegformer/")
            )
            semantics["checkpoint_sha256"] = base["checkpoint_sha256"]
        else:
            selected.append(
                "HiLoFuseNet/finger_regression/models/nn_regressors.py"
            )
        semantics["source_sha256"] = {
            key: hashes[key] for key in selected
        }
        semantics["training"] = base["training"]
        semantics["feature"] = model
    else:
        names = (
            ("data/fingerflex_wavelet.py", "experiments/fingerflex_pls.py",
             "experiments/fingerflex_campaign.py")
            if model == "pls" else
            ("models/temporal_spatial_cnn_gru.py",
             "experiments/fingerflex_campaign.py", "train/lr_schedule.py")
        )
        semantics["source_sha256"] = {
            name: sha256_file(REPO / name) for name in names
        }
        if model == "pls":
            semantics["feature"] = recipe()
            semantics["training"] = study["pls"]
        else:
            semantics["feature"] = {
                "source": "corteg/train_hi.npy[...,0]",
                "transform": "70-200Hz_Hilbert_amplitude",
            }
            semantics["training"] = study["cnn_gru"]
    return semantics


def fit_root(
    model: str,
    study: dict,
    base: dict,
    legacy: Path,
) -> Path:
    """Locate a model-specific fit campaign by semantic identity."""
    semantics = fit_semantics(model, study, base, legacy)
    name = MODEL_NAMES[model]
    return (
        Path(base["artifact_base"]) / "fits"
        / f"{name}__{_digest(semantics)}"
    )


def job_path(root: Path, model: str, fraction: float,
             subject: str, seed: int | None) -> Path:
    """Locate a single fit inside a model-specific campaign."""
    repeat = "deterministic" if seed is None else f"seed{seed}"
    return (
        root / "runs" / f"{int(fraction * 100)}pct"
        / model / subject / repeat
    )


def _validated_old_result(
    path: Path,
    model: str,
    subject: str,
    fraction: float,
    seed: int,
    prepared: Path,
) -> None:
    """Verify saved boundaries, predictions, metrics, and checkpoint."""
    result_path = path / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Old completion is missing: {result_path}.")
    result = json.loads(result_path.read_text())
    expected = {
        "model": model, "subject": subject, "fraction": fraction,
        "seed": seed, "delay_ms": 0,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Old fit identity differs: {result_path}.")
    if not result.get("test_scored_after_restore"):
        raise ValueError(f"Old fit lacks test restore: {result_path}.")
    common = prepared / subject / "common"
    indices = np.load(common / "train_indices.npy")
    test_indices = np.load(common / "test_indices.npy")
    test_y = np.load(common / "test_y.npy")
    fit, validation = split_fit_rows(len(indices), fraction)
    boundaries = result["boundaries"]
    hashes = {
        "fit_indices_sha256": digest_array(indices[fit]),
        "validation_indices_sha256": digest_array(
            indices[validation]
        ),
        "test_grid_sha256": digest_array(test_indices),
        "target_values_sha256": digest_array(test_y),
    }
    if any(boundaries.get(key) != value
           for key, value in hashes.items()):
        raise ValueError(f"Old fit boundary hashes differ: {result_path}.")
    if not (path / "best.pt").is_file():
        raise FileNotFoundError(f"Old fit checkpoint is missing: {path}.")
    saved = np.load(path / "test_predictions.npz")
    values = metric_values(saved["y"], saved["y_hat"])["overall"]
    if any(not np.isclose(
        values[key], result["metrics"]["overall"][key],
        rtol=1e-6, atol=1e-7,
    ) for key in METRICS):
        raise ValueError(f"Old fit metrics differ: {result_path}.")


def ensure_fit_roots(
    study: dict,
    base: dict,
    legacy: Path,
) -> dict[str, Path]:
    """Create model-specific roots and link verified historical fits."""
    prepared = ensure_cache(base)
    roots = {}
    for model in study["models"]:
        semantics = fit_semantics(model, study, base, legacy)
        root = fit_root(model, study, base, legacy)
        for directory in ("provenance", "runs", "status"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        target = root / "prepared"
        if target.is_symlink():
            if target.resolve() != prepared.resolve():
                raise ValueError(f"Prepared link differs: {target}.")
        elif target.exists():
            raise ValueError(f"Expected prepared link at {target}.")
        else:
            target.symlink_to(prepared)
        index = {
            "model": model,
            "semantic_id": _digest(semantics),
            "preprocessing_id": preparation_id(base),
            "semantics": semantics,
        }
        index_path = root / "index.json"
        if index_path.exists():
            if json.loads(index_path.read_text()) != index:
                raise ValueError(f"Fit identity conflict: {index_path}.")
        else:
            index_path.write_text(json.dumps(index, indent=2) + "\n")
        resolved = dict(base)
        if model == "cnn_gru":
            resolved["training"] = {
                **base["training"], **study["cnn_gru"],
            }
        if model == "pls":
            resolved["pls_max_components"] = study["pls"]["max_components"]
        provenance = root / "provenance"
        (provenance / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n"
        )
        selections_path = provenance / "selections.json"
        selections = (
            json.loads(selections_path.read_text())
            if selections_path.exists() else {
                "subjects": [], "fractions": [], "seeds": [],
            }
        )
        for key in ("subjects", "fractions", "seeds"):
            additions = [] if model == "pls" and key == "seeds" else study[key]
            selections[key] = list(dict.fromkeys(
                selections[key] + additions
            ))
        selections_path.write_text(
            json.dumps(selections, indent=2) + "\n"
        )
        roots[model] = root
    for model in LEGACY_MODELS:
        root = roots[model]
        for subject, fraction, seed in itertools.product(
            study["subjects"], study["fractions"], study["seeds"],
        ):
            old = job_path(legacy, model, fraction, subject, seed)
            _validated_old_result(
                old, model, subject, fraction, seed, prepared,
            )
            new = job_path(root, model, fraction, subject, seed)
            if new.is_symlink():
                if new.resolve() != old.resolve():
                    raise ValueError(f"Historical fit link differs: {new}.")
                continue
            if new.exists():
                raise ValueError(f"Historical fit path exists: {new}.")
            new.parent.mkdir(parents=True, exist_ok=True)
            new.symlink_to(old.resolve())
    return roots


def _quarantine_incomplete(output: Path, root: Path) -> None:
    """Move a partial fit aside before a fresh attempt."""
    if not output.exists() or (output / "result.json").is_file():
        return
    destination = (
        root / "status" / "quarantine"
        / f"{output.parent.name}_{output.name}_{time.time_ns()}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.rename(destination)


def _event(path: Path, event: dict, lock: threading.Lock) -> None:
    """Append one synchronized coordinator status event."""
    with lock:
        with path.open("a") as stream:
            stream.write(json.dumps({
                "time_utc": time.time(), **event,
            }) + "\n")


def _run_fit(
    model: str,
    root: Path,
    subject: str,
    fraction: float,
    seed: int | None,
) -> None:
    """Execute one new PLS or CNN-GRU fit from saved local provenance."""
    output = job_path(root, model, fraction, subject, seed)
    if (output / "result.json").is_file():
        return
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (root / "provenance" / "resolved_config.json").read_text()
    )
    if model == "pls":
        prepare_wavelet(config, subject)
        fit_pls(
            root / "prepared", config, subject, fraction, output,
            max_components=config["pls_max_components"],
        )
    else:
        if seed is None:
            raise ValueError("CNN-GRU requires an integer random seed.")
        fit_job(root, config, model, subject, fraction, seed, 0)


def _worker(
    model: str,
    root: Path,
    jobs: queue.Queue,
    events: Path,
    lock: threading.Lock,
    gpu: int | None,
) -> None:
    """Run independent fits, logging errors without stopping other jobs."""
    while True:
        item = jobs.get()
        if item is None:
            jobs.task_done()
            break
        subject, fraction, seed = item
        output = job_path(root, model, fraction, subject, seed)
        if (output / "result.json").is_file():
            _event(events, {
                "state": "skipped", "model": model,
                "subject": subject, "fraction": fraction, "seed": seed,
            }, lock)
            jobs.task_done()
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()), "fit",
            "--model", model, "--fit-root", str(root),
            "--subject", subject, "--fraction", str(fraction),
        ]
        if seed is not None:
            command.extend(("--seed", str(seed)))
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            env.update({
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            })
        _quarantine_incomplete(output, root)
        output.mkdir(parents=True, exist_ok=True)
        _event(events, {
            "state": "running", "model": model, "subject": subject,
            "fraction": fraction, "seed": seed, "gpu": gpu,
        }, lock)
        with (output / "worker.log").open("a") as log:
            completed = subprocess.run(
                command, cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        state = (
            "complete" if completed.returncode == 0
            and (output / "result.json").is_file() else "failed"
        )
        _event(events, {
            "state": state, "model": model, "subject": subject,
            "fraction": fraction, "seed": seed, "gpu": gpu,
            "exit_code": completed.returncode,
        }, lock)
        jobs.task_done()


def coordinate(study: dict, roots: dict[str, Path]) -> None:
    """Dispatch PLS on CPU and CNN-GRU across four capable GPUs."""
    query = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    gpus = rank_gpus(query)
    cnn_status = roots["cnn_gru"] / "status"
    (cnn_status / "gpu_assignment.json").write_text(
        json.dumps({"physical_gpu_indices": gpus}, indent=2) + "\n"
    )
    lock = threading.Lock()
    cnn_queue: queue.Queue = queue.Queue()
    pls_queue: queue.Queue = queue.Queue()
    threads = [
        threading.Thread(
            target=_worker,
            args=("cnn_gru", roots["cnn_gru"], cnn_queue,
                  cnn_status / "jobs.jsonl", lock, gpu),
            daemon=True,
        ) for gpu in gpus
    ]
    threads.append(threading.Thread(
        target=_worker,
        args=("pls", roots["pls"], pls_queue,
              roots["pls"] / "status" / "jobs.jsonl",
              lock, None),
        daemon=True,
    ))
    for thread in threads:
        thread.start()
    for subject, fraction, seed in itertools.product(
        study["subjects"], study["fractions"], study["seeds"],
    ):
        cnn_queue.put((subject, fraction, seed))
    for subject, fraction in itertools.product(
        study["subjects"], study["fractions"],
    ):
        pls_queue.put((subject, fraction, None))
    cnn_queue.join()
    pls_queue.join()
    for _ in gpus:
        cnn_queue.put(None)
    pls_queue.put(None)
    for thread in threads:
        thread.join()


def main() -> None:
    """Initialize, launch, coordinate, or fit model-specific campaigns."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("init", "launch", "coordinate", "fit"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=NEW_MODELS)
    parser.add_argument("--fit-root", type=Path)
    parser.add_argument("--subject")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.command == "fit":
        if args.fit_root is None or args.model is None:
            raise ValueError("Fit requires --fit-root and --model.")
        _run_fit(
            args.model, args.fit_root, args.subject,
            args.fraction, args.seed,
        )
        return
    study, base = load_study(args.config)
    legacy = find_legacy(base, study)
    roots = ensure_fit_roots(study, base, legacy)
    from scripts.fingerflex_comparison import ensure_comparison
    comparison = ensure_comparison(study, base, roots)
    print(comparison, flush=True)
    if args.command == "init":
        return
    if args.command == "launch":
        pending = sum(
            not (job_path(
                roots["cnn_gru"], "cnn_gru", fraction,
                subject, seed,
            ) / "result.json").is_file()
            for subject, fraction, seed in itertools.product(
                study["subjects"], study["fractions"], study["seeds"],
            )
        ) + sum(
            not (job_path(
                roots["pls"], "pls", fraction, subject, None,
            ) / "result.json").is_file()
            for subject, fraction in itertools.product(
                study["subjects"], study["fractions"],
            )
        )
        if pending == 0:
            print("All new fits are complete.", flush=True)
            return
        session = f"fingerflex_fits_{roots['cnn_gru'].name[-12:]}"
        command = shlex.join([
            sys.executable, str(Path(__file__).resolve()),
            "coordinate", "--config", str(args.config.resolve()),
        ])
        log = roots["cnn_gru"] / "status" / "coordinator.log"
        shell_command = (
            f"cd {shlex.quote(str(REPO))} && {command} "
            f"> {shlex.quote(str(log))} 2>&1"
        )
        subprocess.run([
            base["tmux_executable"], "new-session", "-d",
            "-s", session, shell_command,
        ], check=True)
        print(f"tmux session: {session}; pending fits: {pending}", flush=True)
        return
    try:
        coordinate(study, roots)
        from scripts.fingerflex_comparison import report
        report(comparison)
    except Exception:
        (roots["cnn_gru"] / "status"
         / "coordinator_error.log").write_text(
            traceback.format_exc()
        )
        raise


if __name__ == "__main__":
    main()
