"""Launch and report the resumable Stanford FingerFlex GPU campaign.

Input: a tracked campaign JSON, ignored ``configs/fingerflex_local.json``,
registered raw patient files, and a local model checkpoint.
Output: one semantic-hash artifact with ``index.json`` and ``provenance/``,
``prepared/``, ``runs/``, ``status/``, and ``reports/`` directories. Each
worker writes one subject-level fit under ``runs/``; status JSONL records job
starts, completions, failures, and GPU assignment. ``report`` writes paired
subject statistics when both model results exist for all nine patients.
"""
from __future__ import annotations

import argparse
import hashlib
import shlex
import itertools
import json
import os
import queue
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
    load_local_settings,
    preparation_id,
    prepare_cached_subject,
    resolve_campaign,
    sha256_file,
)
CONFIG = REPO / "configs" / "fingerflex_campaign.json"
MIN_FREE_MIB = 16384
GPU_COUNT = 4
METRICS = ("pearson_r", "mse", "r2")
LEGACY_FIT_SOURCE_SHA256 = (
    "e6a7e8e8da575f22f3c0991422ad23b61936d5eb3663448a6dfdfe78e4380f75"
)
PORTABLE_FIT_SOURCE_SHA256 = (
    "3c5ad2c3c0bdfbe3325af8a9afaf4ffb6225acec6f0808284fdbc9863ee7f23c"
)


def source_revision(path: Path) -> str:
    """Get the git commit of the repository containing ``path``."""
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()


def semantic_identity(config: dict) -> tuple[str, dict]:
    """Hash training semantics and the referenced preprocessing identity.

    Job selectors, file roots, devices, and git commit IDs do not change the
    identity. Source *contents* and checkpoint bytes do.
    """
    source_paths = [
        REPO / "experiments" / "fingerflex_campaign.py",
        REPO / "experiments" / "run_regression_hilo_clean.py",
        REPO / "train" / "lr_schedule.py",
        Path(config["hilo_source"]),
    ]
    source_paths.extend(sorted((REPO / "models" / "steegformer").glob(
        "*.py"
    )))
    model_hashes = {}
    for path in source_paths:
        label = (
            str(path.relative_to(REPO)) if path.is_relative_to(REPO)
            else "HiLoFuseNet/finger_regression/models/nn_regressors.py"
        )
        model_hashes[label] = sha256_file(path)
    semantics = {
        "preprocessing_id": preparation_id(config),
        "training": config["training"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "model_source_sha256": model_hashes,
    }
    if config["delay_ms"]:
        semantics["delay_ms"] = config["delay_ms"]
    canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12], semantics


def artifact_root(config: dict) -> Path:
    """Resolve the campaign artifact root from name and semantic hash."""
    identity, _ = semantic_identity(config)
    return Path(config["artifact_base"]) / f"{config['name']}__{identity}"


def _save_selections(provenance: Path, config: dict) -> None:
    """Union job selectors while recording the exact latest invocation."""
    path = provenance / "selections.json"
    keys = ("subjects", "models", "fractions", "seeds")
    current = json.loads(path.read_text()) if path.exists() else {
        key: [] for key in keys
    }
    for key in keys:
        current.setdefault(key, [])
        for item in config[key]:
            if item not in current[key]:
                current[key].append(item)
    path.write_text(json.dumps(current, indent=2) + "\n")
    launches = provenance / "launches"
    launches.mkdir(exist_ok=True)
    invocation = json.dumps(config, sort_keys=True)
    digest = hashlib.sha256(invocation.encode()).hexdigest()[:12]
    (launches / f"{time.time_ns()}_{digest}.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )


def _legacy_training_matches(candidate: Path, config: dict) -> bool:
    """Verify an old completed campaign used the same training semantics."""
    provenance = candidate / "provenance"
    paths = (
        provenance / "resolved_config.json",
        provenance / "semantics.json",
        provenance / "input_hashes.json",
    )
    if not all(path.is_file() for path in paths):
        return False
    old = json.loads(paths[0].read_text())
    if any(old.get(key) != config[key] for key in (
        "training", "checkpoint_sha256"
    )):
        return False
    source_hashes = json.loads(paths[1].read_text()).get(
        "campaign_source_sha256", {}
    )
    old_fit_hash = source_hashes.get(
        "experiments/fingerflex_campaign.py"
    )
    current_fit_hash = sha256_file(
        REPO / "experiments" / "fingerflex_campaign.py"
    )
    compatible_path_refactor = (
        old_fit_hash == LEGACY_FIT_SOURCE_SHA256
        and current_fit_hash == PORTABLE_FIT_SOURCE_SHA256
    )
    if old_fit_hash != current_fit_hash and not compatible_path_refactor:
        return False
    inputs = json.loads(paths[2].read_text())
    if inputs.get("hilo_revision") != source_revision(
        Path(config["hilo_source"]).parent
    ):
        return False
    old_revision = inputs.get("corteg_revision")
    if not old_revision:
        return False
    model_paths = (
        "models/steegformer",
        "experiments/run_regression_hilo_clean.py",
        "train/lr_schedule.py",
    )
    comparison = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet",
         old_revision, "--", *model_paths],
        check=False,
    )
    if comparison.returncode not in (0, 1):
        raise RuntimeError(
            f"Could not compare model source against {old_revision}."
        )
    return comparison.returncode == 0


def _adopt_legacy_runs(
    root: Path,
    config: dict,
    cache_index: dict,
) -> None:
    """Link validated completed fits from the old artifact into this run."""
    source_name = cache_index.get("adopted_from")
    if not source_name:
        return
    source = Path(source_name)
    if not _legacy_training_matches(source, config):
        return
    linked = 0
    for subject in config["subjects"]:
        for job in jobs_for_subject(subject, config):
            result = job_path(source, job) / "result.json"
            if not result.is_file():
                continue
            saved = json.loads(result.read_text())
            fields = ("subject", "model", "fraction", "seed")
            if any(saved.get(key) != job[key] for key in fields):
                raise ValueError(f"Legacy result job differs: {result}.")
            if saved.get("delay_ms") != job["delay"]:
                raise ValueError(f"Legacy result delay differs: {result}.")
            if not saved.get("test_scored_after_restore"):
                raise ValueError(f"Legacy result lacks test audit: {result}.")
            metrics = saved.get("metrics", {}).get("overall", {})
            if set(metrics) != set(METRICS) or not np.isfinite(
                list(metrics.values())
            ).all():
                raise ValueError(
                    f"Legacy result metrics are invalid: {result}."
                )
            if not (result.parent / "best.pt").is_file():
                raise ValueError(
                    f"Legacy validation checkpoint is missing: {result}."
                )
            target = job_path(root, job)
            if target.exists() or target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(result.parent)
            linked += 1
    record = root / "provenance" / "adopted_runs.json"
    if not record.exists():
        record.write_text(json.dumps({
            "source_artifact": str(source),
            "linked_completed_fits": linked,
        }, indent=2) + "\n")


def ensure_artifact(config: dict) -> Path:
    """Create a training root linked to shared prepared samples."""
    checkpoint_digest = sha256_file(Path(config["checkpoint"]))
    if checkpoint_digest != config["checkpoint_sha256"]:
        raise ValueError(
            f"ST-EEGFormer checkpoint hash mismatch: {checkpoint_digest}."
        )
    subjects = ensure_cache(config)
    root = artifact_root(config)
    identity, semantics = semantic_identity(config)
    for folder in ("provenance", "runs", "status", "reports"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    prepared = root / "prepared"
    if prepared.is_symlink():
        if prepared.resolve() != subjects.resolve():
            raise ValueError(f"Prepared cache link differs: {prepared}.")
    elif prepared.exists():
        raise ValueError(
            f"Expected a shared prepared-cache link at {prepared}."
        )
    else:
        prepared.symlink_to(subjects)
    index_path = root / "index.json"
    index = {
        "name": config["name"],
        "semantic_id": identity,
        "preprocessing_id": preparation_id(config),
        "prepared_cache": str(cache_root(config)),
        "relative_paths": {
            key: key for key in (
                "provenance", "prepared", "runs", "status", "reports"
            )
        },
    }
    if index_path.exists():
        if json.loads(index_path.read_text()) != index:
            raise ValueError(f"Artifact identity conflict: {index_path}.")
    else:
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    provenance = root / "provenance"
    semantic_path = provenance / "semantics.json"
    if semantic_path.exists():
        if json.loads(semantic_path.read_text()) != semantics:
            raise ValueError(f"Training semantics changed: {semantic_path}.")
    else:
        semantic_path.write_text(json.dumps(semantics, indent=2) + "\n")
    preprocessing_path = provenance / "resolved_preprocessing.json"
    if preprocessing_path.exists():
        if json.loads(preprocessing_path.read_text()) != (
            config["preprocessing"]
        ):
            raise ValueError(
                f"Preprocessing settings changed: {preprocessing_path}."
            )
    else:
        preprocessing_path.write_text(
            json.dumps(config["preprocessing"], indent=2) + "\n"
        )
    (provenance / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    _save_selections(provenance, config)
    cache_index = json.loads((cache_root(config) / "index.json").read_text())
    _adopt_legacy_runs(root, config, cache_index)
    inputs = {
        "checkpoint_sha256": checkpoint_digest,
        "raw_sha256": cache_index["raw_sha256"],
        "corteg_revision": source_revision(REPO),
        "hilo_revision": source_revision(
            Path(config["hilo_source"]).parent
        ),
    }
    (provenance / "input_hashes.json").write_text(
        json.dumps(inputs, indent=2) + "\n"
    )
    return root


def _refresh_input_hashes(root: Path, config: dict) -> None:
    """Record verified raw hashes after adding prepared subjects."""
    cache_index = json.loads((cache_root(config) / "index.json").read_text())
    path = root / "provenance" / "input_hashes.json"
    inputs = json.loads(path.read_text())
    inputs["raw_sha256"] = cache_index["raw_sha256"]
    path.write_text(json.dumps(inputs, indent=2) + "\n")


def _artifact_config(
    root: Path,
    *,
    require_local: bool = False,
) -> dict:
    """Load saved settings, adding local paths for old fit artifacts."""

    path = root / "provenance" / "resolved_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run configuration is missing: {path}.")
    config = json.loads(path.read_text())
    selections = root / "provenance" / "selections.json"
    if selections.exists():
        config.update(json.loads(selections.read_text()))
    config.setdefault("delay_ms", 0)
    config.setdefault("models", ["corteg", "hilo"])
    if require_local and (
        "hilo_source" not in config or "tmux_executable" not in config
    ):
        local = load_local_settings(
            REPO / "configs" / "fingerflex_local.json"
        )
        for key in ("hilo_source", "tmux_executable"):
            config.setdefault(key, local[key])
    return config


def rank_gpus(query: str, required: int = GPU_COUNT) -> list[int]:
    """Select most free GPUs, with utilization as tie breaker.

    Parameters
    ----------
    query : str
        NVIDIA CSV rows of index, free MiB, and utilization percent.
    required : int, optional
        Number of GPUs, default 4.

    Returns
    -------
    list[int]
        Physical GPU indices in descending free-memory order.
    """
    entries = []
    for line in query.splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 3:
            raise ValueError(f"Expected three NVIDIA columns, got {line!r}.")
        index, free, utilization = map(int, cells)
        entries.append((index, free, utilization))
    if len(entries) < required:
        raise ValueError(f"Need {required} GPUs, found {len(entries)}.")
    entries.sort(key=lambda row: (-row[1], row[2], row[0]))
    selected = entries[:required]
    if any(row[1] < MIN_FREE_MIB for row in selected):
        raise RuntimeError(
            f"Selected GPUs lack {MIN_FREE_MIB} MiB free: {selected}."
        )
    return [row[0] for row in selected]


def select_gpus() -> list[int]:
    """Query NVIDIA memory and select four fit-capable GPUs."""
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True,
    )
    return rank_gpus(query)


def jobs_for_subject(subject: str, config: dict) -> list[dict]:
    """Enumerate configured independent fits for one patient."""
    jobs = [{
        "subject": subject, "fraction": fraction,
        "model": model, "seed": seed, "delay": config["delay_ms"],
    } for fraction, model, seed in itertools.product(
        config["fractions"], config["models"], config["seeds"]
    )]
    jobs.sort(key=lambda job: (
        -job["fraction"], job["model"] != "corteg", job["seed"],
    ))
    return jobs


def job_path(root: Path, job: dict) -> Path:
    """Return structured output directory for one job."""
    fraction = f"{int(job['fraction'] * 100)}pct"
    if job["delay"]:
        fraction += f"_delay{job['delay']}ms"
    return (
        root / "runs" / fraction / job["model"]
        / job["subject"] / f"seed{job['seed']}"
    )


def _event(root: Path, event: dict, lock: threading.Lock) -> None:
    """Append one synchronized job transition to status history."""
    record = {"time_utc": time.time(), **event}
    with lock:
        with (root / "status" / "jobs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def _worker(
    gpu: int,
    work: queue.Queue,
    root: Path,
    lock: threading.Lock,
) -> None:
    """Run queued independent fits on one assigned physical GPU."""
    while True:
        job = work.get()
        if job is None:
            work.task_done()
            return
        output = job_path(root, job)
        output.mkdir(parents=True, exist_ok=True)
        if (output / "result.json").exists():
            _event(root, {"state": "skipped", "job": job,
                          "gpu": gpu}, lock)
            work.task_done()
            continue
        _event(root, {"state": "running", "job": job,
                      "gpu": gpu}, lock)
        command = [
            sys.executable, str(REPO / "scripts" / "fingerflex_campaign.py"),
            "fit", "--root", str(root), "--model", job["model"],
            "--subject", job["subject"],
            "--fraction", str(job["fraction"]),
            "--seed", str(job["seed"]),
            "--delay", str(job["delay"]),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with (output / "worker.log").open("a") as log:
            completed = subprocess.run(
                command, cwd=REPO, env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
        state = (
            "complete" if completed.returncode == 0
            and (output / "result.json").exists() else "failed"
        )
        _event(root, {"state": state, "job": job, "gpu": gpu,
                      "exit_code": completed.returncode}, lock)
        work.task_done()


def coordinate(root: Path, config: dict) -> None:
    """Reuse or prepare subjects and run selected fits on four GPUs."""
    selected = select_gpus()
    status = root / "status"
    (status / "gpu_assignment.json").write_text(
        json.dumps({"physical_gpu_indices": selected,
                    "minimum_free_mib": MIN_FREE_MIB}, indent=2)
    )
    lock = threading.Lock()
    work: queue.Queue = queue.Queue()
    threads = [threading.Thread(
        target=_worker, args=(gpu, work, root, lock), daemon=True,
    ) for gpu in selected]
    for thread in threads:
        thread.start()
    ordered = list(dict.fromkeys(config["subjects"]))
    try:
        expected = sum(len(jobs_for_subject(subject, config))
                       for subject in ordered)
        if expected == 0:
            raise ValueError("The campaign has no subject-level jobs.")
        for subject in ordered:
            _event(root, {"state": "preparing", "subject": subject}, lock)
            reused = prepare_cached_subject(
                subject, config, (config["delay_ms"],)
            )
            _refresh_input_hashes(root, config)
            _event(root, {
                "state": "reused" if reused else "prepared",
                "subject": subject,
            }, lock)
            for job in jobs_for_subject(subject, config):
                work.put(job)
                _event(root, {"state": "queued", "job": job}, lock)
        work.join()
    except Exception:
        (status / "coordinator_error.log").write_text(
            traceback.format_exc()
        )
        raise
    finally:
        for _ in threads:
            work.put(None)
        for thread in threads:
            thread.join()


def _paired_test(difference: np.ndarray) -> float:
    """Exact two-sided paired sign-flip p-value across subject differences."""
    if len(difference) < 2 or not np.isfinite(difference).all():
        raise ValueError(
            "Paired test needs at least two finite subject differences."
        )
    observed = abs(float(difference.mean()))
    signs = np.array(list(itertools.product(
        (-1, 1), repeat=len(difference)
    )))
    null = np.abs((signs * difference).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjust exactly the three overall-metric p-values."""
    if set(pvalues) != set(METRICS):
        raise ValueError("Holm family must contain r, MSE, and R².")
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    floor = 0.0
    for index, metric in enumerate(ordered):
        floor = max(floor, (len(ordered) - index) * pvalues[metric])
        adjusted[metric] = min(1.0, floor)
    return adjusted


def report(root: Path, config: dict) -> dict:
    """Aggregate completed seeds and paired subjects without imputations."""
    summary: dict = {"delay_ms": config["delay_ms"], "fractions": {}}
    for fraction in config["fractions"]:
        name = f"{int(fraction * 100)}pct"
        per_subject = {}
        for subject in config["subjects"]:
            pair = {}
            for model in config["models"]:
                values = []
                for seed in config["seeds"]:
                    path = job_path(root, {
                        "subject": subject, "fraction": fraction,
                        "model": model, "seed": seed,
                        "delay": config["delay_ms"],
                    }) / "result.json"
                    if path.exists():
                        values.append(json.loads(path.read_text())[
                            "metrics"]["overall"])
                if values:
                    pair[model] = {
                        "seeds": len(values),
                        "mean": {key: float(np.mean([row[key] for row in values]))
                                 for key in METRICS},
                        "std": {key: float(np.std([row[key] for row in values],
                                                  ddof=1)) if len(values) > 1
                                else None for key in METRICS},
                    }
            per_subject[subject] = pair
        complete = (
            set(config["models"]) == {"corteg", "hilo"}
            and len(config["subjects"]) >= 2
            and all(
                all(
                    per_subject[subject].get(model, {}).get("seeds")
                    == len(config["seeds"])
                    for model in ("corteg", "hilo")
                )
                for subject in config["subjects"]
            )
        )
        entry = {"subjects": per_subject, "paired_complete": complete}
        if complete:
            differences = {
                key: np.array([
                    per_subject[subject]["corteg"]["mean"][key]
                    - per_subject[subject]["hilo"]["mean"][key]
                    for subject in config["subjects"]
                ]) for key in METRICS
            }
            raw = {key: _paired_test(value)
                   for key, value in differences.items()}
            corrected = _holm(raw)
            entry["paired_test"] = {
                key: {"mean_difference": float(value.mean()),
                      "p_two_sided": raw[key], "p_holm": corrected[key]}
                for key, value in differences.items()
            }
        summary["fractions"][name] = entry
    path = root / "reports" / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Dispatch launch, coordinator, fit, and report commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=(
            "prepare", "launch", "coordinate", "fit", "report"
        )
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--local-config", type=Path)
    parser.add_argument("--model", choices=("corteg", "hilo"))
    parser.add_argument("--subject")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--delay", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        config = resolve_campaign(args.config, args.local_config)
        subjects = ensure_cache(config)
        for subject in config["subjects"]:
            reused = prepare_cached_subject(
                subject, config, (config["delay_ms"],)
            )
            print(
                f"{subject}: {'reused' if reused else 'prepared'} "
                f"{subjects / subject}", flush=True,
            )
    elif args.command == "launch":
        config = resolve_campaign(args.config, args.local_config)
        root = ensure_artifact(config)
        print(root, flush=True)
        pending = sum(
            not (job_path(root, job) / "result.json").exists()
            for subject in config["subjects"]
            for job in jobs_for_subject(subject, config)
        )
        if pending == 0:
            print("All selected fits are already complete.", flush=True)
            return
        session = f"fingerflex_{root.name[-12:]}"
        command = shlex.join([
            "cd", str(REPO),
        ]) + " && " + shlex.join([
            sys.executable,
            str(REPO / "scripts" / "fingerflex_campaign.py"),
            "coordinate", "--root", str(root),
        ]) + " > " + shlex.quote(str(
            root / "status" / "coordinator.log"
        )) + " 2>&1"
        subprocess.run(
            [config["tmux_executable"], "new-session",
             "-d", "-s", session, command], check=True,
        )
        print(f"tmux session: {session}", flush=True)
    else:
        if args.root is None:
            raise ValueError("--root is required after launch.")
        config = _artifact_config(
            args.root,
            require_local=args.command in ("coordinate", "fit"),
        )
        if args.command == "coordinate":
            coordinate(args.root, config)
            report(args.root, _artifact_config(args.root))
        elif args.command == "fit":
            from experiments.fingerflex_campaign import fit_job
            fit_job(
                args.root, config, args.model, args.subject,
                args.fraction, args.seed, args.delay,
            )
        else:
            print(json.dumps(report(args.root, config), indent=2))


if __name__ == "__main__":
    main()
