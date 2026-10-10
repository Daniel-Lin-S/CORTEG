"""Launch and report the resumable Stanford FingerFlex GPU campaign.

Input: ``configs/fingerflex_campaign.json`` and registered raw patient files.
Output: one semantic-hash artifact with ``index.json`` and ``provenance/``,
``prepared/``, ``runs/``, ``status/``, and ``reports/`` directories. Each
worker writes one subject-level fit under ``runs/``; status JSONL records job
starts, completions, failures, and GPU assignment. ``report`` writes paired
subject statistics when both model results exist for all nine patients.
"""
from __future__ import annotations

import argparse
import hashlib
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

from data.fingerflex_campaign import prepare_subject
CONFIG = REPO / "configs" / "fingerflex_campaign.json"
MIN_FREE_MIB = 16384
GPU_COUNT = 4
METRICS = ("pearson_r", "mse", "r2")


def sha256_file(path: Path) -> str:
    """Hash one file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(path: Path) -> str:
    """Get the git commit of the repository containing ``path``."""
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()


def semantic_identity(config: dict) -> tuple[str, dict]:
    """Hash effective scientific settings, excluding seeds and file roots."""
    semantics = {
        key: value for key, value in config.items()
        if key not in ("name", "seeds", "raw_root", "artifact_base")
    }
    semantics["checkpoint"] = sha256_file(Path(config["checkpoint"]))
    semantics["hilo_source_revision"] = source_revision(
        Path("/home/lyh/codes/HiLoFuseNet")
    )
    semantics["corteg_source_revision"] = source_revision(REPO)
    semantics["campaign_source_sha256"] = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in (
            REPO / "data" / "fingerflex_campaign.py",
            REPO / "experiments" / "fingerflex_campaign.py",
            REPO / "scripts" / "fingerflex_campaign.py",
        )
    }
    canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12], semantics


def artifact_root(config: dict) -> Path:
    """Resolve the campaign artifact root from name and semantic hash."""
    identity, _ = semantic_identity(config)
    return Path(config["artifact_base"]) / f"{config['name']}__{identity}"


def ensure_artifact(config: dict) -> Path:
    """Create structured campaign directories and immutable provenance."""
    root = artifact_root(config)
    identity, semantics = semantic_identity(config)
    for folder in ("provenance", "prepared", "runs", "status", "reports"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index = {
        "name": config["name"],
        "semantic_id": identity,
        "relative_paths": {
            key: key for key in (
                "provenance", "prepared", "runs", "status", "reports"
            )
        },
    }
    if index_path.exists():
        previous = json.loads(index_path.read_text())
        if previous != index:
            raise ValueError(f"Artifact identity conflict: {index_path}.")
    else:
        index_path.write_text(json.dumps(index, indent=2))
    provenance = root / "provenance"
    config_path = provenance / "resolved_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        previous.pop("seeds", None)
        current = dict(config)
        current.pop("seeds", None)
        if previous != current:
            raise ValueError(f"Semantic config mismatch: {config_path}.")
    config_path.write_text(json.dumps(config, indent=2))
    (provenance / "semantics.json").write_text(
        json.dumps(semantics, indent=2)
    )
    checkpoint_digest = sha256_file(Path(config["checkpoint"]))
    if checkpoint_digest != config["checkpoint_sha256"]:
        raise ValueError(
            f"ST-EEGFormer checkpoint hash mismatch: {checkpoint_digest}."
        )
    inputs = {
        "checkpoint_sha256": checkpoint_digest,
        "corteg_revision": source_revision(REPO),
        "hilo_revision": source_revision(
            Path("/home/lyh/codes/HiLoFuseNet")
        ),
        "raw_sha256": {
            subject: sha256_file(
                Path(config["raw_root"])
                / subject / f"{subject}_fingerflex.mat"
            ) for subject in config["subjects"]
        },
    }
    input_path = provenance / "input_hashes.json"
    if input_path.exists() and json.loads(input_path.read_text()) != inputs:
        raise ValueError(f"Input hashes changed: {input_path}.")
    input_path.write_text(json.dumps(inputs, indent=2))
    return root


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
    """Enumerate the approved independent fits for one patient."""
    jobs = []
    for fraction, model, seed in itertools.product(
        config["fractions"], ("corteg", "hilo"), config["seeds"]
    ):
        jobs.append({
            "subject": subject, "fraction": fraction,
            "model": model, "seed": seed, "delay": 0,
        })
    if subject in config["delay_subjects"]:
        jobs += [{
            "subject": subject, "fraction": 1.0,
            "model": "corteg", "seed": seed, "delay": 40,
        } for seed in config["seeds"]]
    jobs.sort(key=lambda job: (
        0 if job["subject"] == "mv" and job["model"] == "corteg"
        and job["fraction"] == 1.0 and job["seed"] == 0
        and job["delay"] == 0 else 1,
        job["delay"], job["fraction"] != 1.0,
        job["model"] != "corteg", job["seed"],
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
    """Prepare subjects, queue 171 jobs, and occupy four selected GPUs."""
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
    ordered = ["mv"] + [
        subject for subject in config["subjects"] if subject != "mv"
    ]
    try:
        for subject in ordered:
            delays = (0, 40) if subject in config["delay_subjects"] else (0,)
            _event(root, {"state": "preparing", "subject": subject}, lock)
            prepare_subject(
                subject, Path(config["raw_root"]),
                root / "prepared", config, delays,
            )
            _event(root, {"state": "prepared", "subject": subject}, lock)
            for job in jobs_for_subject(subject, config):
                work.put(job)
                _event(root, {"state": "queued", "job": job}, lock)
        if work.qsize() != 171:
            # Workers may already consume queued jobs; count explicitly.
            expected = sum(len(jobs_for_subject(s, config))
                           for s in config["subjects"])
            if expected != 171:
                raise ValueError(f"Expected 171 jobs, got {expected}.")
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
    if len(difference) != 9 or not np.isfinite(difference).all():
        raise ValueError("Paired test needs nine finite subject differences.")
    observed = abs(float(difference.mean()))
    signs = np.array(list(itertools.product((-1, 1), repeat=9)))
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
    summary: dict = {"fractions": {}, "delay_probe": {}}
    for fraction in config["fractions"]:
        name = f"{int(fraction * 100)}pct"
        per_subject = {}
        for subject in config["subjects"]:
            pair = {}
            for model in ("corteg", "hilo"):
                values = []
                for seed in config["seeds"]:
                    path = job_path(root, {
                        "subject": subject, "fraction": fraction,
                        "model": model, "seed": seed, "delay": 0,
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
        complete = all(
            all(per_subject[subject].get(model, {}).get("seeds")
                == len(config["seeds"]) for model in ("corteg", "hilo"))
            for subject in config["subjects"]
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
    for subject in config["delay_subjects"]:
        probe = {}
        for delay in (0, 40):
            values = []
            for seed in config["seeds"]:
                path = job_path(root, {
                    "subject": subject, "fraction": 1.0,
                    "model": "corteg", "seed": seed, "delay": delay,
                }) / "result.json"
                if path.exists():
                    values.append(json.loads(path.read_text())[
                        "metrics"]["overall"])
            probe[str(delay)] = {
                "seeds": len(values),
                "mean": {key: float(np.mean([row[key] for row in values]))
                         for key in METRICS} if values else None,
            }
        summary["delay_probe"][subject] = probe
    path = root / "reports" / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Dispatch launch, coordinator, fit, and report commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("launch", "coordinate", "fit", "report")
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model", choices=("corteg", "hilo"))
    parser.add_argument("--subject")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--delay", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.command == "launch":
        root = ensure_artifact(config)
        print(root, flush=True)
        session = f"fingerflex_{artifact_root(config).name[-12:]}"
        command = (
            f"cd {REPO} && {sys.executable} "
            f"{REPO / 'scripts' / 'fingerflex_campaign.py'} "
            f"coordinate --root {root} "
            f"> {root / 'status' / 'coordinator.log'} 2>&1"
        )
        subprocess.run(
            ["/home/lyh/plugins/tmux/bin/tmux", "new-session",
             "-d", "-s", session, command], check=True,
        )
        print(f"tmux session: {session}", flush=True)
    elif args.command == "coordinate":
        coordinate(args.root, config)
        report(args.root, config)
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
