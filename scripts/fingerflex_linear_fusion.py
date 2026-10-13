"""Run the Stanford FingerFlex neural extension and ridge fusion study.

Input: configs/fingerflex_linear_fusion.json, ignored compute_local.json,
existing model folders under <artifact_base>/fits, and their shared prepared
arrays. Output: new 50pct neural fits in those folders, one stable folder per
ridge model, and a four-model comparison under <artifact_base>/comparisons.
Completion is recorded by each fit's result.json; status/jobs.jsonl records
worker transitions. Local paths remain in ignored settings and artifacts.
"""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_cache import (
    preparation_id,
    resolve_campaign,
    sha256_file,
)
from experiments.fingerflex_campaign import fit_job
from experiments.fingerflex_ridge import fit_ridge
from infrastructure.compute import available_gpus, load_resources
from infrastructure.fit_attempts import (
    append_fit_event, quarantine_incomplete,
)
from infrastructure.training_lifecycle import (
    attempt_log_path, coordinator_scope, fit_task, register_campaign,
)
from scripts.fingerflex_comparison import (
    ensure_comparison,
    find_fit_roots,
    load_comparison_study,
    report,
)
from scripts.fingerflex_fit_campaigns import _validated_old_result
from scripts.fingerflex_linear_identity import ridge_index, ridge_root

CONFIG = REPO / "configs" / "fingerflex_linear_fusion.json"
NEURAL = ("corteg", "hilo")
LINEAR = ("ridge_early", "ridge_late")
LEGACY_TRAINER_SHA256 = (
    "e6a7e8e8da575f22f3c0991422ad23b61936d5eb3663448a6dfdfe78e4380f75"
)
CURRENT_TRAINER_SHA256 = (
    "34c4ff5292351ae567921d6bc2fd913845d5be93850cd0fad82e19df827c42e3"
)
RESOURCES = REPO / "configs" / "compute_local.json"
SOURCE_FILES = (
    "experiments/fingerflex_campaign.py",
    "experiments/fingerflex_ridge.py",
    "experiments/ridge_cv.py",
    "models/linear_fusion.py",
    "data/fingerflex_linear.py",
    "preprocessing/channel_time_features.py",
    "evaluation/regression.py",
    "infrastructure/compute.py",
    "scripts/fingerflex_linear_identity.py",
    "scripts/fingerflex_linear_fusion.py",
)


def _source_hashes(root: Path) -> dict[str, str]:
    """Hash local and model-specific sources used by one new fit."""
    hashes = {
        name: sha256_file(REPO / name) for name in SOURCE_FILES
    }
    index = json.loads((root / "index.json").read_text())
    recorded = index["semantics"].get("source_sha256", {})
    if recorded:
        config = json.loads(
            (root / "provenance" / "resolved_config.json").read_text()
        )
        hashes.update({
            key: sha256_file(_source_file(key, config))
            for key in recorded
        })
    return hashes


def _source_file(key: str, base: dict) -> Path:
    """Resolve a recorded source key to its current local file."""
    if key.startswith("HiLoFuseNet/"):
        return Path(base["hilo_source"])
    return REPO / key


def audit_compatibility(
    study: dict,
    base: dict,
    roots: dict[str, Path],
) -> dict:
    """Verify recorded neural fits can share a study with current fits."""
    if sha256_file(REPO / SOURCE_FILES[0]) != CURRENT_TRAINER_SHA256:
        raise ValueError(
            "Current neural training source changed since compatibility "
            "review; inspect its behavior before reusing saved fits."
        )
    audit = {}
    for model in NEURAL:
        root = roots[model]
        index = json.loads((root / "index.json").read_text())
        previous = index["semantics"]
        hashes = previous["source_sha256"]
        if hashes["experiments/fingerflex_campaign.py"] != (
            LEGACY_TRAINER_SHA256
        ):
            raise ValueError(
                f"Unknown saved {model} training revision in {root}."
            )
        mismatches = [
            key for key, expected in hashes.items()
            if key != "experiments/fingerflex_campaign.py"
            and sha256_file(_source_file(key, base)) != expected
        ]
        if mismatches:
            raise ValueError(
                f"{model} behavior-bearing source changed: {mismatches}."
            )
        saved = json.loads(
            (root / "provenance" / "resolved_config.json").read_text()
        )
        if saved["training"] != base["training"]:
            raise ValueError(
                f"{model} saved and current training settings differ."
            )
        if index["preprocessing_id"] != preparation_id(base):
            raise ValueError(
                f"{model} saved preprocessing identity differs."
            )
        if model == "corteg" and previous["checkpoint_sha256"] != (
            base["checkpoint_sha256"]
        ):
            raise ValueError(
                "CORTEG pretrained checkpoint differs from saved fits."
            )
        for subject, fraction, seed in itertools.product(
            study["subjects"], (0.25, 1.0), study["seeds"],
        ):
            path = (
                root / "runs" / f"{int(fraction * 100)}pct"
                / model / subject / f"seed{seed}"
            )
            _validated_old_result(
                path, model, subject, fraction, seed,
                root / "prepared",
            )
        audit[model] = {
            "saved_trainer_sha256": LEGACY_TRAINER_SHA256,
            "current_trainer_sha256": CURRENT_TRAINER_SHA256,
            "unchanged_behavior_sources": sorted(
                set(hashes) - {"experiments/fingerflex_campaign.py"}
            ),
            "completed_fits_verified": (
                len(study["subjects"]) * 2 * len(study["seeds"])
            ),
            "review": (
                "Training-loop extraction, configurable HiLo path, and "
                "CNN-GRU support preserve the CORTEG/HiLo optimizer, "
                "normalization, model, scheduler, and checkpoint rules. "
                "The HiLo adapter-rate edit does not affect its base-only "
                "parameter group."
            ),
        }
    return audit


def _initialize_ridge_roots(base: dict) -> None:
    """Create stable model folders and link shared prepared arrays."""
    prepared = (
        Path(base["artifact_base"]) / "preprocessing"
        / preparation_id(base) / "subjects"
    )
    if not prepared.is_dir():
        raise FileNotFoundError(
            f"Shared FingerFlex preparation is absent: {prepared}."
        )
    for model in LINEAR:
        root = ridge_root(base, model)
        for name in ("runs", "status", "provenance"):
            (root / name).mkdir(parents=True, exist_ok=True)
        link = root / "prepared"
        if link.is_symlink():
            if link.resolve() != prepared.resolve():
                raise ValueError(
                    f"{model} prepared link differs: {link}."
                )
        elif link.exists():
            raise ValueError(
                f"Expected prepared symlink for {model}: {link}."
            )
        else:
            link.symlink_to(prepared)
        index = ridge_index(base, model)
        path = root / "index.json"
        if path.is_file():
            if json.loads(path.read_text()) != index:
                raise ValueError(
                    f"Stable model identity conflict: {path}."
                )
        else:
            path.write_text(json.dumps(index, indent=2) + "\n")


def _register_progress(
    study: dict, roots: dict[str, Path], comparison: Path,
) -> None:
    """Register every selected fit, including reusable neural fractions."""
    tasks = []
    for model, fraction, subject in itertools.product(
        study["models"], study["fractions"], study["subjects"],
    ):
        seeds = study["seeds"] if model in NEURAL else (None,)
        for seed in seeds:
            repeat = (
                "deterministic" if seed is None else f"seed{seed}"
            )
            marker = (
                roots[model] / "runs" / f"{int(fraction * 100)}pct"
                / model / subject / repeat / "result.json"
            )
            tasks.append(fit_task(
                roots[model], marker, model=model, subject=subject,
                fraction=fraction, seed=seed,
            ))
    register_campaign(comparison, tasks)


def initialize(study: dict, base: dict) -> tuple[dict, Path]:
    """Validate reusable fits and prepare stable new artifacts."""
    roots = {
        model: find_fit_roots({
            **study, "models": [model],
        }, base)[model]
        for model in NEURAL
    }
    audit = audit_compatibility(study, base, roots)
    for model in NEURAL:
        path = (
            roots[model] / "provenance"
            / "linear_fusion_compatibility.json"
        )
        if path.is_file() and json.loads(path.read_text()) != audit[model]:
            raise ValueError(f"Compatibility audit differs: {path}.")
        if not path.is_file():
            path.write_text(json.dumps(audit[model], indent=2) + "\n")
    _initialize_ridge_roots(base)
    for model in LINEAR:
        roots[model] = ridge_root(base, model)
    ordered = {model: roots[model] for model in study["models"]}
    comparison = ensure_comparison(study, base, ordered)
    _register_progress(study, ordered, comparison)
    return ordered, comparison


def _job_path(
    root: Path,
    model: str,
    subject: str,
    fraction: float,
) -> Path:
    """Locate one new deterministic or three-seed fit parent."""
    return (
        root / "runs" / f"{int(fraction * 100)}pct"
        / model / subject
    )


def _worker(
    jobs: queue.Queue,
    roots: dict[str, Path],
    lock: threading.Lock,
    gpu: int | None,
    resources: dict[str, int],
) -> None:
    """Run independent fits and keep the queue live after failures."""
    while True:
        item = jobs.get()
        if item is None:
            jobs.task_done()
            return
        attempt_id = uuid.uuid4().hex
        try:
            _run_fit_attempt(
                item, roots, lock, gpu, resources, attempt_id,
            )
        except Exception as error:
            model, subject, fraction, seed = item
            append_fit_event(roots[model], {
                "state": "failed", "model": model,
                "subject": subject, "fraction": fraction,
                "seed": seed, "exit_code": None,
                "attempt_id": attempt_id,
                "error": f"{type(error).__name__}: {error}",
            }, lock)
        finally:
            jobs.task_done()


def _run_fit_attempt(
    item: tuple[str, str, float, int | None],
    roots: dict[str, Path], lock: threading.Lock,
    gpu: int | None, resources: dict[str, int],
    attempt_id: str,
) -> None:
    """Persist one fit result or one attributable failure event."""
    model, subject, fraction, seed = item
    root = roots[model]
    parent = _job_path(root, model, subject, fraction)
    repeat = "deterministic" if seed is None else f"seed{seed}"
    output = parent / repeat
    if (output / "result.json").is_file():
        append_fit_event(root, {
            "state": "skipped", "model": model,
            "subject": subject, "fraction": fraction,
            "seed": seed,
        }, lock)
        return
    quarantine_incomplete(output, root)
    output.mkdir(parents=True, exist_ok=True)
    selectors = {
        "model": model, "subject": subject,
        "fraction": fraction, "seed": seed,
    }
    attempt_log = attempt_log_path(root, selectors, attempt_id)
    (output / "worker.log").symlink_to(attempt_log)
    (output / "source_revision.json").write_text(
        json.dumps(_source_hashes(root), indent=2) + "\n"
    )
    command = [
        sys.executable, str(REPO / "scripts"
                            / "fingerflex_linear_fusion.py"),
        "fit", "--model", model, "--subject", subject,
        "--fraction", str(fraction),
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            env[name] = str(resources["cpu_threads_per_worker"])
    append_fit_event(root, {
        "state": "running", "model": model,
        "subject": subject, "fraction": fraction,
        "seed": seed, "gpu": gpu,
        "attempt_id": attempt_id,
        "attempt_log": str(attempt_log),
    }, lock)
    with attempt_log.open("w") as log:
        completed = subprocess.run(
            command, cwd=REPO, env=env, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    state = (
        "complete" if completed.returncode == 0
        and (output / "result.json").is_file()
        else "failed"
    )
    append_fit_event(root, {
        "state": state, "model": model,
        "subject": subject, "fraction": fraction,
        "seed": seed, "gpu": gpu,
        "exit_code": completed.returncode,
        "attempt_id": attempt_id,
        "attempt_log": str(attempt_log),
    }, lock)


def coordinate(
    kind: str, study: dict, roots: dict, resources: dict[str, int],
) -> None:
    """Run the configured parallel fits for one resource class."""
    comparison_parent = next(iter(roots.values())).parents[1]
    matches = list((comparison_parent / "comparisons").glob(
        study["name"] + "__*"
    ))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one comparison root, got {matches}."
        )
    comparison = matches[0]
    _register_progress(study, roots, comparison)
    lock = threading.RLock()
    jobs: queue.Queue = queue.Queue()
    devices = (
        available_gpus(resources) if kind == "neural"
        else [None] * resources["cpu_workers"]
    )
    threads = [
        threading.Thread(
            target=_worker,
            args=(jobs, roots, lock, gpu, resources),
            daemon=True,
        ) for gpu in devices
    ]
    models = NEURAL if kind == "neural" else LINEAR
    with coordinator_scope(comparison, kind, models):
        for thread in threads:
            thread.start()
        _dispatch_jobs(jobs, roots, lock, kind, study, models)
        jobs.join()
        for _ in threads:
            jobs.put(None)
        for thread in threads:
            thread.join()


def _dispatch_jobs(
    jobs: queue.Queue, roots: dict[str, Path],
    lock: threading.Lock, kind: str, study: dict,
    models: tuple[str, ...],
) -> None:
    """Queue each selected fit and record retries at actual enqueue."""
    fractions = (0.5,) if kind == "neural" else study["fractions"]
    for model, fraction, subject in itertools.product(
        models, fractions, study["subjects"],
    ):
        seeds = study["seeds"] if kind == "neural" else (None,)
        for seed in seeds:
            marker = (
                _job_path(roots[model], model, subject, fraction)
                / (
                    "deterministic" if seed is None
                    else f"seed{seed}"
                ) / "result.json"
            )
            if marker.is_file():
                continue
            with lock:
                jobs.put((model, subject, fraction, seed))
                append_fit_event(roots[model], {
                    "state": "queued", "model": model,
                    "subject": subject, "fraction": fraction,
                    "seed": seed,
                }, lock)


def _session_command(
    kind: str, config: Path, local_resources: Path,
) -> str:
    """Build one detached coordinator command and its log path."""
    command = shlex.join([
        sys.executable,
        str(REPO / "scripts" / "fingerflex_linear_fusion.py"),
        "coordinate", "--kind", kind,
        "--config", str(config.resolve()),
        "--local-resources", str(local_resources.resolve()),
    ])
    return f"cd {shlex.quote(str(REPO))} && {command}"


def launch(
    study: dict,
    base: dict,
    roots: dict[str, Path],
    config: Path,
    local_resources: Path,
) -> None:
    """Launch separate detached CPU and GPU coordinators."""
    executable = base["tmux_executable"]
    sessions = ("fingerflex_linear_neural", "fingerflex_linear_linear")
    for session in sessions:
        existing = subprocess.run(
            [executable, "has-session", "-t", session],
            capture_output=True, check=False,
        )
        if existing.returncode == 0:
            raise RuntimeError(
                f"Coordinator session already exists: {session}."
            )
    available_gpus(load_resources(local_resources))
    for kind, root in (
        ("neural", roots["corteg"]),
        ("linear", roots["ridge_early"]),
    ):
        session = f"fingerflex_linear_{kind}"
        log = root / "status" / "linear_fusion_coordinator.log"
        shell = (
            _session_command(kind, config, local_resources)
            + f" > {shlex.quote(str(log))} 2>&1"
        )
        subprocess.run([
            executable, "new-session", "-d", "-s",
            session, shell,
        ], check=True)
        print(f"Detached session {session}; log {log.resolve()}")


def main() -> None:
    """Initialize, launch, coordinate, or fit the study."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("init", "launch", "coordinate", "fit"),
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--local-resources", type=Path, default=RESOURCES,
    )
    parser.add_argument(
        "--kind", choices=("neural", "linear"),
    )
    parser.add_argument("--model", choices=(*NEURAL, *LINEAR))
    parser.add_argument("--subject")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    study, base = load_comparison_study(args.config)
    if args.command == "fit":
        if args.model is None or args.subject is None or (
            args.fraction is None
        ):
            raise ValueError(
                "Fit needs model, subject, and fraction."
            )
        root = (
            find_fit_roots({
                **study, "models": [args.model],
            }, base)[args.model]
        )
        output = _job_path(
            root, args.model, args.subject, args.fraction,
        ) / (
            "deterministic" if args.seed is None
            else f"seed{args.seed}"
        )
        if args.model in NEURAL:
            if args.seed is None:
                raise ValueError("Neural fit requires a seed.")
            saved = json.loads(
                (root / "provenance"
                 / "resolved_config.json").read_text()
            )
            fit_job(
                root, saved, args.model, args.subject,
                args.fraction, args.seed, 0,
            )
        else:
            fit_ridge(
                root / "prepared", args.subject,
                args.fraction, args.model, output,
            )
        return
    if args.command in ("init", "launch"):
        roots, comparison = initialize(study, base)
        print(comparison.resolve(), flush=True)
        if args.command == "launch":
            launch(
                study, base, roots, args.config, args.local_resources,
            )
        return
    if args.kind is None:
        raise ValueError("Coordinate requires --kind.")
    roots = find_fit_roots(study, base)
    matches = list(
        (Path(base["artifact_base"]) / "comparisons").glob(
            study["name"] + "__*"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one initialized comparison, got {matches}."
        )
    comparison = matches[0]
    status_root = roots["corteg" if args.kind == "neural" else "ridge_early"]
    event_lock = threading.Lock()
    append_fit_event(status_root, {
        "state": "coordinator_running", "kind": args.kind,
    }, event_lock)
    try:
        coordinate(
            args.kind, study, roots,
            load_resources(args.local_resources),
        )
        with (comparison / "report.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                report(comparison)
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
    except Exception as error:
        append_fit_event(status_root, {
            "state": "coordinator_failed", "kind": args.kind,
            "error": f"{type(error).__name__}: {error}",
        }, event_lock)
        raise
    append_fit_event(status_root, {
        "state": "coordinator_complete", "kind": args.kind,
    }, event_lock)


if __name__ == "__main__":
    main()
