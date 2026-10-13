"""Run the two-arm Stanford FingerFlex ridge binning comparison.

Input: configs/fingerflex_ridge_binning.json, ignored local compute
settings, completed unbinned ridge fits, and shared prepared arrays.
Output: one hashed binned-ridge root under fits, one hashed two-arm
comparison under comparisons, and per-campaign training logs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_cache import preparation_id, sha256_file
from experiments.fingerflex_binned_ridge import (
    MODEL, fit_binned_ridge,
)
from infrastructure.compute import load_resources
from infrastructure.fit_attempts import (
    append_fit_event, quarantine_incomplete,
)
from infrastructure.training_lifecycle import (
    attempt_log_path, coordinator_scope, fit_task,
    register_campaign,
)
from scripts.fingerflex_comparison import (
    ensure_comparison, find_fit_roots, load_comparison_study,
    report,
)
from scripts.fingerflex_linear_identity import (
    binned_ridge_identity, binned_ridge_root,
    binned_ridge_semantics,
)

CONFIG = REPO / "configs" / "fingerflex_ridge_binning.json"
RESOURCES = REPO / "configs" / "compute_local.json"
CONTROL = "ridge_early"
SESSION = "fingerflex_ridge_binning"
SOURCES = (
    "preprocessing/temporal_bins.py",
    "preprocessing/channel_time_features.py",
    "data/fingerflex_linear.py",
    "experiments/fingerflex_binned_ridge.py",
    "experiments/fingerflex_ridge.py",
    "experiments/fingerflex_campaign.py",
    "experiments/ridge_cv.py",
    "models/linear_fusion.py",
    "evaluation/regression.py",
    "scripts/fingerflex_ridge_binning.py",
    "scripts/fingerflex_linear_identity.py",
    "data/fingerflex_cache.py",
)


def _fit_path(root: Path, fraction: float, subject: str) -> Path:
    """Locate one deterministic fit folder."""
    return (
        root / "runs" / f"{int(fraction * 100)}pct"
        / MODEL / subject / "deterministic"
    )


def _save_once(path: Path, value: dict) -> None:
    """Create immutable metadata or validate an existing copy."""
    payload = json.dumps(value, indent=2) + "\n"
    normalized = json.loads(payload)
    if path.is_file():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"Resolved artifact metadata differs: {path}.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def initialize(study: dict, base: dict) -> tuple[dict, Path]:
    """Prepare one stable binned fit root and two-arm comparison."""
    if tuple(study["models"]) != (MODEL, CONTROL):
        raise ValueError("Binning study needs binned then unbinned ridge.")
    settings = study["binning"]
    root = binned_ridge_root(base, settings)
    prepared = (
        Path(base["artifact_base"]) / "preprocessing"
        / preparation_id(base) / "subjects"
    )
    if not prepared.is_dir():
        raise FileNotFoundError(f"Shared preparation is absent: {prepared}.")
    for name in ("runs", "status", "provenance"):
        (root / name).mkdir(parents=True, exist_ok=True)
    link = root / "prepared"
    if link.is_symlink():
        if link.resolve() != prepared.resolve():
            raise ValueError(f"Prepared link differs: {link}.")
    elif link.exists():
        raise ValueError(f"Prepared path is not a link: {link}.")
    else:
        link.symlink_to(prepared, target_is_directory=True)
    _save_once(root / "index.json", {
        "model": MODEL,
        "semantic_id": binned_ridge_identity(base, settings),
        "preprocessing_id": preparation_id(base),
        "semantics": binned_ridge_semantics(base, settings),
    })
    _save_once(root / "provenance" / "resolved_config.json", {
        "semantics": binned_ridge_semantics(base, settings),
        "preprocessing_id": preparation_id(base),
    })
    roots = find_fit_roots(study, base)
    comparison = ensure_comparison(study, base, roots)
    tasks = [
        fit_task(
            roots[model],
            (
                roots[model] / "runs"
                / f"{int(fraction * 100)}pct"
                / model / subject / "deterministic" / "result.json"
            ),
            model=model, subject=subject,
            fraction=fraction, seed=None,
        )
        for model, fraction, subject in itertools.product(
            study["models"], study["fractions"], study["subjects"],
        )
    ]
    register_campaign(comparison, tasks)
    return roots, comparison


def _worker(
    jobs: queue.Queue, root: Path, lock: threading.Lock,
    threads: int, config: Path,
) -> None:
    """Run queued fits and preserve diagnostic evidence on failure."""
    while True:
        item = jobs.get()
        if item is None:
            jobs.task_done()
            return
        subject, fraction = item
        attempt_id = uuid.uuid4().hex
        selectors = {
            "model": MODEL, "subject": subject,
            "fraction": fraction, "seed": None,
        }
        attempt_log = attempt_log_path(
            root, selectors, attempt_id,
        )
        try:
            _run_attempt(
                root, subject, fraction, attempt_id,
                attempt_log, lock, threads, config,
            )
        except Exception as error:
            with attempt_log.open("a") as stream:
                traceback.print_exception(error, file=stream)
            append_fit_event(root, {
                **selectors, "state": "failed",
                "attempt_id": attempt_id,
                "attempt_log": str(attempt_log),
                "exit_code": None,
                "error": f"{type(error).__name__}: {error}",
            }, lock)
        finally:
            jobs.task_done()


def _run_attempt(
    root: Path, subject: str, fraction: float,
    attempt_id: str, attempt_log: Path,
    lock: threading.Lock, threads: int, config: Path,
) -> None:
    """Launch one isolated fit and record its exact outcome."""
    output = _fit_path(root, fraction, subject)
    selectors = {
        "model": MODEL, "subject": subject,
        "fraction": fraction, "seed": None,
    }
    if (output / "result.json").is_file():
        append_fit_event(
            root, {**selectors, "state": "skipped"}, lock,
        )
        return
    quarantine_incomplete(output, root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "worker.log").symlink_to(attempt_log)
    (output / "source_revision.json").write_text(
        json.dumps({
            source: sha256_file(REPO / source)
            for source in SOURCES
        }, indent=2) + "\n"
    )
    command = [
        sys.executable, str(REPO / "scripts"
                            / "fingerflex_ridge_binning.py"),
        "fit", "--config", str(config.resolve()),
        "--subject", subject, "--fraction", str(fraction),
    ]
    environment = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    ):
        environment[variable] = str(threads)
    append_fit_event(root, {
        **selectors, "state": "running",
        "attempt_id": attempt_id,
        "attempt_log": str(attempt_log),
    }, lock)
    with attempt_log.open("w") as stream:
        completed = subprocess.run(
            command, cwd=REPO, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    state = (
        "complete" if completed.returncode == 0
        and (output / "result.json").is_file() else "failed"
    )
    append_fit_event(root, {
        **selectors, "state": state,
        "attempt_id": attempt_id,
        "attempt_log": str(attempt_log),
        "exit_code": completed.returncode,
    }, lock)


def coordinate(
    study: dict, roots: dict, comparison: Path,
    resources: dict, config: Path,
) -> None:
    """Dispatch unfinished binned fits to configured CPU workers."""
    root = roots[MODEL]
    lock = threading.RLock()
    jobs: queue.Queue = queue.Queue()
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                jobs, root, lock,
                resources["cpu_threads_per_worker"], config,
            ),
            daemon=True,
        )
        for _ in range(resources["cpu_workers"])
    ]
    with coordinator_scope(comparison, "cpu", (MODEL,)):
        for thread in threads:
            thread.start()
        for fraction, subject in itertools.product(
            study["fractions"], study["subjects"],
        ):
            if (_fit_path(root, fraction, subject) / "result.json").is_file():
                continue
            with lock:
                jobs.put((subject, fraction))
                append_fit_event(root, {
                    "state": "queued", "model": MODEL,
                    "subject": subject, "fraction": fraction,
                    "seed": None,
                }, lock)
        jobs.join()
        for _ in threads:
            jobs.put(None)
        for thread in threads:
            thread.join()
    report(comparison)


def launch(base: dict, config: Path, resources: Path,
           root: Path) -> None:
    """Start one detached CPU coordinator without restarting a live one."""
    executable = base["tmux_executable"]
    existing = subprocess.run(
        [executable, "has-session", "-t", SESSION],
        capture_output=True, check=False,
    )
    if existing.returncode == 0:
        raise RuntimeError(f"Coordinator session is already live: {SESSION}.")
    load_resources(resources)
    log = root / "status" / "binning_coordinator.log"
    command = shlex.join([
        sys.executable, str(REPO / "scripts"
                            / "fingerflex_ridge_binning.py"),
        "coordinate", "--config", str(config.resolve()),
        "--local-resources", str(resources.resolve()),
    ])
    shell = (
        f"cd {shlex.quote(str(REPO))} && {command} "
        f"> {shlex.quote(str(log))} 2>&1"
    )
    subprocess.run(
        [executable, "new-session", "-d", "-s", SESSION, shell],
        check=True,
    )
    print(f"Detached session {SESSION}; log {log.resolve()}")


def main() -> None:
    """Initialize, fit, report, launch, or coordinate the study."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=(
            "init", "fit", "report", "launch", "coordinate",
        ),
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--local-resources", type=Path, default=RESOURCES,
    )
    parser.add_argument("--subject")
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()
    study, base = load_comparison_study(args.config)
    if args.command == "fit":
        if args.subject not in study["subjects"] or (
            args.fraction not in study["fractions"]
        ):
            raise ValueError("Fit selectors are absent from the study.")
        root = binned_ridge_root(base, study["binning"])
        fit_binned_ridge(
            root / "prepared", args.subject, args.fraction,
            _fit_path(root, args.fraction, args.subject),
            study["binning"],
        )
        return
    roots, comparison = initialize(study, base)
    print(comparison.resolve(), flush=True)
    if args.command == "launch":
        launch(base, args.config, args.local_resources, roots[MODEL])
    elif args.command == "coordinate":
        coordinate(
            study, roots, comparison,
            load_resources(args.local_resources), args.config,
        )
    elif args.command == "report":
        result = report(comparison)
        print(f"Complete: {result['complete']}")


if __name__ == "__main__":
    main()
