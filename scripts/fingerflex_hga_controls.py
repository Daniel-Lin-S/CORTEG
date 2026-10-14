"""Launch resumable Stanford HGA-control fits and comparisons.

Input: configs/fingerflex_hga_controls.json, source campaign, shared
prepared arrays, ignored compute_local.json, and existing reference fits.
Output: five semantic fit roots, two semantic comparison roots, one
training-log folder per group, attempt logs, and detached coordinators.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import itertools
import json
import os
import shlex
import subprocess
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_cache import (
    preparation_id, resolve_campaign, sha256_file,
)
from data.fingerflex_filterbank import cache_subject
from experiments.fingerflex_hga_controls import (
    FILTERBANK_MODELS, NEW_MODELS, fit_control, fit_path,
)
from infrastructure.compute import available_gpus, load_resources
from infrastructure.compute_tokens import worker_slot
from infrastructure.fit_attempts import (
    append_fit_event, quarantine_incomplete,
)
from infrastructure.training_lifecycle import (
    attempt_log_path, coordinator_scope, fit_task,
    refresh_status, register_campaign,
)
from scripts.fingerflex_comparison import (
    ensure_comparison, report,
)
from scripts.fingerflex_hga_identity import (
    filterbank_root, filterbank_semantics, fit_root,
    fit_semantics, semantic_hash, validate_settings,
)
from scripts.fingerflex_linear_identity import ridge_root

CONFIG = REPO / "configs" / "fingerflex_hga_controls.json"
RESOURCES = REPO / "configs" / "compute_local.json"
SESSIONS = {
    "hga_models": "fingerflex_hga_models",
    "filterbank": "fingerflex_filterbank_hga",
}
SOURCES = (
    "data/fingerflex_filterbank.py",
    "data/fingerflex_linear.py",
    "experiments/fingerflex_hga_controls.py",
    "experiments/hga_control_cv.py",
    "experiments/hga_branch_fit.py",
    "experiments/fingerflex_ridge.py",
    "experiments/ridge_cv.py",
    "models/hga_regressors.py",
    "models/temporal_spatial_cnn_gru.py",
    "preprocessing/naplib_filterbank.py",
    "scripts/fingerflex_hga_controls.py",
    "scripts/fingerflex_hga_identity.py",
)


def load_study(path: Path) -> tuple[dict, dict]:
    """Resolve portable choices and source preprocessing locally."""
    study = json.loads(path.read_text())
    validate_settings(study)
    source = REPO / study["source_campaign"]
    base = resolve_campaign(source)
    if base["delay_ms"] != 0:
        raise ValueError("HGA controls require zero-delay preparation.")
    for key in ("subjects", "seeds"):
        if not set(study[key]).issubset(base[key]):
            raise ValueError(f"{key} selectors exceed source campaign.")
    if any(not 0 < value <= 1 for value in study["fractions"]):
        raise ValueError("Fit fractions must lie in (0,1].")
    return study, base


def _save_once(path: Path, value: dict) -> None:
    """Create immutable resolved metadata or verify an existing copy."""
    payload = json.dumps(value, indent=2) + "\n"
    normalized = json.loads(payload)
    if path.is_file():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"Resolved artifact metadata differs: {path}.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _reference_root(base: dict, model: str) -> Path:
    """Resolve one existing compatible CORTEG or ridge-late root."""
    if model == "ridge_late":
        root = ridge_root(base, model)
        if not (root / "index.json").is_file():
            raise FileNotFoundError(f"Ridge-late root absent: {root}.")
        return root
    if model != "corteg":
        raise ValueError(f"Unknown reference model: {model}.")
    parent = Path(base["artifact_base"]) / "fits"
    matches = []
    for path in parent.glob("*/index.json"):
        index = json.loads(path.read_text())
        if index.get("model") == "corteg" and (
            index.get("preprocessing_id") == preparation_id(base)
        ) and index.get("semantics", {}).get("delay_ms") == 0:
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one compatible CORTEG root, got {matches}."
        )
    return matches[0]


def _fit_root_with_legacy_grid(
    base: dict, study: dict, model: str,
) -> tuple[Path, dict | None]:
    """Reuse a unique completed CNN root whose only change is its grid."""
    current = fit_root(base, model, study)
    if model != "cnn_gru_late":
        return current, None
    target = json.loads(json.dumps(fit_semantics(base, model, study)))
    if current.is_dir():
        index = json.loads((current / "index.json").read_text())
        previous = index["semantics"]
        normalized = copy.deepcopy(previous)
        normalized["neural"]["grid"] = study["cnn_gru"]
        if normalized != target:
            raise ValueError(
                f"CNN-GRU fit root semantics differ: {current}."
            )
        return current, (
            previous["neural"]["grid"]
            if previous != target else None
        )
    parent = current.parent
    matches = []
    for index_path in parent.glob("cnn_gru_late__*/index.json"):
        index = json.loads(index_path.read_text())
        previous = index.get("semantics", {})
        if "neural" not in previous:
            continue
        normalized = copy.deepcopy(previous)
        normalized["neural"]["grid"] = study["cnn_gru"]
        if normalized == target:
            matches.append((index_path.parent, previous["neural"]["grid"]))
    if not matches:
        return current, None
    if len(matches) != 1:
        raise ValueError(
            f"Expected one compatible CNN-GRU fit root, got {matches}."
        )
    return matches[0]


def _new_root(base: dict, study: dict, model: str) -> Path:
    """Create one stable model root with fitting-only semantics."""
    root = fit_root(base, model, study)
    for name in ("runs", "cv", "status", "provenance"):
        (root / name).mkdir(parents=True, exist_ok=True)
    prepared = (
        Path(base["artifact_base"]) / "preprocessing"
        / preparation_id(base) / "subjects"
    )
    if not prepared.is_dir():
        raise FileNotFoundError(f"Prepared rows absent: {prepared}.")
    link = root / "prepared"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(prepared, target_is_directory=True)
    elif link.resolve() != prepared.resolve():
        raise ValueError(f"Prepared link differs: {link}.")
    semantics = fit_semantics(base, model, study)
    identity = semantic_hash(semantics)
    index_path = root / "index.json"
    if index_path.is_file():
        recorded = json.loads(index_path.read_text())
        if recorded["semantics"] != json.loads(json.dumps(semantics)):
            alias_path = root.parent / "semantic_aliases.json"
            aliases = json.loads(alias_path.read_text())
            if aliases.get(f"{model}__{identity}") != root.name:
                raise ValueError(
                    f"Fit semantics differ without an alias: {root}."
                )
            _save_once(
                root / "provenance" / "semantic_versions"
                / f"{identity}.json",
                {"semantics": semantics, "study": study},
            )
            return root
    _save_once(index_path, {
        "model": model, "semantic_id": identity,
        "preprocessing_id": preparation_id(base),
        "semantics": semantics,
    })
    _save_once(root / "provenance/resolved_config.json", {
        "semantics": semantics,
        "source_campaign": study["source_campaign"],
        "preprocessing_id": preparation_id(base),
    })
    resolved_study = root / "provenance/resolved_study.json"
    if not resolved_study.is_file():
        _save_once(resolved_study, study)
    return root


def _initialize(study: dict, base: dict) -> tuple[dict, dict, Path]:
    """Register both groups, including reused reference completions."""
    cache = filterbank_root(base, study)
    cache.mkdir(parents=True, exist_ok=True)
    _save_once(cache / "index.json", {
        "semantic_id": semantic_hash(
            filterbank_semantics(base, study)
        ),
        "semantics": filterbank_semantics(base, study),
    })
    roots = {
        "corteg": _reference_root(base, "corteg"),
        "ridge_late": _reference_root(base, "ridge_late"),
    }
    roots.update({
        model: _new_root(base, study, model)
        for model in NEW_MODELS
    })
    comparisons = {}
    for group, definition in study["groups"].items():
        models = definition["models"]
        selected = {model: roots[model] for model in models}
        comparison_study = {
            "name": definition["name"],
            "models": models,
            "subjects": study["subjects"],
            "fractions": study["fractions"],
            "seeds": study["seeds"],
            "comparison": study["comparison"],
            "plot_labels": {
                model: study["plot_labels"][model]
                for model in models
            },
        }
        comparison = ensure_comparison(
            comparison_study, base, selected,
        )
        tasks = [
            fit_task(
                roots[model],
                fit_path(
                    roots[model], model, fraction, subject, seed,
                ) / "result.json",
                model=model, subject=subject,
                fraction=fraction, seed=seed,
            )
            for model, fraction, subject in itertools.product(
                models, study["fractions"], study["subjects"],
            )
            for seed in (
                study["seeds"] if model in (
                    "corteg", "cnn_gru_late", "mlp_late"
                ) else [None]
            )
        ]
        register_campaign(comparison, tasks)
        comparisons[group] = comparison
    return roots, comparisons, cache


def initialize(study: dict, base: dict) -> tuple[dict, dict, Path]:
    """Serialize comparison registration across detached coordinators."""
    folder = REPO / "logs" / "training"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "hga_initialization.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            return _initialize(study, base)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _provenance() -> dict:
    """Record source bytes and current commit for each new final fit."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
    ).strip()
    return {
        "git_revision": revision,
        "source_sha256": {
            source: sha256_file(REPO / source)
            for source in SOURCES
        },
    }


def _settings(study: dict, resources: dict) -> dict:
    """Attach machine-local SVR cache memory outside fit semantics."""
    if "svr_cache_mib" not in resources:
        raise ValueError(
            "Local compute settings need a positive svr_cache_mib."
        )
    settings = {
        **study["selection"],
        "svr": study["svr"],
        "cnn_gru": study["cnn_gru"],
        "mlp": study["mlp"],
        "neural_training": study["neural_training"],
        "svr_cache_mib": resources["svr_cache_mib"],
    }
    return settings


def _seeds(model: str, study: dict) -> list[int | None]:
    """Return final-fit seed selectors for one new control."""
    if model in ("cnn_gru_late", "mlp_late"):
        return study["seeds"]
    return [None]


def _attempt(
    model: str, subject: str, fraction: float,
    root: Path, study: dict, base: dict, resources: dict,
    cache: Path, lock: threading.RLock, slot_folder: Path,
    gpus: list[int], config: Path,
    attempt_id: str | None = None,
) -> None:
    """Run one isolated subject/fraction attempt and log every seed."""
    seeds = _seeds(model, study)
    pending = [
        seed for seed in seeds
        if not (fit_path(root, model, fraction, subject, seed)
                / "result.json").is_file()
    ]
    if not pending:
        return
    attempt_id = uuid.uuid4().hex if attempt_id is None else attempt_id
    selectors = [
        {
            "model": model, "subject": subject,
            "fraction": fraction, "seed": seed,
        } for seed in pending
    ]
    attempt_log = attempt_log_path(root, selectors[0], attempt_id)
    for seed in pending:
        output = fit_path(root, model, fraction, subject, seed)
        quarantine_incomplete(output, root)
        output.mkdir(parents=True, exist_ok=True)
        (output / "worker.log").symlink_to(attempt_log)
        (output / "source_revision.json").write_text(
            json.dumps(_provenance(), indent=2) + "\n"
        )
    kind = "gpu" if model in (
        "cnn_gru_late", "mlp_late"
    ) else "cpu"
    indices = gpus if kind == "gpu" else list(
        range(resources["cpu_workers"])
    )
    with worker_slot(slot_folder, kind, indices) as slot:
        for selector in selectors:
            append_fit_event(root, {
                **selector, "state": "running",
                "attempt_id": attempt_id,
                "attempt_log": str(attempt_log),
            }, lock)
        command = [
            sys.executable, str(REPO / "scripts"
                                / "fingerflex_hga_controls.py"),
            "fit", "--config", str(config.resolve()),
            "--model", model, "--subject", subject,
            "--fraction", str(fraction),
        ]
        env = dict(os.environ)
        for variable in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            env[variable] = str(resources["cpu_threads_per_worker"])
        if kind == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        with attempt_log.open("w") as stream:
            completed = subprocess.run(
                command, cwd=REPO, env=env,
                stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
    for selector in selectors:
        output = fit_path(
            root, model, fraction, subject, selector["seed"],
        )
        state = (
            "complete" if (output / "result.json").is_file()
            else "failed"
        )
        append_fit_event(root, {
            **selector, "state": state,
            "attempt_id": attempt_id,
            "attempt_log": str(attempt_log),
            "exit_code": completed.returncode,
        }, lock)


def _guard_attempt(*args: object) -> None:
    """Keep other independent tasks running after one worker failure."""
    attempt_id = uuid.uuid4().hex
    try:
        _attempt(*args, attempt_id)
    except Exception as error:
        model, subject, fraction, root = args[:4]
        selectors = {
            "model": model, "subject": subject,
            "fraction": fraction,
            "seed": _seeds(model, args[4])[0],
        }
        attempt_log = attempt_log_path(
            root, selectors, attempt_id,
        )
        with attempt_log.open("a") as stream:
            traceback.print_exception(error, file=stream)
        print(
            f"Worker failed for {model}/{subject}/{fraction}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr, flush=True,
        )
        traceback.print_exc()
        for seed in _seeds(model, args[4]):
            marker = fit_path(
                root, model, fraction, subject, seed,
            ) / "result.json"
            if marker.is_file():
                continue
            append_fit_event(root, {
                "model": model, "subject": subject,
                "fraction": fraction, "seed": seed,
                "state": "failed", "exit_code": None,
                "attempt_id": attempt_id,
                "attempt_log": str(attempt_log),
                "error": f"{type(error).__name__}: {error}",
            }, args[8])


def coordinate(
    group: str, study: dict, base: dict, roots: dict,
    comparison: Path, cache: Path,
    resources: dict, config: Path,
) -> None:
    """Run each group's queued tasks under shared global slots."""
    lock = threading.RLock()
    models = [
        model for model in study["groups"][group]["models"]
        if model in NEW_MODELS
    ]
    jobs = [
        (model, subject, fraction)
        for model, subject, fraction in itertools.product(
            models, study["subjects"], study["fractions"],
        )
    ]
    jobs.sort(key=lambda row: (
        study["subjects"].index(row[1]),
        study["fractions"].index(row[2]),
        models.index(row[0]),
    ))
    for model, subject, fraction in jobs:
        root = roots[model]
        for seed in _seeds(model, study):
            if (fit_path(root, model, fraction, subject, seed)
                    / "result.json").is_file():
                continue
            append_fit_event(root, {
                "model": model, "subject": subject,
                "fraction": fraction, "seed": seed,
                "state": "queued",
            }, lock)
    gpu_models = ("cnn_gru_late", "mlp_late")
    cpu_jobs = [row for row in jobs if row[0] not in gpu_models]
    gpu_jobs = [row for row in jobs if row[0] in gpu_models]
    gpus = available_gpus(resources) if gpu_jobs else []
    slots = REPO / "logs" / "training" / "shared_slots"
    with coordinator_scope(comparison, group, tuple(models)):
        with ThreadPoolExecutor(
            max_workers=resources["cpu_workers"]
        ) as cpu_pool, ThreadPoolExecutor(
            max_workers=resources["gpu_workers"]
        ) as gpu_pool:
            futures = []
            for pool, selected in (
                (cpu_pool, cpu_jobs), (gpu_pool, gpu_jobs),
            ):
                for model, subject, fraction in selected:
                    futures.append(pool.submit(
                        _guard_attempt,
                        model, subject, fraction,
                        roots[model], study, base, resources,
                        cache, lock, slots, gpus, config,
                    ))
            for future in futures:
                future.result()
    report(comparison)


def launch(base: dict, comparisons: dict,
           config: Path, resources: Path) -> None:
    """Start both detached coordinators after preflight."""
    executable = base["tmux_executable"]
    limits = load_resources(resources)
    available_gpus(limits)
    for group, comparison in comparisons.items():
        session = SESSIONS[group]
        live = subprocess.run(
            [executable, "has-session", "-t", session],
            capture_output=True, check=False,
        )
        if live.returncode == 0:
            raise RuntimeError(
                f"Coordinator session is already live: {session}."
            )
        log = comparison / "coordinator.log"
        command = shlex.join([
            sys.executable,
            str(REPO / "scripts/fingerflex_hga_controls.py"),
            "coordinate", "--group", group,
            "--config", str(config.resolve()),
            "--local-resources", str(resources.resolve()),
        ])
        shell = (
            f"cd {shlex.quote(str(REPO))} && {command} "
            f"> {shlex.quote(str(log))} 2>&1"
        )
        subprocess.run(
            [executable, "new-session", "-d", "-s", session, shell],
            check=True,
        )
        print(f"Detached {session}; log {log.resolve()}", flush=True)


def main() -> None:
    """Initialize, fit, coordinate, report, or launch both groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=(
            "init", "fit", "coordinate", "report", "launch",
        ),
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--local-resources", type=Path, default=RESOURCES,
    )
    parser.add_argument("--group", choices=tuple(SESSIONS))
    parser.add_argument("--model", choices=NEW_MODELS)
    parser.add_argument("--subject")
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()
    study, base = load_study(args.config)
    if args.command == "fit":
        if args.model is None or args.subject not in study["subjects"] or (
            args.fraction not in study["fractions"]
        ):
            raise ValueError("Fit model, subject, or fraction is invalid.")
        root, legacy_grid = _fit_root_with_legacy_grid(
            base, study, args.model,
        )
        cache = filterbank_root(base, study)
        subject_cache = None
        if args.model in FILTERBANK_MODELS:
            prepared = root / "prepared" / args.subject
            subject_cache = cache_subject(
                args.subject, base, prepared,
                cache, study["filterbank"],
            )
        settings = _settings(
            study, load_resources(args.local_resources),
        )
        if legacy_grid is not None:
            settings["cnn_gru_legacy_grid"] = legacy_grid
        fit_control(
            root / "prepared", args.model,
            args.subject, args.fraction,
            root, settings,
            study["seeds"],
            torch.device(
                "cuda:0" if args.model in (
                    "cnn_gru_late", "mlp_late"
                ) else "cpu"
            ), subject_cache,
        )
        return
    roots, comparisons, cache = initialize(study, base)
    if args.command == "launch":
        launch(base, comparisons, args.config, args.local_resources)
    elif args.command == "coordinate":
        if args.group is None:
            raise ValueError("Coordinator needs one group selector.")
        coordinate(
            args.group, study, base, roots,
            comparisons[args.group], cache,
            load_resources(args.local_resources), args.config,
        )
    elif args.command == "report":
        for group, root in comparisons.items():
            summary = report(root)
            print(f"{group}: {root.resolve()} {summary['complete']}")
        print(json.dumps(refresh_status(), indent=2))
    else:
        for group, root in comparisons.items():
            print(f"{group}: {root.resolve()}")


if __name__ == "__main__":
    main()
