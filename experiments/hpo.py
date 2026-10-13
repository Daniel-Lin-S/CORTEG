"""Coordinate model searches without depending on a dataset or model.

Input: a configuration grid, candidate metadata, subject selectors, and task
callbacks. Output: deterministic candidates, pending jobs, selected candidate,
and synchronized worker status events. Dataset adapters own fit artifacts.
"""
from __future__ import annotations

import itertools
import json
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from infrastructure.training_lifecycle import (
    append_training_event, attempt_log_path,
)

import numpy as np
import torch


def enumerate_grid(grid: dict[str, list]) -> list[dict]:
    """Expand a Cartesian search space, omitting partial exclusions.

    Parameters
    ----------
    grid : dict[str, list]
        Named parameter lists and optional ``exclude`` dictionaries.

    Returns
    -------
    list[dict]
        Unique candidate dictionaries in configuration order.
    """
    names = tuple(key for key in grid if key != "exclude")
    if not names:
        raise ValueError("Search grid has no parameters.")
    for name in names:
        values = grid[name]
        if not values or len({json.dumps(x) for x in values}) != len(values):
            raise ValueError(f"Grid parameter {name} needs unique values.")
    exclusions = grid.get("exclude", [])
    if any(not exclusion or not set(exclusion) <= set(names)
           for exclusion in exclusions):
        raise ValueError("Grid exclusions must name search parameters.")
    points = [dict(zip(names, values)) for values in itertools.product(
        *(grid[name] for name in names)
    )]
    points = [point for point in points if not any(
        all(point[name] == value for name, value in exclusion.items())
        for exclusion in exclusions
    )]
    if not points:
        raise ValueError("Search grid has no valid candidates.")
    return points


def pending_jobs(
    candidates: dict[str, dict], units: list[str],
    final: dict, selected: str | None, selection: dict,
    complete: Callable[[dict], bool],
) -> list[dict]:
    """Return incomplete screening or selected final jobs.

    Parameters
    ----------
    candidates : dict[str, dict]
        Candidate metadata keyed by stable candidate name.
    units : list[str]
        Independent fit identifiers.
    final : dict
        Final ``fractions`` and ``seeds`` selectors.
    selected : str or None
        Frozen winner, or None during screening.
    selection : dict
        Screening ``fraction`` and ``seed`` selectors.
    complete : Callable[[dict], bool]
        Task callback checking a fit completion marker.

    Returns
    -------
    list[dict]
        Pending job descriptors.
    """
    if not units or len(set(units)) != len(units):
        raise ValueError("Fit units must be nonempty and unique.")
    jobs = []
    if selected is None:
        for key, metadata in candidates.items():
            if metadata.get("historical", False):
                continue
            for unit in units:
                jobs.append({
                    "candidate": key, "unit": unit,
                    "fraction": selection["fraction"],
                    "seed": selection["seed"], "mode": "screen",
                })
    else:
        if selected not in candidates:
            raise ValueError(f"Selected candidate {selected} is unknown.")
        for unit, fraction, seed in itertools.product(
            units, final["fractions"], final["seeds"],
        ):
            mode = "score" if (
                fraction == selection["fraction"]
                and seed == selection["seed"]
            ) and not (
                candidates[selected].get("historical", False)
            ) else "final"
            jobs.append({
                "candidate": selected, "unit": unit,
                "fraction": fraction, "seed": seed, "mode": mode,
            })
    return [job for job in jobs if not complete(job)]


def select_candidate(
    search: Path, candidates: dict[str, dict],
    scores: dict[str, float], parameter_order: tuple[str, ...],
    selection_rule: dict,
) -> dict:
    """Freeze the highest mean validation score with ascending tie break.

    Parameters
    ----------
    search : Path
        Search artifact root.
    candidates : dict[str, dict]
        Metadata containing each candidate's ``parameters``.
    scores : dict[str, float]
        Equal-subject validation means for every candidate.
    parameter_order : tuple[str, ...]
        Ascending tie-break parameter names.
    selection_rule : dict
        Provenance of validation selection.

    Returns
    -------
    dict
        Frozen selection record.
    """
    path = search / "selection.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["candidate"] not in candidates:
            raise ValueError("Frozen selection names an unknown candidate.")
        return saved
    if set(scores) != set(candidates) or any(
        not np.isfinite(score) for score in scores.values()
    ):
        raise ValueError("Need finite scores for every candidate.")
    winner = min(scores, key=lambda key: (
        -scores[key], *(
            candidates[key]["parameters"][name]
            for name in parameter_order
        ),
    ))
    selected = {
        "candidate": winner,
        "architecture": candidates[winner]["parameters"],
        "fit_root": candidates[winner]["fit_root"],
        "validation_scores": scores,
        "selection_rule": selection_rule,
        "exploratory": True,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(selected, indent=2) + "\n")
    temporary.replace(path)
    return selected


def dispatch_jobs(
    search: Path, jobs: list[dict], workers: list[int],
    execute: Callable[[dict, int, Path], tuple[str, int]],
) -> None:
    """Execute independent jobs with one worker per assigned device.

    Parameters
    ----------
    search : Path
        Search root receiving ``status/jobs.jsonl`` events.
    jobs : list[dict]
        Pending independent jobs.
    workers : list[int]
        Assigned device identifiers.
    execute : Callable[[dict, int, Path], tuple[str, int]]
        Task callback receiving an attempt log path and returning state
        and process exit code.
    """
    if not jobs:
        return
    if not workers:
        raise ValueError("Cannot dispatch jobs without workers.")
    work: queue.Queue = queue.Queue()
    lock = threading.RLock()

    def worker(device: int) -> None:
        while True:
            job = work.get()
            if job is None:
                work.task_done()
                return
            attempt_id = uuid.uuid4().hex
            attempt_log = attempt_log_path(search, job, attempt_id)
            event = {
                "time_utc": time.time(), "state": "running",
                "attempt_id": attempt_id,
                "attempt_log": str(attempt_log),
                "device": device, **job,
            }
            with lock:
                with (search / "status" / "jobs.jsonl").open("a") as log:
                    log.write(json.dumps(event) + "\n")
                append_training_event(search, event)
            failure = None
            try:
                state, exit_code = execute(
                    job, device, attempt_log,
                )
            except Exception as error:
                state, exit_code = "failed", -1
                failure = f"{type(error).__name__}: {error}"
            event = {
                "time_utc": time.time(), "state": state,
                "attempt_id": attempt_id,
                "attempt_log": str(attempt_log),
                "device": device, "exit_code": exit_code, **job,
            }
            if failure is not None:
                event["error"] = failure
            with lock:
                with (search / "status" / "jobs.jsonl").open("a") as log:
                    log.write(json.dumps(event) + "\n")
                append_training_event(search, event)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(device,))
               for device in workers]
    for thread in threads:
        thread.start()
    for job in jobs:
        event = {"time_utc": time.time(), "state": "queued", **job}
        with lock:
            work.put(job)
            with (search / "status" / "jobs.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
            append_training_event(search, event)
    work.join()
    for _ in threads:
        work.put(None)
    for thread in threads:
        thread.join()


def select_gpus(runtime: dict) -> list[int]:
    """Choose fit-capable CUDA devices by free memory then utilization.

    Parameters
    ----------
    runtime : dict
        ``gpu_workers`` and ``minimum_gpu_free_mib`` configuration.

    Returns
    -------
    list[int]
        Physical GPU indices, most free first.
    """
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in query.splitlines():
        cells = [int(cell.strip()) for cell in line.split(",")]
        if len(cells) != 3:
            raise ValueError(f"Expected three NVIDIA columns: {line!r}.")
        rows.append(tuple(cells))
    count = runtime["gpu_workers"]
    if len(rows) < count:
        raise RuntimeError(f"Need {count} GPUs, found {len(rows)}.")
    rows.sort(key=lambda row: (-row[1], row[2], row[0]))
    minimum = runtime["minimum_gpu_free_mib"]
    if any(row[1] < minimum for row in rows[:count]):
        raise RuntimeError(
            f"Selected GPUs need at least {minimum} MiB free each."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for GPU fit scheduling.")
    return [row[0] for row in rows[:count]]
