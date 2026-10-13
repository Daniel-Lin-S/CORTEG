"""Record fit transitions and quarantine incomplete fit folders.

Input: a model fit root, event selectors, and one incomplete fit path.
Output: append-only status/jobs.jsonl, per-campaign training logs, and
quarantined incomplete attempts under status/quarantine. Completed fit
markers are never changed.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from infrastructure.training_lifecycle import append_training_event


def append_fit_event(
    root: Path, event: dict, lock: threading.Lock,
) -> None:
    """Write one synchronized model and campaign transition."""
    record = {"time_utc": time.time(), **event}
    with lock:
        with (root / "status" / "jobs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        append_training_event(root, record)


def quarantine_incomplete(output: Path, root: Path) -> None:
    """Move a prior incomplete attempt aside before retrying."""
    if not output.exists() or (output / "result.json").is_file():
        return
    destination = (
        root / "status" / "quarantine"
        / f"{output.parent.name}_{output.name}_{time.time_ns()}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.rename(destination)
