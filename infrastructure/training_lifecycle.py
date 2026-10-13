"""Record brief training transitions in one repository-local log.

Input: a fit or search artifact root and a status event dictionary, or
existing ``<root>/status/jobs.jsonl`` files for historical import.
Output: ignored ``logs/training/events.jsonl`` with one flat record per
transition. The log is diagnostic and never changes fit artifacts.
"""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "logs" / "training" / "events.jsonl"
FIELDS = (
    "state", "model", "subject", "fraction", "seed",
    "candidate", "mode", "kind", "exit_code", "error",
)


def _record(root: Path, event: dict) -> dict:
    """Flatten a coordinator event without machine-specific fields.

    Parameters
    ----------
    root : Path
        Fit or search artifact root.
    event : dict
        One status transition with ``time_utc`` and ``state``.

    Returns
    -------
    dict
        JSON-safe transition with a UTC timestamp and fit selectors.
    """
    if "time_utc" not in event or "state" not in event:
        raise ValueError("Training event needs time_utc and state.")
    job = event.get("job", {})
    values = {**job, **event}
    record = {
        "time_utc": datetime.fromtimestamp(
            values["time_utc"], timezone.utc,
        ).isoformat(),
        "campaign": root.name,
    }
    record.update({
        key: values[key] for key in FIELDS if key in values
    })
    return record


def append_training_event(root: Path, event: dict) -> None:
    """Append one transition safely across coordinator processes.

    Parameters
    ----------
    root : Path
        Fit or search artifact root.
    event : dict
        Status transition already saved in the artifact status history.
    """
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.write(json.dumps(_record(root, event)) + "\n")
            stream.flush()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def import_training_events(roots: list[Path]) -> int:
    """Import saved transitions once, preserving existing log entries.

    Parameters
    ----------
    roots : list[Path]
        Fit or search roots with ``status/jobs.jsonl`` histories.

    Returns
    -------
    int
        Number of previously absent transitions appended to the log.
    """
    events = []
    for root in roots:
        history = root / "status" / "jobs.jsonl"
        if not history.is_file():
            raise FileNotFoundError(
                f"Training status history is missing: {history}."
            )
        for line in history.read_text().splitlines():
            events.append(_record(root, json.loads(line)))
    events.sort(key=lambda row: row["time_utc"])
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.seek(0)
            seen = {line for line in stream.read().splitlines()}
            added = 0
            for event in events:
                line = json.dumps(event)
                if line not in seen:
                    stream.write(line + "\n")
                    seen.add(line)
                    added += 1
            stream.flush()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    return added
