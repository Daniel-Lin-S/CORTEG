"""Maintain local training events, progress, and failure diagnostics.

Input: registered fit tasks with completion-marker paths and coordinator
transitions. Output: ignored logs/training/<campaign>/events.jsonl, status.json,
errors.jsonl, and registry.json. Fit artifacts are read, never rewritten.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import tempfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "logs" / "training" / "events.jsonl"
FIELDS = (
    "state", "model", "subject", "unit", "fraction", "seed",
    "candidate", "mode", "kind", "exit_code", "error",
    "attempt_id", "attempt_log", "delay",
)
SELECTORS = (
    "model", "subject", "unit", "fraction", "seed",
    "candidate", "mode", "delay",
)
TRACEBACK_LINE = re.compile(r"^[\w.]+(?:Error|Exception|Exit): .+")


def _now() -> str:
    """Return the current UTC time in an unambiguous format."""
    return datetime.now(timezone.utc).isoformat()


def _folder(campaign: Path) -> Path:
    """Locate one semantic campaign's private diagnostic directory."""
    return LOG.parent / campaign.name


def _paths(folder: Path) -> tuple[Path, Path, Path, Path]:
    """Locate one campaign's event, registry, status, and error files."""
    return (
        folder / "events.jsonl", folder / "registry.json",
        folder / "status.json", folder / "errors.jsonl",
    )


def _record(root: Path, event: dict) -> dict:
    """Flatten one fit event for its campaign history."""
    if "time_utc" not in event or "state" not in event:
        raise ValueError("Training event needs time_utc and state.")
    values = {**event.get("job", {}), **event}
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


def _atomic_json(path: Path, value: dict) -> None:
    """Replace a JSON snapshot after its complete contents are durable."""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _load_registry(folder: Path) -> dict:
    """Read a campaign registry without inventing missing work."""
    registry = _paths(folder)[1]
    if not registry.is_file():
        return {"campaigns": {}}
    return json.loads(registry.read_text())


def _task_key(root: Path, selectors: dict) -> str:
    """Identify a fit independently of its attempt and device."""
    value = {
        "fit_root": str(root.resolve()),
        **{key: selectors[key] for key in SELECTORS if key in selectors},
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def fit_task(root: Path, marker: Path, **selectors: object) -> dict:
    """Describe one expected fit and its authoritative completion marker.

    Parameters
    ----------
    root : Path
        Semantic fit or HPO search root.
    marker : Path
        Completion marker for this exact subject, fraction, and seed.
    **selectors : object
        Model, subject or unit, fraction, seed, and optional HPO fields.

    Returns
    -------
    dict
        JSON-safe task description for the local progress registry.
    """
    if not selectors or not marker.is_absolute():
        raise ValueError(
            "A registered task needs selectors and an absolute marker."
        )
    return {
        "id": _task_key(root, selectors),
        "fit_root": str(root.resolve()),
        "marker": str(marker),
        "selectors": selectors,
        "active_owner": None,
    }


def attempt_log_path(root: Path, selectors: dict,
                     attempt_id: str) -> Path:
    """Reserve a stable diagnostic file outside quarantined fit folders."""
    if not attempt_id:
        raise ValueError("Attempt ID must be nonempty.")
    folder = root / "status" / "attempts" / _task_key(root, selectors)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{attempt_id}.log"


def _process_start(pid: int) -> str | None:
    """Read Linux process start ticks to guard against PID reuse."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    except PermissionError:
        return "unknown"
    return stat.rsplit(") ", 1)[1].split()[19]


def _pid_namespace(pid: int) -> str | None:
    """Read a Linux PID namespace identity when it is visible."""
    try:
        return os.readlink(f"/proc/{pid}/ns/pid")
    except (FileNotFoundError, PermissionError):
        return None


def _owner_alive(owner: dict) -> bool | None:
    """Classify a coordinator only when its identity is verifiable."""
    expected_namespace = owner.get("pid_namespace")
    if expected_namespace is not None and (
        expected_namespace != _pid_namespace(os.getpid())
    ):
        return None
    current = _process_start(owner["pid"])
    if current == "unknown":
        return None
    return current is not None and current == owner["start_ticks"]


def _group(task: dict) -> str:
    """Group task counts by semantic fit and fraction or HPO stage."""
    selectors = task["selectors"]
    parts = [Path(task["fit_root"]).name]
    for key in ("candidate", "mode", "model", "fraction"):
        if key in selectors:
            parts.append(f"{key}={selectors[key]}")
    return "/".join(parts)


def _snapshot(registry: dict) -> dict:
    """Count registered fits from markers and validated active owners."""
    campaigns = {}
    for path, campaign in registry["campaigns"].items():
        groups: dict[str, dict] = {}
        for task in campaign["tasks"].values():
            key = _group(task)
            group = groups.setdefault(key, {
                "total": 0, "completed": 0, "pending": 0,
                "running": 0, "active": [],
            })
            group["total"] += 1
            if Path(task["marker"]).is_file():
                group["completed"] += 1
                continue
            owner = campaign["owners"].get(task["active_owner"])
            if owner is not None and _owner_alive(owner) is True:
                group["running"] += 1
                group["active"].append(task["selectors"])
            else:
                group["pending"] += 1
        for group in groups.values():
            if group["total"] != sum(
                group[key] for key in ("completed", "pending", "running")
            ):
                raise ValueError("Training task counts do not reconcile.")
        campaigns[path] = {
            "name": Path(path).name,
            "state": (
                "complete" if all(
                    row["completed"] == row["total"]
                    for row in groups.values()
                ) else "running" if any(
                    row["running"] for row in groups.values()
                ) else (
                    campaign["state"] if campaign.get("state") in (
                        "interrupted", "incomplete",
                    ) else "pending"
                )
            ),
            "groups": groups,
        }
    return {"updated_at_utc": _now(), "campaigns": campaigns}


def _save(folder: Path, registry: dict) -> None:
    """Persist one campaign's registry and concise snapshot."""
    _, registry_path, status_path, error_path = _paths(folder)
    _atomic_json(registry_path, registry)
    _atomic_json(status_path, _snapshot(registry))
    error_path.touch(exist_ok=True)


def _error(folder: Path, record: dict) -> None:
    """Append a failure or retry transition under a campaign lock."""
    path = _paths(folder)[3]
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()


def _reason(event: dict) -> str:
    """Extract only an attributable exception or terminating signal."""
    if event.get("error"):
        return str(event["error"])
    exit_code = event.get("exit_code")
    if isinstance(exit_code, int) and exit_code < 0:
        try:
            return f"terminated by {signal.Signals(-exit_code).name}"
        except ValueError:
            pass
    path = event.get("attempt_log")
    if path and Path(path).is_file():
        lines = Path(path).read_text(errors="replace").splitlines()
        if any(line.startswith("Traceback (most recent call last)")
               for line in lines):
            for line in reversed(lines):
                if TRACEBACK_LINE.match(line):
                    return line
    return "reason unavailable"


def _last_failure(folder: Path, task_id: str) -> str | None:
    """Find the most recent failure attempt in one campaign."""
    path = _paths(folder)[3]
    if not path.is_file():
        return None
    for line in reversed(path.read_text().splitlines()):
        row = json.loads(line)
        if row.get("task_id") == task_id and (
            row.get("event") == "fit_failed"
        ):
            return row.get("attempt_id")
    return None


def _update_tasks(registry: dict, root: Path, event: dict) -> None:
    """Link a fit transition to registered work and its active owner."""
    values = {**event.get("job", {}), **event}
    task_id = _task_key(root, values)
    for campaign in registry["campaigns"].values():
        task = campaign["tasks"].get(task_id)
        if task is None:
            continue
        if event["state"] == "running":
            for owner_id, owner in campaign["owners"].items():
                if owner["pid"] == os.getpid() and (
                    not owner["models"] or
                    values.get("model") in owner["models"]
                ):
                    task["active_owner"] = owner_id
                    break
        elif event["state"] in ("complete", "failed", "skipped"):
            task["active_owner"] = None


def _matching_folders(root: Path, event: dict) -> list[Path]:
    """Find campaigns owning a fit, or its root for general events."""
    values = {**event.get("job", {}), **event}
    task_id = _task_key(root, values)
    folders = []
    fit_states = {"queued", "running", "complete", "failed", "skipped"}
    for registry_path in LOG.parent.glob("*/registry.json"):
        registry = json.loads(registry_path.read_text())
        for campaign in registry["campaigns"].values():
            if task_id in campaign["tasks"]:
                folders.append(registry_path.parent)
                break
            if event.get("state") not in fit_states and any(
                task["fit_root"] == str(root.resolve())
                for task in campaign["tasks"].values()
            ):
                folders.append(registry_path.parent)
                break
    return folders or [_folder(root)]


def _append_in_folder(folder: Path, root: Path, event: dict) -> None:
    """Update one campaign's event, error, and snapshot files."""
    folder.mkdir(parents=True, exist_ok=True)
    event_path = _paths(folder)[0]
    with event_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.write(json.dumps(_record(root, event)) + "\n")
            stream.flush()
            registry = _load_registry(folder)
            _update_tasks(registry, root, event)
            values = {**event.get("job", {}), **event}
            task_id = _task_key(root, values)
            previous = _last_failure(folder, task_id)
            if event["state"] == "failed":
                _error(folder, {
                    "time_utc": _now(), "event": "fit_failed",
                    "campaign": folder.name, "task_id": task_id,
                    "selectors": {
                        key: values[key] for key in SELECTORS
                        if key in values
                    },
                    "attempt_id": event.get("attempt_id"),
                    "retry_of": previous,
                    "exit_code": event.get("exit_code"),
                    "reason": _reason(event),
                    "attempt_log": event.get("attempt_log"),
                    "queued_for_retry": False,
                })
            elif event["state"] == "queued":
                if previous is not None:
                    _error(folder, {
                        "time_utc": _now(), "event": "retry_queued",
                        "campaign": folder.name, "task_id": task_id,
                        "retry_of": previous,
                        "queued_for_retry": True,
                    })
            elif event["state"] in ("running", "complete") and (
                previous is not None and event.get("attempt_id") != previous
            ):
                _error(folder, {
                    "time_utc": _now(),
                    "event": (
                        "retry_started" if event["state"] == "running"
                        else "retry_succeeded"
                    ),
                    "campaign": folder.name, "task_id": task_id,
                    "retry_of": previous,
                    "attempt_id": event.get("attempt_id"),
                    "queued_for_retry": False,
                })
            _save(folder, registry)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def append_training_event(root: Path, event: dict) -> None:
    """Append a transition to each registered campaign using the fit."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for folder in _matching_folders(root, event):
        _append_in_folder(folder, root, event)


def register_campaign(campaign: Path, tasks: list[dict]) -> None:
    """Register expected fits in one semantic campaign folder."""
    if not tasks or len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Expected nonempty, unique campaign tasks.")
    folder = _folder(campaign)
    folder.mkdir(parents=True, exist_ok=True)
    event_path = _paths(folder)[0]
    with event_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            registry = _load_registry(folder)
            if registry["campaigns"] and (
                str(campaign.resolve()) not in registry["campaigns"]
            ):
                raise ValueError(
                    f"Campaign log folder name conflicts: {folder}."
                )
            entry = registry["campaigns"].setdefault(
                str(campaign.resolve()), {
                    "tasks": {}, "owners": {}, "state": "pending",
                },
            )
            for task in tasks:
                previous = entry["tasks"].get(task["id"])
                if previous is not None and (
                    previous["marker"] != task["marker"]
                ):
                    raise ValueError(
                        f"Task completion marker changed: {task['id']}."
                    )
                if previous is None:
                    entry["tasks"][task["id"]] = task
            _reconcile(folder, registry)
            _save(folder, registry)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _reconcile(folder: Path, registry: dict) -> None:
    """Record abandoned coordinators only when their process is gone."""
    for path, campaign in registry["campaigns"].items():
        for owner_id, owner in list(campaign["owners"].items()):
            if _owner_alive(owner) is not False:
                continue
            unfinished = sum(
                not Path(task["marker"]).is_file()
                for task in campaign["tasks"].values()
                if not owner["models"] or (
                    task["selectors"].get("model") in owner["models"]
                )
            )
            if unfinished:
                _error(folder, {
                    "time_utc": _now(),
                    "event": "campaign_interrupted",
                    "campaign": Path(path).name,
                    "reason": (
                        "coordinator process exited without a final event"
                    ),
                    "unfinished_tasks": unfinished,
                    "queued_for_retry": False,
                })
                campaign["state"] = "interrupted"
            for task in campaign["tasks"].values():
                if task["active_owner"] == owner_id:
                    task["active_owner"] = None
            del campaign["owners"][owner_id]


def refresh_status() -> dict:
    """Reconcile all registered campaigns and return their snapshots."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    combined = {"updated_at_utc": _now(), "campaigns": {}}
    for registry_path in sorted(LOG.parent.glob("*/registry.json")):
        folder = registry_path.parent
        with _paths(folder)[0].open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                registry = _load_registry(folder)
                _reconcile(folder, registry)
                _save(folder, registry)
                combined["campaigns"].update(
                    _snapshot(registry)["campaigns"]
                )
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
    return combined


def _finish_owner(campaign: Path, owner_id: str,
                  error: BaseException | None) -> None:
    """Close a coordinator and record unfinished or failed work."""
    folder = _folder(campaign)
    with _paths(folder)[0].open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            registry = _load_registry(folder)
            entry = registry["campaigns"][str(campaign.resolve())]
            owner = entry["owners"].pop(owner_id, None)
            if owner is None:
                _error(folder, {
                    "time_utc": _now(),
                    "event": "coordinator_owner_missing",
                    "campaign": campaign.name,
                    "reason": (
                        "coordinator owner record was absent at finalization"
                    ),
                    "queued_for_retry": False,
                })
                owner = {"models": []}
            unfinished = sum(
                not Path(task["marker"]).is_file()
                for task in entry["tasks"].values()
                if not owner["models"] or (
                    task["selectors"].get("model") in owner["models"]
                )
            )
            if error is not None or unfinished:
                event = (
                    "campaign_interrupted" if isinstance(
                        error, (KeyboardInterrupt, SystemExit)
                    ) else "coordinator_failed" if error is not None
                    else "campaign_incomplete"
                )
                _error(folder, {
                    "time_utc": _now(), "event": event,
                    "campaign": campaign.name,
                    "reason": (
                        f"{type(error).__name__}: {error}"
                        if error is not None else
                        "coordinator exited with unfinished tasks"
                    ),
                    "unfinished_tasks": unfinished,
                    "queued_for_retry": False,
                })
                entry["state"] = (
                    "interrupted" if event == "campaign_interrupted"
                    else "incomplete"
                )
            else:
                entry["state"] = "complete"
            for task in entry["tasks"].values():
                if task["active_owner"] == owner_id:
                    task["active_owner"] = None
            _save(folder, registry)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def coordinator_scope(campaign: Path, kind: str,
                      models: tuple[str, ...] = ()) -> Iterator[None]:
    """Track a coordinator through normal, failed, and catchable exits."""
    start = _process_start(os.getpid())
    if start is None or start == "unknown":
        raise RuntimeError("Cannot validate coordinator process identity.")
    owner_id = uuid.uuid4().hex
    folder = _folder(campaign)
    with _paths(folder)[0].open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            registry = _load_registry(folder)
            entry = registry["campaigns"][str(campaign.resolve())]
            _reconcile(folder, registry)
            entry["owners"][owner_id] = {
                "pid": os.getpid(), "start_ticks": start,
                "pid_namespace": _pid_namespace(os.getpid()),
                "kind": kind, "models": list(models),
            }
            entry["state"] = "running"
            _save(folder, registry)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    old_handlers = {}

    def _stop(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(signal.Signals(signum).name)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _stop)
        try:
            yield
        except BaseException as error:
            _finish_owner(campaign, owner_id, error)
            raise
        else:
            _finish_owner(campaign, owner_id, None)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def import_training_events(roots: list[Path]) -> int:
    """Import saved fit transitions into matching campaign folders."""
    added = 0
    for root in roots:
        history = root / "status" / "jobs.jsonl"
        if not history.is_file():
            raise FileNotFoundError(
                f"Training status history is missing: {history}."
            )
        for line in history.read_text().splitlines():
            event = json.loads(line)
            record = json.dumps(_record(root, event))
            for folder in _matching_folders(root, event):
                folder.mkdir(parents=True, exist_ok=True)
                with _paths(folder)[0].open("a+") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    try:
                        stream.seek(0)
                        if record not in stream.read().splitlines():
                            stream.write(record + "\n")
                            added += 1
                    finally:
                        fcntl.flock(stream, fcntl.LOCK_UN)
    return added


def migrate_root_log(campaign: Path, fit_roots: list[Path]) -> int:
    """Move a verified legacy root log into one registered campaign.

    Parameters
    ----------
    campaign : Path
        Campaign already registered under its semantic folder.
    fit_roots : list[Path]
        Fit roots whose historical events belong to that campaign.

    Returns
    -------
    int
        Number of previously absent records copied.
    """
    if not LOG.is_file():
        return 0
    folder = _folder(campaign)
    if not _paths(folder)[1].is_file():
        raise FileNotFoundError(
            f"Register campaign before log migration: {folder}."
        )
    source_lines = LOG.read_text().splitlines()
    allowed = {root.name for root in fit_roots}
    for line in source_lines:
        record = json.loads(line)
        if record.get("campaign") not in allowed:
            raise ValueError(
                "Legacy training log contains another campaign; "
                f"kept source at {LOG}."
            )
    old_registry = LOG.parent / "registry.json"
    old_status = LOG.parent / "status.json"
    old_errors = LOG.parent / "errors.jsonl"
    if old_registry.is_file():
        saved = json.loads(old_registry.read_text())
        if set(saved["campaigns"]) != {str(campaign.resolve())}:
            raise ValueError(
                "Legacy registry contains another campaign; "
                f"kept source at {old_registry}."
            )
    if old_errors.is_file() and old_errors.read_text().strip():
        raise ValueError(
            f"Legacy error log is nonempty; kept source at {old_errors}."
        )
    destination = _paths(folder)[0]
    with destination.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.seek(0)
            seen = Counter(stream.read().splitlines())
            needed = Counter(source_lines)
            added = 0
            for line in source_lines:
                if seen[line] < needed[line]:
                    stream.write(line + "\n")
                    seen[line] += 1
                    added += 1
            stream.flush()
            os.fsync(stream.fileno())
            if any(seen[line] < count for line, count in needed.items()):
                raise RuntimeError(
                    "Campaign event migration did not preserve all lines."
                )
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    LOG.unlink()
    for path in (old_registry, old_status, old_errors):
        path.unlink(missing_ok=True)
    return added
