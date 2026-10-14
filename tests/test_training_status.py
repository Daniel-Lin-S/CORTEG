"""Test task-count reconciliation and future failure lifecycle records.

Input: temporary completion markers, coordinator events, and process
identities. Output: assertions only; no real training jobs run.
"""
from __future__ import annotations

import json
import signal
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from infrastructure.training_lifecycle import (
    _owner_alive,
    append_training_event,
    coordinator_scope,
    fit_task,
    migrate_root_log,
    refresh_status,
    register_campaign,
)


class TrainingStatusTests(unittest.TestCase):
    """Verify the local registry and diagnostic log contracts."""

    def test_counts_retry_and_failure_reason(self) -> None:
        """Use markers for completion and live owners for running fits."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            campaign = parent / "study__hash"
            root = parent / "ridge__hash"
            first = root / "runs" / "50pct" / "bp" / "result.json"
            second = root / "runs" / "50pct" / "cc" / "result.json"
            first.parent.mkdir(parents=True)
            first.write_text("{}")
            log = parent / "logs" / "events.jsonl"
            tasks = [
                fit_task(
                    root, marker, model="ridge", subject=subject,
                    fraction=0.5, seed=None,
                ) for marker, subject in (
                    (first, "bp"), (second, "cc"),
                )
            ]
            with patch("infrastructure.training_lifecycle.LOG", log):
                register_campaign(campaign, tasks)
                group = next(iter(refresh_status()["campaigns"][str(
                    campaign
                )]["groups"].values()))
                self.assertEqual(
                    (group["total"], group["completed"],
                     group["pending"], group["running"]),
                    (2, 1, 1, 0),
                )
                with coordinator_scope(campaign, "cpu", ("ridge",)):
                    event = {
                        "time_utc": time.time(), "model": "ridge",
                        "subject": "cc", "fraction": 0.5,
                        "seed": None, "attempt_id": "attempt1",
                    }
                    append_training_event(
                        root, {**event, "state": "running"},
                    )
                    group = next(iter(refresh_status()[
                        "campaigns"][str(campaign)]["groups"].values()))
                    self.assertEqual(group["running"], 1)
                    self.assertEqual(
                        group["active"][0]["subject"], "cc",
                    )
                    attempt_log = parent / "attempt.log"
                    attempt_log.write_text(
                        "Traceback (most recent call last):\n"
                        "RuntimeError: fit failed\n"
                    )
                    append_training_event(root, {
                        **event, "state": "failed", "exit_code": 1,
                        "attempt_log": str(attempt_log),
                    })
                    append_training_event(root, {
                        **event, "state": "queued",
                    })
                    append_training_event(root, {
                        **event, "state": "running",
                        "attempt_id": "attempt2",
                    })
                    second.parent.mkdir(parents=True)
                    second.write_text("{}")
                    append_training_event(root, {
                        **event, "state": "complete",
                        "attempt_id": "attempt2", "exit_code": 0,
                    })
                result = refresh_status()["campaigns"][str(campaign)]
                self.assertEqual(result["state"], "complete")
                errors = [json.loads(line) for line in (
                    log.parent / campaign.name / "errors.jsonl"
                ).read_text().splitlines()]
                self.assertEqual(errors[0]["reason"], (
                    "RuntimeError: fit failed"
                ))
                self.assertEqual(
                    [row["event"] for row in errors], [
                        "fit_failed", "retry_queued", "retry_started",
                        "retry_succeeded",
                    ],
                )
                self.assertFalse(errors[0]["queued_for_retry"])
                self.assertTrue(errors[1]["queued_for_retry"])

    def test_other_pid_namespace_does_not_mark_owner_dead(self) -> None:
        """Treat an invisible host coordinator as unverifiable."""
        owner = {
            "pid": 12345,
            "start_ticks": "42",
            "pid_namespace": "pid:[host]",
        }
        with patch(
            "infrastructure.training_lifecycle._pid_namespace",
            return_value="pid:[sandbox]",
        ), patch(
            "infrastructure.training_lifecycle._process_start",
            return_value=None,
        ):
            self.assertIsNone(_owner_alive(owner))

    def test_interrupted_coordinator_reconciles_once(self) -> None:
        """A vanished process leaves pending work and one error record."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            campaign = parent / "study__hash"
            root = parent / "model__hash"
            marker = root / "result.json"
            log = parent / "logs" / "events.jsonl"
            with patch("infrastructure.training_lifecycle.LOG", log):
                task = fit_task(
                    root, marker, model="model", subject="bp",
                    fraction=0.25, seed=0,
                )
                register_campaign(campaign, [task])
                registry = log.parent / campaign.name / "registry.json"
                saved = json.loads(registry.read_text())
                entry = saved["campaigns"][str(campaign)]
                entry["owners"]["dead"] = {
                    "pid": 999999999, "start_ticks": "0",
                    "kind": "gpu", "models": ["model"],
                }
                entry["tasks"][task["id"]]["active_owner"] = "dead"
                registry.write_text(json.dumps(saved))
                first = refresh_status()
                second = refresh_status()
                self.assertEqual(first["campaigns"][str(
                    campaign
                )]["state"], "interrupted")
                self.assertEqual(second["campaigns"][str(
                    campaign
                )]["groups"][next(iter(second["campaigns"][str(
                    campaign
                )]["groups"]))]["pending"], 1)
                errors = (
                    log.parent / campaign.name / "errors.jsonl"
                ).read_text()
                self.assertEqual(
                    errors.count("campaign_interrupted"), 1,
                )

    def test_signal_exit_and_hpo_groups(self) -> None:
        """A handled signal records unfinished work by candidate stage."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            campaign = parent / "search__hash"
            root = parent / "fit__hash"
            log = parent / "logs" / "events.jsonl"
            tasks = [
                fit_task(
                    root, root / name / "result.json",
                    candidate=name, mode=mode, fraction=0.5,
                    subject="bp", seed=0,
                ) for name, mode in (
                    ("candidate_a", "screen"),
                    ("candidate_b", "final"),
                )
            ]
            with patch("infrastructure.training_lifecycle.LOG", log):
                register_campaign(campaign, tasks)
                status = refresh_status()["campaigns"][str(campaign)]
                self.assertEqual(len(status["groups"]), 2)
                with self.assertRaises(KeyboardInterrupt):
                    with coordinator_scope(campaign, "hpo"):
                        signal.raise_signal(signal.SIGTERM)
                status = refresh_status()["campaigns"][str(campaign)]
                self.assertEqual(status["state"], "interrupted")
                self.assertTrue(all(
                    group["pending"] == 1
                    for group in status["groups"].values()
                ))
                errors = [json.loads(line) for line in (
                    log.parent / campaign.name / "errors.jsonl"
                ).read_text().splitlines()]
                self.assertEqual(
                    errors[0]["event"], "campaign_interrupted",
                )
                self.assertEqual(errors[0]["unfinished_tasks"], 2)

    def test_root_event_migration_preserves_repeated_lines(self) -> None:
        """Move matching history and retain an unmatched source."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            campaign = parent / "study__hash"
            root = parent / "model__hash"
            log = parent / "logs" / "events.jsonl"
            task = fit_task(
                root, root / "result.json", model="model",
                subject="bp", fraction=0.5, seed=0,
            )
            with patch("infrastructure.training_lifecycle.LOG", log):
                register_campaign(campaign, [task])
                record = json.dumps({
                    "campaign": root.name, "state": "queued",
                })
                log.write_text(record + "\n" + record + "\n")
                self.assertEqual(
                    migrate_root_log(campaign, [root]), 2,
                )
                destination = (
                    log.parent / campaign.name / "events.jsonl"
                )
                self.assertEqual(
                    destination.read_text().splitlines(),
                    [record, record],
                )
                self.assertFalse(log.exists())
                log.write_text(json.dumps({
                    "campaign": "unrelated__hash",
                    "state": "running",
                }) + "\n")
                with self.assertRaisesRegex(
                    ValueError, "another campaign",
                ):
                    migrate_root_log(campaign, [root])
                self.assertTrue(log.is_file())

    def test_concurrent_events_keep_valid_snapshot(self) -> None:
        """Concurrent fit events do not tear JSON or lose history."""
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "logs" / "events.jsonl"
            root = Path(temporary) / "fit__hash"
            with patch("infrastructure.training_lifecycle.LOG", log):
                def write(index: int) -> None:
                    append_training_event(root, {
                        "time_utc": time.time(), "state": "queued",
                        "subject": f"subject{index}",
                    })
                with ThreadPoolExecutor(max_workers=4) as workers:
                    list(workers.map(write, range(12)))
                folder = log.parent / root.name
                self.assertEqual(
                    len((folder / "events.jsonl").read_text().splitlines()),
                    12,
                )
                json.loads((folder / "status.json").read_text())


    def test_shared_fit_root_keeps_campaign_events_separate(self) -> None:
        """Route a fit transition only to its registered campaign."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "model__hash"
            log = parent / "logs" / "events.jsonl"
            first = parent / "first__hash"
            second = parent / "second__hash"
            with patch("infrastructure.training_lifecycle.LOG", log):
                for campaign, subject in (
                    (first, "bp"), (second, "cc"),
                ):
                    task = fit_task(
                        root, root / subject / "result.json",
                        model="model", subject=subject,
                        fraction=0.5, seed=0,
                    )
                    register_campaign(campaign, [task])
                append_training_event(root, {
                    "time_utc": time.time(),
                    "state": "queued", "model": "model",
                    "subject": "bp", "fraction": 0.5, "seed": 0,
                })
                first_events = (
                    log.parent / first.name / "events.jsonl"
                ).read_text().splitlines()
                second_events = (
                    log.parent / second.name / "events.jsonl"
                ).read_text().splitlines()
                self.assertEqual(len(first_events), 1)
                self.assertEqual(len(second_events), 0)

if __name__ == "__main__":
    unittest.main()
