"""Test the repository-local training lifecycle log.

Input: temporary fit status histories and synthetic transitions. Output:
assertions for flattening, failure visibility, and idempotent import.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infrastructure.training_lifecycle import (
    append_training_event,
    import_training_events,
)


class TrainingLifecycleTests(unittest.TestCase):
    """Keep the overall training log complete and importable."""

    def test_failed_fit_and_history_import(self) -> None:
        """Preserve failure details without duplicating saved events."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "ridge_early__abc123"
            status = root / "status"
            status.mkdir(parents=True)
            log = parent / "logs" / "events.jsonl"
            failed = {
                "time_utc": 1_700_000_000.0,
                "state": "failed",
                "job": {
                    "model": "ridge_early",
                    "subject": "bp",
                    "fraction": 0.5,
                    "seed": None,
                },
                "exit_code": 1,
            }
            completed = {
                **failed,
                "time_utc": failed["time_utc"] + 10,
                "state": "complete",
                "exit_code": 0,
            }
            history = status / "jobs.jsonl"
            history.write_text(
                json.dumps(failed) + "\n"
                + json.dumps(completed) + "\n"
            )
            with patch(
                "infrastructure.training_lifecycle.LOG", log,
            ):
                append_training_event(root, failed)
                self.assertEqual(import_training_events([root]), 1)
                self.assertEqual(import_training_events([root]), 0)
            rows = [
                json.loads(line) for line in log.read_text().splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["state"], "failed")
            self.assertEqual(rows[0]["exit_code"], 1)
            self.assertEqual(rows[0]["campaign"], root.name)
            self.assertEqual(rows[1]["state"], "complete")
            self.assertNotIn("job", rows[0])


if __name__ == "__main__":
    unittest.main()
