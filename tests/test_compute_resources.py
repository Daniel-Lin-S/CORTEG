"""Test reusable local compute configuration validation.

Input: temporary JSON settings. Output: assertions only.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infrastructure.compute import load_resources


class ComputeResourceTests(unittest.TestCase):
    """Validate positive local worker and GPU thresholds."""

    def test_local_resource_validation(self) -> None:
        """Worker and device thresholds require positive integers."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resources.json"
            settings = {
                "gpu_workers": 4,
                "cpu_workers": 4,
                "cpu_threads_per_worker": 4,
                "min_gpu_free_mib": 16000,
            }
            path.write_text(json.dumps(settings))
            self.assertEqual(load_resources(path), settings)
            settings["gpu_workers"] = 0
            path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "gpu_workers"):
                load_resources(path)
