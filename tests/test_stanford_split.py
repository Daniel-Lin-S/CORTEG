"""Validate Stanford duration split rules against prepared provenance.

Input: the repository's Stanford split JSON and temporary resolved config.
Output: assertions on exclusive train endpoints and selected rule names.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.stanford_split import load_split_config, split_boundary


class StanfordSplitTests(unittest.TestCase):
    """Keep cleaning and feature split boundaries aligned."""

    def test_duration_rules_match_original_extractor(self) -> None:
        """Use the fixed-time rule at 600 s and whole-sample parts below."""
        split = load_split_config()
        self.assertEqual(
            split_boundary(600_000, 1_000, split),
            (400_000, "at_or_above_cutoff"),
        )
        self.assertEqual(
            split_boundary(465_840, 1_000, split),
            (310_560, "below_cutoff"),
        )
        self.assertEqual(
            split_boundary(178_961, 1_000, split),
            (119_306, "below_cutoff"),
        )

    def test_split_settings_load_from_prepared_provenance(self) -> None:
        """Read the resolved split rather than newer repository defaults."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolved = {"split": load_split_config()}
            resolved["split"]["duration_cutoff_seconds"] = 700
            (root / "resolved_config.json").write_text(
                json.dumps(resolved)
            )
            split = load_split_config(root)
            self.assertEqual(
                split_boundary(610_040, 1_000, split),
                (406_692, "below_cutoff"),
            )
