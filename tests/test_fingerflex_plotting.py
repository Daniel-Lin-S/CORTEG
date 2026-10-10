"""Check saved-result plot extraction and significance annotations.

Input: temporary synthetic report/provenance JSON objects with two subjects,
two models, three fractions, and three metrics.
Output: assertions and temporary PNG files; no GPU or raw data required.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plotting.fingerflex import load_results, plot_aggregate
from plotting.regression import draw_significance_bracket

FRACTIONS = (0.1, 0.25, 1.0)
METRICS = ("pearson_r", "mse", "r2")
SUBJECTS = ("bp", "jp")
MODELS = ("corteg", "hilo")


def _fixture(root: Path) -> Path:
    """Write a complete minimal saved-report fixture."""
    reports = root / "reports"
    provenance = root / "provenance"
    reports.mkdir()
    provenance.mkdir()
    (provenance / "resolved_config.json").write_text(json.dumps({
        "subjects": SUBJECTS, "models": MODELS,
        "fractions": FRACTIONS, "seeds": (0, 1, 2),
    }))
    fractions = {}
    for index, fraction in enumerate(FRACTIONS):
        subjects = {}
        for subject_index, subject in enumerate(SUBJECTS):
            subjects[subject] = {}
            for model_index, model in enumerate(MODELS):
                subjects[subject][model] = {
                    "seeds": 3,
                    "mean": {
                        "pearson_r": 0.2 + index * 0.1
                        + model_index * 0.02 + subject_index * 0.01,
                        "mse": 1.4 - index * 0.2
                        + model_index * 0.04 + subject_index * 0.03,
                        "r2": -0.2 + index * 0.1
                        + model_index * 0.02 + subject_index * 0.01,
                    },
                    "std": {metric: 0.02 for metric in METRICS},
                }
        fractions[f"{int(fraction * 100)}pct"] = {
            "subjects": subjects,
            "paired_complete": True,
            "paired_test": {
                metric: {"p_holm": 0.03 if index > 0 else 0.2}
                for metric in METRICS
            },
        }
    summary = reports / "summary.json"
    summary.write_text(json.dumps({"fractions": fractions}))
    return summary


class FingerflexPlottingTests(unittest.TestCase):
    """Protect plot table shape and corrected-test marker behavior."""

    def test_load_and_render_aggregate(self) -> None:
        """Subject-weighted aggregate reads complete finite seed scores."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root)
            table = load_results(root)
            self.assertEqual(table.means.shape, (2, 2, 3, 3))
            np.testing.assert_array_equal(
                table.corrected_p[0], (0.2, 0.03, 0.03)
            )
            output = root / "plots" / "aggregate.png"
            plot_aggregate(table, output)
            self.assertGreater(output.stat().st_size, 1000)

    def test_missing_seed_or_nonfinite_score_fails(self) -> None:
        """Incomplete or invalid input cannot silently create plots."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = _fixture(root)
            summary = json.loads(summary_path.read_text())
            row = summary["fractions"]["25pct"]["subjects"]["bp"]["corteg"]
            row["seeds"] = 2
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "Expected 3"):
                load_results(root)
            row["seeds"] = 3
            row["mean"]["mse"] = float("nan")
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "Invalid mean"):
                load_results(root)

    def test_bracket_requires_corrected_significance(self) -> None:
        """Only Holm p-values below 0.05 produce an annotation."""
        figure, axis = plt.subplots()
        try:
            with patch.object(axis, "annotate") as annotate:
                draw_significance_bracket(axis, 0.25, 0.3, 0.4, 0.2)
                annotate.assert_not_called()
                draw_significance_bracket(axis, 0.25, 0.3, 0.4, 0.03)
                annotate.assert_called_once()
        finally:
            plt.close(figure)
