"""Check monitor aggregation, outlier handling, and aggregate figure output.

Input: synthetic subject, fraction, seed, and epoch histories.
Output: assertions plus temporary PNG figures; no GPU or raw data required.
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
from matplotlib.figure import Figure

from plotting.fingerflex_training_monitors import (
    _collect_histories,
    _panels_for_monitor,
    generate_training_monitor_plots,
)
from plotting.training_monitors import (
    EpochSummary,
    MonitorCurve,
    MonitorPanel,
    MonitorPlotStyle,
    draw_offscale_arrows,
    plot_monitor_panels,
    robust_y_limits,
)

SUBJECTS = ("s1", "s2")
SEEDS = (0, 1, 2)
FRACTIONS = (1.0, 0.25, 0.1)
LENGTHS = {
    1.0: (4, 5),
    0.25: (3, 4),
    0.1: (2, 2),
}


def _history_row(epoch: int, subject_index: int, seed: int) -> dict:
    """Build one complete epoch monitor row."""
    value = float(epoch + subject_index + seed)
    finger_r2 = [value / 10.0 + index for index in range(5)]
    if epoch == 1 and subject_index == 0 and seed == 0:
        finger_r2[0] = -300.0
    return {
        "epoch": epoch,
        "train_mse": value + 0.5,
        "validation": {
            "overall": {
                "mse": value,
                "pearson_r": value / 10.0,
                "r2": value / 10.0 - 1.0,
            },
            "fingers": {
                "pearson_r": [value / 10.0 + index for index in range(5)],
                "mse": [value + index for index in range(5)],
                "r2": finger_r2,
            },
        },
        "learning_rates": {"base": 0.001, "adapter": 0.01},
        "gradient_norm_mean": value,
        "group_gradient_norm_mean": {
            "base": value / 2.0,
            "adapter": value / 3.0,
        },
        "parameter_groups": {
            "base": {"norm": value, "change": value / 10.0},
            "adapter": {"norm": value / 2.0, "change": value / 5.0},
        },
        "cuda_peak_allocated_mib": 50.0,
    }


def _write_fit_root(root: Path) -> Path:
    """Write a complete synthetic fit root with unequal history lengths."""
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "resolved_config.json").write_text(json.dumps({
        "subjects": SUBJECTS,
        "fractions": FRACTIONS,
        "seeds": SEEDS,
    }))
    for fraction in FRACTIONS:
        percent = int(fraction * 100)
        for subject_index, subject in enumerate(SUBJECTS):
            for seed in SEEDS:
                length = LENGTHS[fraction][subject_index]
                run = root / "runs" / f"{percent}pct" / "corteg" \
                    / subject / f"seed{seed}"
                run.mkdir(parents=True)
                rows = [
                    _history_row(epoch, subject_index, seed)
                    for epoch in range(1, length + 1)
                ]
                (run / "history.jsonl").write_text("".join(
                    json.dumps(row) + "\n" for row in rows
                ))
    return root


class TrainingMonitorTests(unittest.TestCase):
    """Protect common cutoffs, robust ranges, and plot presentation rules."""

    def test_fraction_panels_use_shared_history_cutoffs(self) -> None:
        """Every subject and seed contributes through each fraction cutoff."""
        with tempfile.TemporaryDirectory() as temporary:
            root = _write_fit_root(Path(temporary))
            collected, count, cutoffs = _collect_histories(
                root, list(SUBJECTS), list(FRACTIONS), list(SEEDS),
            )
            self.assertEqual(count, 18)
            self.assertEqual(cutoffs, {1.0: 4, 0.25: 3, 0.1: 2})
            panels = _panels_for_monitor(
                "finger_r2", collected["finger_r2"],
                list(FRACTIONS), list(SUBJECTS), list(SEEDS), cutoffs,
            )
            self.assertEqual(
                [len(panel.curves[0].summary.epoch) for panel in panels],
                [4, 3, 2],
            )
            for panel in panels:
                for curve in panel.curves:
                    np.testing.assert_array_equal(
                        curve.summary.n_subjects,
                        np.full(len(curve.summary.epoch), len(SUBJECTS)),
                    )

    def test_robust_limits_include_curve_and_band_edges(self) -> None:
        """Central limits use pooled means and both SD-band edges."""
        mean = np.zeros(100)
        std = np.ones(100)
        mean[0] = -300.0
        std[0] = 5.0
        summary = EpochSummary(
            epoch=np.arange(1, 101),
            mean=mean,
            std=std,
            n_subjects=np.full(100, 9),
        )
        curve = MonitorCurve("Finger 1", "blue", "-", summary)
        panels = (
            MonitorPanel("100%", (curve,)),
            MonitorPanel("25%", (curve,)),
        )
        limits = robust_y_limits(panels)
        values = np.concatenate((
            summary.mean,
            summary.mean - summary.std,
            summary.mean + summary.std,
        ) * 2)
        expected = np.quantile(values, (0.025, 0.975))
        np.testing.assert_allclose(limits, expected)
        self.assertGreater(limits[0], -300.0)

    def test_offscale_arrows_mark_boundary_crossings(self) -> None:
        """Crossings, including values exactly on a boundary, get arrows."""
        figure, axis = plt.subplots()
        try:
            draw_offscale_arrows(
                axis,
                np.arange(1, 5),
                np.array([0.0, 2.0, 3.0, 0.0]),
                (-2.0, 2.0),
                "blue",
            )
            self.assertEqual(len(axis.texts), 2)
        finally:
            plt.close(figure)

    def test_single_curve_omits_legend_and_uses_subtitle(self) -> None:
        """Single-line figures retain the subtitle and omit legends."""
        summary = EpochSummary(
            epoch=np.arange(1, 4),
            mean=np.array([0.1, 0.2, 0.3]),
            std=np.array([0.01, 0.02, 0.03]),
            n_subjects=np.full(3, 9),
        )
        panels = (MonitorPanel(
            "100% training",
            (MonitorCurve("Overall", "blue", "-", summary),),
        ),)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "single.png"
            with patch.object(Figure, "legend") as legend:
                with patch.object(
                    Figure, "text", autospec=True, side_effect=Figure.text,
                ) as text:
                    plot_monitor_panels(
                        "Monitor", "Value", panels, output,
                        MonitorPlotStyle(), subtitle="N = 9 subjects",
                    )
            legend.assert_not_called()
            self.assertTrue(any(
                call.args[3] == "N = 9 subjects"
                for call in text.call_args_list
            ))
            self.assertGreater(output.stat().st_size, 1000)

    def test_generator_writes_eleven_figures_to_new_default_path(self) -> None:
        """All monitors render under plots without a reports layer."""
        with tempfile.TemporaryDirectory() as temporary:
            root = _write_fit_root(Path(temporary))
            paths = generate_training_monitor_plots(root)
            output = root / "plots" / "training_monitors"
            self.assertEqual(len(paths), 11)
            self.assertTrue(all(path.parent == output for path in paths))
            self.assertFalse((root / "reports").exists())
            self.assertFalse(any("cuda" in path.name for path in paths))
            self.assertTrue(all(path.stat().st_size > 1000 for path in paths))


if __name__ == "__main__":
    unittest.main()
