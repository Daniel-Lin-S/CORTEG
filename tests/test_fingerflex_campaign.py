"""Synthetic checks for Stanford preparation, metrics, scheduling, and resume.

Input: deterministic in-memory arrays and temporary status folders.
Output: assertions only; no raw dataset or GPU access.
"""
from __future__ import annotations

import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from data.fingerflex_campaign import clean_recording, target_grid
from experiments.fingerflex_campaign import (
    metric_values,
    normalize,
    parameter_groups,
    split_fit_rows,
)
from scripts.fingerflex_campaign import (
    _holm,
    _paired_test,
    _worker,
    job_path,
    jobs_for_subject,
    rank_gpus,
)

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1]
     / "configs" / "fingerflex_campaign.json").read_text()
)


class FingerflexCampaignTests(unittest.TestCase):
    """Check invariants that can fail without full model training."""

    def test_train_only_channel_screen(self) -> None:
        """Test-set outliers cannot change retained channel selection."""
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(4000, 5))
        raw[:2000, 4] *= 10
        first = clean_recording(raw, 2000, CONFIG["preprocessing"])
        raw[2000:, 0] *= 1000
        second = clean_recording(raw, 2000, CONFIG["preprocessing"])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertNotIn(4, first[1])
        np.testing.assert_allclose(first[0].mean(axis=1), 0, atol=1e-6)

    def test_delay_target_grid(self) -> None:
        """Delays shift only windows, and both grids reserve 40 samples."""
        zero = target_grid(8000, 0)
        delayed = target_grid(8000, 40)
        np.testing.assert_array_equal(zero, delayed)
        self.assertTrue(np.all(zero + 40 < 8000))
        self.assertEqual(zero[1] - zero[0], 40)

    def test_fraction_and_normalization_isolation(self) -> None:
        """Val/test changes cannot change selected fit normalization."""
        fit, validation = split_fit_rows(101, 0.25)
        self.assertEqual((fit.stop, validation.start), (22, 91))
        train = np.arange(22 * 3 * 4, dtype=np.float32).reshape(22, 3, 4)
        val = np.ones((10, 3, 4), dtype=np.float32)
        test = np.ones((10, 3, 4), dtype=np.float32)
        _, mean, std = normalize(train, val, test, "corteg")
        _, changed_mean, changed_std = normalize(
            train, val * 1000, test * 1000, "corteg",
        )
        np.testing.assert_array_equal(mean, changed_mean)
        np.testing.assert_array_equal(std, changed_std)

    def test_metrics_edges(self) -> None:
        """R² retains negative scores and rejects undefined values."""
        truth = np.arange(30).reshape(6, 5).astype(float)
        pred = truth + 100
        self.assertLess(metric_values(truth, pred)["overall"]["r2"], 0)
        with self.assertRaisesRegex(ValueError, "Constant test target"):
            metric_values(np.ones((6, 5)), pred)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            metric_values(truth, np.full((6, 5), np.nan))

    def test_parameter_lr_groups(self) -> None:
        """Only adapter parameters receive a fraction-aware boost."""
        class Toy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.channel_adapter = nn.Linear(2, 2)
                self.head = nn.Linear(2, 5)
        model = Toy()
        for fraction, expected in ((1.0, 0.01), (0.25, 0.01),
                                   (0.1, 0.002)):
            groups, _ = parameter_groups(
                model, "corteg", fraction, CONFIG["training"],
            )
            rates = {group["group_name"]: group["lr"]
                     for group in groups}
            self.assertEqual(rates, {"adapter": expected, "base": 0.001})

    def test_gpu_rank_and_resume(self) -> None:
        """Rank by free memory and skip an already completed fit."""
        query = (
            "0, 20000, 90\n1, 30000, 90\n2, 22000, 10\n"
            "3, 21000, 20\n4, 25000, 0"
        )
        self.assertEqual(rank_gpus(query), [1, 4, 2, 3])
        job = jobs_for_subject("mv", CONFIG)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "status").mkdir()
            path = job_path(root, job)
            path.mkdir(parents=True)
            (path / "result.json").write_text("{}")
            work: queue.Queue = queue.Queue()
            work.put(job)
            work.put(None)
            with patch("subprocess.run") as process:
                _worker(0, work, root, threading.Lock())
            process.assert_not_called()
            events = (root / "status" / "jobs.jsonl").read_text()
            self.assertIn('"state": "skipped"', events)

    def test_report_family_and_job_count(self) -> None:
        """Count approved jobs and correct only the three metrics."""
        total = sum(len(jobs_for_subject(subject, CONFIG))
                    for subject in CONFIG["subjects"])
        self.assertEqual(total, 171)
        adjusted = _holm({"pearson_r": 0.01, "mse": 0.2,
                          "r2": 0.03})
        self.assertEqual(adjusted["pearson_r"], 0.03)
        self.assertEqual(adjusted["r2"], 0.06)
        self.assertEqual(_paired_test(np.zeros(9)), 1.0)


if __name__ == "__main__":
    unittest.main()
