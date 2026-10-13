"""Exercise dataset-independent HPO and regression training contracts.

Inputs are synthetic candidate metadata and tiny in-memory regression arrays.
Outputs are assertions on selection, resume, history, and checkpoint files.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.hpo import (
    dispatch_jobs, enumerate_grid, pending_jobs, select_candidate,
    select_gpus,
)
from experiments.regression_trainer import train_regressor


class GenericHpoTests(unittest.TestCase):
    """Check HPO logic without a dataset adapter."""

    def test_grid_and_resume_selectors(self) -> None:
        """Generic selectors and exclusions do not assume a dataset."""
        grid = enumerate_grid({
            "width": [4, 8], "dropout": [0.0, 0.2],
            "exclude": [{"width": 8, "dropout": 0.2}],
        })
        self.assertEqual(len(grid), 3)
        candidates = {
            "a": {"parameters": grid[0], "fit_root": "/tmp/a"},
            "b": {"parameters": grid[1], "fit_root": "/tmp/b",
                  "historical": True},
        }
        finished = {("a", "case1", "screen")}
        complete = lambda job: (
            job["candidate"], job["unit"], job["mode"]
        ) in finished
        pending = pending_jobs(
            candidates, ["case1", "case2"],
            {"fractions": [1.0, 0.5], "seeds": [3, 5]},
            None, {"fraction": 1.0, "seed": 3}, complete,
        )
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["unit"], "case2")
        self.assertEqual(pending[0]["seed"], 3)
        final = pending_jobs(
            candidates, ["case1"],
            {"fractions": [1.0, 0.5], "seeds": [3, 5]},
            "a", {"fraction": 1.0, "seed": 3},
            lambda job: False,
        )
        self.assertEqual(len(final), 4)
        self.assertEqual(
            [job["mode"] for job in final].count("score"), 1,
        )

    def test_selection_and_worker_log(self) -> None:
        """Select by validation only and log independent work."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "status").mkdir()
            candidates = {
                "a": {"parameters": {"width": 4},
                      "fit_root": str(root / "a")},
                "b": {"parameters": {"width": 8},
                      "fit_root": str(root / "b")},
            }
            selected = select_candidate(
                root, candidates, {"a": 0.5, "b": 0.5},
                ("width",), {"metric": "correlation"},
            )
            self.assertEqual(selected["candidate"], "a")
            self.assertEqual(select_candidate(
                root, candidates, {"a": 0.1, "b": 0.9},
                ("width",), {},
            ), selected)
            jobs = [{"candidate": "a", "subject": "case1"}]
            with patch(
                "infrastructure.training_lifecycle.LOG",
                root / "training.jsonl",
            ):
                dispatch_jobs(
                    root, jobs, [7],
                    lambda job, device, log: ("complete", 0),
                )
            records = [json.loads(line) for line in (
                root / "status" / "jobs.jsonl"
            ).read_text().splitlines()]
            self.assertEqual(
                [record["state"] for record in records],
                ["queued", "running", "complete"],
            )
            self.assertTrue(all(
                record["device"] == 7 for record in records
                if record["state"] != "queued"
            ))

    def test_gpu_selection_uses_free_memory_and_capacity(self) -> None:
        """Assign highest-free devices without requiring idle GPUs."""
        query = "0, 5000, 90\n1, 9000, 95\n2, 7000, 10\n"
        with patch(
            "experiments.hpo.subprocess.check_output",
            return_value=query,
        ), patch(
            "experiments.hpo.torch.cuda.is_available",
            return_value=True,
        ):
            selected = select_gpus({
                "gpu_workers": 2, "minimum_gpu_free_mib": 4096,
            })
            self.assertEqual(selected, [1, 2])
            with self.assertRaisesRegex(RuntimeError, "at least"):
                select_gpus({
                    "gpu_workers": 3,
                    "minimum_gpu_free_mib": 6000,
                })

    def test_generic_regression_training(self) -> None:
        """A supplied model and loader produce a selected checkpoint."""
        x = torch.linspace(-1, 1, 40)[:, None]
        y = 2 * x
        loader = DataLoader(TensorDataset(x, y), batch_size=10)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam([{
            "params": list(model.parameters()), "lr": 0.02,
            "group_name": "base",
        }])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1,
        )
        initial = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
        }

        def forward(network, batch, device):
            return network(batch[0].to(device)), batch[1].to(device)

        def metrics(truth, prediction):
            correlation = float(np.corrcoef(
                truth[:, 0], prediction[:, 0],
            )[0, 1])
            return {"overall": {"pearson_r": correlation}}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint, norms = train_regressor(
                model, loader, loader, optimizer, scheduler, None,
                torch.device("cpu"), forward, metrics,
                {"base": list(initial)}, initial, output, 3, 2,
            )
            self.assertTrue((output / "best.pt").is_file())
            self.assertEqual(len((
                output / "history.jsonl"
            ).read_text().splitlines()), 3)
            self.assertIn("epoch", checkpoint)
            self.assertIn("base", norms)


if __name__ == "__main__":
    unittest.main()
