"""Verify CNN-GRU search enumeration, isolation, selection, and resume.

Input is synthetic prepared training arrays and temporary search artifacts.
Output is assertions only; test-target files and GPUs are not required.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.fingerflex_campaign import (
    _load_arrays, parameter_groups,
)
from models.temporal_spatial_cnn_gru import TemporalSpatialCNNGRU
from data.fingerflex_cnn_gru_hpo_adapter import (
    DEFAULT_ARCHITECTURE, architecture_grid, architecture_key,
    pending_jobs, select,
)

GRID = {
    "n_convolution_layers": [2, 3, 4],
    "conv_stride": [2, 3],
    "drop_out": [0.0, 0.1, 0.3, 0.5],
    "exclude": [{
        "n_convolution_layers": 4, "conv_stride": 3,
    }],
}


class CnnGruHpoTests(unittest.TestCase):
    """Protect validation-only selection and complete-fit reuse."""

    def test_grid_and_shapes(self) -> None:
        """All requested combinations produce five finite outputs."""
        architectures = architecture_grid({"grid": GRID})
        self.assertEqual(len(architectures), 20)
        self.assertEqual(len({
            architecture_key(value) for value in architectures
        }), 20)
        for architecture in architectures:
            model = TemporalSpatialCNNGRU(
                input_channels=2, input_length=200,
                outputs=5, **architecture,
            )
            model.eval()
            with torch.no_grad():
                result = model(torch.zeros(1, 2, 200))
            self.assertEqual(result.shape, (1, 5))
            self.assertTrue(torch.isfinite(result).all())

    def test_default_parameters_match_explicit_constructor(self) -> None:
        """Passing the default architecture preserves model behavior."""
        torch.manual_seed(3)
        implicit = TemporalSpatialCNNGRU(2, 200, 5)
        torch.manual_seed(3)
        explicit = TemporalSpatialCNNGRU(
            2, 200, 5, **DEFAULT_ARCHITECTURE,
        )
        self.assertEqual(
            list(implicit.state_dict()), list(explicit.state_dict()),
        )
        for key, value in implicit.state_dict().items():
            torch.testing.assert_close(
                value, explicit.state_dict()[key], rtol=0, atol=0,
            )

    def test_cnn_groups_do_not_require_adapter_rate(self) -> None:
        """CNN-GRU uses its base rate without CORTEG-only settings."""
        model = torch.nn.Linear(2, 1)
        groups, names = parameter_groups(
            model, "cnn_gru", 1.0, {"base_lr": 0.001},
        )
        self.assertEqual([group["lr"] for group in groups], [0.001])
        self.assertEqual(set(names), {"base"})

    def test_screening_never_reads_test_files(self) -> None:
        """Validation preparation works without any test arrays."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = root / "bp" / "common"
            feature = root / "bp" / "corteg" / "0"
            common.mkdir(parents=True)
            feature.mkdir(parents=True)
            train_y = np.arange(500, dtype=np.float32).reshape(100, 5)
            train_x = np.arange(
                100 * 2 * 200 * 2, dtype=np.float32,
            ).reshape(100, 2, 200, 2)
            np.save(common / "train_y.npy", train_y)
            np.save(common / "train_indices.npy", np.arange(100))
            np.save(feature / "train_hi.npy", train_x)
            output = root / "output"
            output.mkdir()
            arrays, boundaries = _load_arrays(
                root, "bp", "cnn_gru", 0, 1.0, output,
                include_test=False,
            )
            self.assertEqual(len(arrays["x"]), 2)
            self.assertEqual(len(arrays["y"]), 2)
            self.assertNotIn("test_grid_sha256", boundaries)
            self.assertNotIn("target_values_sha256", boundaries)

    def test_selection_tie_break_and_freeze(self) -> None:
        """Coordinator selects one setting without reading fit results."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = "depth2_stride2_drop0.0"
            second = "depth3_stride3_drop0.3"
            (root / "index.json").write_text(json.dumps({
                "candidates": {
                    first: str(root / first),
                    second: str(root / second),
                },
                "architectures": {
                    first: {
                        "n_convolution_layers": 2,
                        "conv_stride": 2, "drop_out": 0.0,
                    },
                    second: DEFAULT_ARCHITECTURE,
                },
            }))
            with patch(
                "data.fingerflex_cnn_gru_hpo_adapter.selection_scores",
                return_value={first: 0.4, second: 0.4},
            ):
                winner = select(root, {"selection": {"seed": 0}})
            self.assertEqual(winner["candidate"], first)
            with patch(
                "data.fingerflex_cnn_gru_hpo_adapter.selection_scores",
                side_effect=AssertionError("Selection was recomputed."),
            ):
                self.assertEqual(
                    select(root, {"selection": {}}), winner,
                )

    def test_pending_screen_skips_default_and_completed(self) -> None:
        """A resumed search schedules only missing candidate fits."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "baseline"
            other = root / "other"
            default = architecture_key(DEFAULT_ARCHITECTURE)
            candidate = "depth2_stride2_drop0.0"
            (root / "index.json").write_text(json.dumps({
                "candidates": {
                    default: str(base), candidate: str(other),
                },
                "architectures": {
                    default: DEFAULT_ARCHITECTURE,
                    candidate: {
                        "n_convolution_layers": 2,
                        "conv_stride": 2, "drop_out": 0.0,
                    },
                },
            }))
            spec = {"subjects": ["bp", "jp"]}
            path = (
                other / "runs" / "100pct" / "cnn_gru" / "bp"
                / "seed0"
            )
            path.mkdir(parents=True)
            (path / "validation_result.json").write_text("{}")
            pending = pending_jobs(root, spec, "screen")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["subject"], "jp")


if __name__ == "__main__":
    unittest.main()
