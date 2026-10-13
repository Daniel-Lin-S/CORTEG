"""Verify equal-time bins and saved joint-CV ridge fit behavior.

Input: synthetic broadband/HGA windows and targets in temporary folders.
Output: assertions for bin edges, selection, and complete fit artifacts.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.temporal_bins import mean_time_bins
from experiments.fingerflex_binned_ridge import fit_binned_ridge
from experiments.ridge_cv import choose_bin_alpha
from plotting.fingerflex_comparison import (
    BINNING_TITLE, _draw_curves, _study_title,
)
from scripts.fingerflex_linear_identity import (
    binned_ridge_identity, binned_ridge_root,
)


class FingerflexBinningTests(unittest.TestCase):
    """Guard the binning and fit-selection contracts."""

    def test_equal_time_bins_and_ties(self) -> None:
        """Both sample rates span a full second without empty bins."""
        for samples in (128, 200):
            ramp = np.arange(samples, dtype=np.float32)
            values = ramp.reshape(1, 1, samples)
            for bins in (50, 25, 12, 8):
                actual = mean_time_bins(values, bins)[0, 0]
                self.assertEqual(actual.shape, (bins,))
                self.assertTrue(np.diff(actual).min() > 0)
                self.assertEqual(actual[0], np.mean(
                    ramp[:int(np.ceil(samples / bins))],
                ))
                self.assertEqual(actual[-1], np.mean(
                    ramp[int(np.ceil((bins - 1) * samples / bins)):],
                ))
        scores = np.ones((2, 2, 5))
        self.assertEqual(
            choose_bin_alpha(scores, (50, 8), (1.0, 10.0), 5),
            (1, 1),
        )
        with self.assertRaisesRegex(ValueError, "finite CV"):
            choose_bin_alpha(
                np.full((2, 2, 5), np.nan),
                (50, 8), (1.0, 10.0), 5,
            )

    def test_plot_names_shared_fusion_in_title(self) -> None:
        """Only binning status differs between the two legend entries."""
        index = {
            "fit_roots": {
                "ridge_early_binned": "binned",
                "ridge_early": "plain",
            },
        }
        self.assertEqual(_study_title(index), BINNING_TITLE)
        figure, axis = plt.subplots()
        _draw_curves(
            axis, np.array([0.25, 0.5, 1.0]),
            np.ones((2, 3)), np.full((2, 3), np.nan),
            "pearson_r", tuple(index["fit_roots"]),
        )
        self.assertEqual(
            axis.get_legend_handles_labels()[1],
            ["Temporal binning (CV selected)", "No temporal binning"],
        )
        plt.close(figure)

    def test_hashed_identity_excludes_invocation(self) -> None:
        """Subject and worker selections cannot change a fit root."""
        base = {
            "artifact_base": "/tmp/corteg_binning_test",
            "preprocessing": {"fixed": "recipe"},
        }
        settings = {
            "bin_counts": [50, 25, 12, 8],
            "folds": 5, "purge_rows": 25,
            "alpha_per_training_row": [1.0, 10.0],
        }
        first = binned_ridge_identity(base, settings)
        changed = {**base, "subjects": ["bp"], "cpu_workers": 8}
        self.assertEqual(
            first, binned_ridge_identity(changed, settings),
        )
        self.assertEqual(
            binned_ridge_root(base, settings).name,
            f"ridge_early_binned__{first}",
        )
        modified = {**settings, "bin_counts": [50, 25, 8]}
        self.assertNotEqual(
            first, binned_ridge_identity(base, modified),
        )

    def test_fit_saves_selected_grid_and_resumes(self) -> None:
        """One fit saves aligned predictions and a last-written marker."""
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = root / "bp" / "common"
            features = root / "bp" / "corteg" / "0"
            common.mkdir(parents=True)
            features.mkdir(parents=True)
            broad = rng.normal(size=(300, 2, 128)).astype(np.float32)
            high = rng.normal(size=(300, 2, 200, 2)).astype(np.float32)
            test_broad = rng.normal(
                size=(30, 2, 128),
            ).astype(np.float32)
            test_high = rng.normal(
                size=(30, 2, 200, 2),
            ).astype(np.float32)
            train_y = np.tile(
                (broad.mean(axis=2) + high[..., 0].mean(axis=2))
                @ np.ones((2, 1)), (1, 5),
            )
            test_y = np.tile(
                (test_broad.mean(axis=2)
                 + test_high[..., 0].mean(axis=2))
                @ np.ones((2, 1)), (1, 5),
            )
            arrays = {
                common / "train_y.npy": train_y,
                common / "test_y.npy": test_y,
                common / "train_indices.npy": np.arange(300),
                common / "test_indices.npy": np.arange(30) + 300,
                features / "train_x.npy": broad,
                features / "test_x.npy": test_broad,
                features / "train_hi.npy": high,
                features / "test_hi.npy": test_high,
            }
            for path, values in arrays.items():
                np.save(path, values)
            settings = {
                "bin_counts": [25, 8],
                "folds": 5, "purge_rows": 25,
                "alpha_per_training_row": [1.0, 10.0],
            }
            output = root / "fit"
            fit_binned_ridge(root, "bp", 1.0, output, settings)
            result_path = output / "result.json"
            result = json.loads(result_path.read_text())
            history = json.loads((output / "cv_history.json").read_text())
            saved = joblib.load(output / "best.joblib")
            self.assertIn(result["selected_bin_count"], (25, 8))
            self.assertEqual(
                saved["selected_bin_count"],
                result["selected_bin_count"],
            )
            self.assertEqual(np.array(history["scores"]).shape, (2, 2, 5))
            with np.load(output / "test_predictions.npz") as prediction:
                self.assertEqual(prediction["y_hat"].shape, (30, 5))
            original = result_path.read_bytes()
            fit_binned_ridge(root, "bp", 1.0, output, settings)
            self.assertEqual(result_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
