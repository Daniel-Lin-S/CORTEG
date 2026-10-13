"""Test blocked ridge fusion and four-model saved-result comparison.

Input is temporary synthetic broadband, high-gamma, target, and fit files.
Output is assertions only; no campaign data or GPU is required.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.regression import regression_metrics
from experiments.fingerflex_ridge import fit_ridge
from experiments.ridge_cv import _folds, convex_weights
from scripts.fingerflex_comparison import report
from scripts.fingerflex_linear_identity import ridge_identity, ridge_root

MODELS = ("corteg", "hilo", "ridge_early", "ridge_late")
FINGERS = 5


class FingerflexLinearFusionTests(unittest.TestCase):
    """Protect the ridge and paired-report contracts."""

    def test_fit_identity_excludes_invocation_settings(self) -> None:
        """Hash fit semantics but exclude selected work and resources."""
        base = {
            "preprocessing": {"fixed": "recipe"},
            "artifact_base": "/tmp/fingerflex_test_artifacts",
        }
        first = ridge_identity(base, "ridge_early")
        selected = {
            **base,
            "subjects": ["bp"],
            "seeds": [0],
            "gpu_workers": 4,
            "memory_limit_gib": 32,
        }
        self.assertEqual(first, ridge_identity(selected, "ridge_early"))
        self.assertEqual(
            ridge_root(base, "ridge_early").name,
            f"ridge_early__{first}",
        )
        changed = {**base, "preprocessing": {"fixed": "other"}}
        self.assertNotEqual(
            first, ridge_identity(changed, "ridge_early"),
        )

    def test_purged_folds_and_convex_weights(self) -> None:
        """Every row is held out once with an overlap-free boundary."""
        folds = _folds(450)
        held_out = np.concatenate([held for _, held in folds])
        np.testing.assert_array_equal(np.sort(held_out), np.arange(450))
        for training, held in folds:
            self.assertTrue(np.all(
                np.abs(training[:, None] - held) > 25
            ))
        target = np.ones((6, FINGERS))
        weight = convex_weights(target, np.zeros_like(target), target)
        np.testing.assert_array_equal(weight, np.ones(FINGERS))
        with self.assertRaisesRegex(ValueError, "identical"):
            convex_weights(target, target, target)

    def test_flat_prediction_correlation_is_zero(self) -> None:
        """A flat predicted finger keeps finite MSE and R²."""
        truth = np.arange(20, dtype=np.float32).reshape(4, FINGERS)
        predicted = np.zeros_like(truth)
        result = regression_metrics(truth, predicted)
        self.assertEqual(result["overall"]["pearson_r"], 0.0)
        self.assertTrue(np.isfinite(
            list(result["overall"].values())
        ).all())

    def test_both_fits_complete_and_resume(self) -> None:
        """Early and late fits write complete deterministic artifacts."""
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = root / "bp" / "common"
            features = root / "bp" / "corteg" / "0"
            common.mkdir(parents=True)
            features.mkdir(parents=True)
            low = rng.normal(size=(500, 2, 128)).astype(np.float32)
            high = rng.normal(size=(500, 2, 200, 2)).astype(np.float32)
            test_low = rng.normal(size=(35, 2, 128)).astype(np.float32)
            test_high = rng.normal(
                size=(35, 2, 200, 2)
            ).astype(np.float32)
            weight = rng.normal(size=(2, FINGERS))
            train_y = (
                low.mean(axis=2) + high[..., 0].mean(axis=2)
            ) @ weight
            test_y = (
                test_low.mean(axis=2)
                + test_high[..., 0].mean(axis=2)
            ) @ weight
            arrays = {
                common / "train_y.npy": train_y.astype(np.float32),
                common / "test_y.npy": test_y.astype(np.float32),
                common / "train_indices.npy": np.arange(500),
                common / "test_indices.npy": np.arange(35) + 500,
                features / "train_x.npy": low,
                features / "test_x.npy": test_low,
                features / "train_hi.npy": high,
                features / "test_hi.npy": test_high,
            }
            for path, value in arrays.items():
                np.save(path, value)
            for model in MODELS[2:]:
                output = root / "runs" / model
                fit_ridge(root, "bp", 1.0, model, output)
                result_path = output / "result.json"
                result = json.loads(result_path.read_text())
                self.assertTrue(result["test_scored_after_restore"])
                self.assertIsNone(result["seed"])
                if model == "ridge_late":
                    self.assertEqual(
                        len(result["fusion_weights"]), FINGERS,
                    )
                original = result_path.read_bytes()
                fit_ridge(root, "bp", 1.0, model, output)
                self.assertEqual(result_path.read_bytes(), original)

    def test_signflip_report_with_deterministic_models(self) -> None:
        """Report all six subject-level pairs and deterministic SD N/A."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison = root / "comparison"
            (comparison / "reports").mkdir(parents=True)
            fit_roots = {
                model: root / "fits" / model for model in MODELS
            }
            (comparison / "index.json").write_text(json.dumps({
                "semantic_id": "synthetic",
                "semantics": {
                    "fit_ids": {model: model for model in MODELS},
                    "rules": {
                        "metrics": ["pearson_r", "mse", "r2"],
                        "paired_unit": "subject",
                        "test": "exact_sign_flip",
                        "holm_family": (
                            "metrics_within_pair_and_fraction"
                        ),
                    },
                },
                "fit_roots": {
                    model: str(path)
                    for model, path in fit_roots.items()
                },
                "subjects": ["bp", "cc"],
                "fractions": [0.5],
                "seeds": [0, 1],
            }))
            truth = np.arange(50, dtype=np.float32).reshape(
                10, FINGERS,
            )
            for subject in ("bp", "cc"):
                for index, model in enumerate(MODELS):
                    seeds = (None,) if model.startswith("ridge") else (0, 1)
                    for seed in seeds:
                        repeat = (
                            "deterministic" if seed is None
                            else f"seed{seed}"
                        )
                        folder = (
                            fit_roots[model] / "runs" / "50pct"
                            / model / subject / repeat
                        )
                        folder.mkdir(parents=True)
                        checkpoint = (
                            "best.joblib" if model.startswith("ridge")
                            else "best.pt"
                        )
                        (folder / checkpoint).write_bytes(b"fit")
                        offset = float(index + (seed or 0)) / 10
                        predicted = truth + offset
                        np.savez(
                            folder / "test_predictions.npz",
                            y=truth, y_hat=predicted,
                        )
                        (folder / "result.json").write_text(json.dumps({
                            "model": model,
                            "subject": subject,
                            "fraction": 0.5,
                            "seed": seed,
                            "delay_ms": 0,
                            "test_scored_after_restore": True,
                            "boundaries": {
                                key: "same" for key in (
                                    "fit_indices_sha256",
                                    "validation_indices_sha256",
                                    "test_grid_sha256",
                                    "target_values_sha256",
                                )
                            },
                            "metrics": regression_metrics(
                                truth, predicted,
                            ),
                        }))
            summary = report(comparison)
            self.assertTrue(summary["complete"])
            fraction = summary["fractions"]["50pct"]
            self.assertEqual(len(fraction["comparisons"]), 6)
            self.assertEqual(
                fraction["subjects"]["bp"]["ridge_early"][
                    "seed_variance"
                ], "not_applicable",
            )
            self.assertTrue(
                (comparison / "plots" / "paired_matrix.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
