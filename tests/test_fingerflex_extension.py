"""Verify new FingerFlex features, fits, and comparison artifacts.

Input is synthetic feature and target arrays in temporary directories.
Output is assertions and temporary reports and plots; no raw data or GPU.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.fingerflex_cache import cache_root
from data.fingerflex_wavelet import _write_features, feature_root
from experiments.fingerflex_pls import fit_pls
from models.temporal_spatial_cnn_gru import TemporalSpatialCNNGRU
from scripts.fingerflex_fit_campaigns import fit_semantics
from scripts.fingerflex_comparison import ensure_comparison, report


class FingerflexExtensionTests(unittest.TestCase):
    """Protect the deterministic baseline and report contracts."""

    def test_model_is_dataset_agnostic(self) -> None:
        """Architecture accepts arbitrary channel, duration, output counts."""
        model = TemporalSpatialCNNGRU(4, 200, 3)
        result = model(torch.randn(2, 4, 200))
        self.assertEqual(tuple(result.shape), (2, 3))
        with self.assertRaisesRegex(ValueError, "Expected input shape"):
            model(torch.randn(2, 5, 200))

    def test_wavelet_pools_to_upstream_pls_shape(self) -> None:
        """Wavelet bins and target-window count match the configured grid."""
        rng = np.random.default_rng(1)
        signal = rng.normal(size=(1500, 3))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "wavelet.npy"
            _write_features(
                output, signal, np.array([1000, 1040, 1080]),
            )
            features = np.load(output)
            self.assertEqual(features.shape, (3, 3, 10, 15))
            self.assertTrue(np.isfinite(features).all())

    def test_pls_one_fit_with_validation_selection(self) -> None:
        """PLS writes one deterministic completion and restored test score."""
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = {
                "artifact_base": str(base),
                "preprocessing": {
                    "ecog_scale": 1.0,
                },
            }
            prepared = cache_root(config) / "subjects"
            common = prepared / "bp" / "common"
            features = feature_root(config, "bp")
            common.mkdir(parents=True)
            features.mkdir(parents=True)
            train_x = rng.normal(size=(60, 2, 10, 15)).astype(np.float32)
            test_x = rng.normal(size=(12, 2, 10, 15)).astype(np.float32)
            weight = rng.normal(size=(300, 5))
            train_y = train_x.reshape(60, -1) @ weight
            test_y = test_x.reshape(12, -1) @ weight
            arrays = {
                common / "train_y.npy": train_y,
                common / "test_y.npy": test_y,
                common / "train_indices.npy": np.arange(60),
                common / "test_indices.npy": np.arange(12) + 60,
                features / "train_x.npy": train_x,
                features / "test_x.npy": test_x,
            }
            for path, value in arrays.items():
                np.save(path, value)
            output = base / "fit"
            fit_pls(
                prepared, config, "bp", 1.0, output,
                max_components=2,
            )
            result = json.loads(
                (output / "result.json").read_text()
            )
            self.assertIsNone(result["seed"])
            self.assertIn(result["selected_components"], (1, 2))
            self.assertTrue(result["test_scored_after_restore"])
            self.assertEqual(
                len((output / "component_history.jsonl")
                    .read_text().splitlines()),
                2,
            )
            fit_pls(
                prepared, config, "bp", 1.0, output,
                max_components=2,
            )

    def test_fit_and_comparison_identity_exclude_selectors(self) -> None:
        """Adding selectors reuses fit and comparison campaign roots."""
        with tempfile.TemporaryDirectory() as temporary:
            base_dir = Path(temporary)
            base = {
                "artifact_base": str(base_dir),
                "preprocessing": {"ecog_scale": 1.0},
            }
            legacy = base_dir / "legacy"
            provenance = legacy / "provenance"
            provenance.mkdir(parents=True)
            (provenance / "semantics.json").write_text(json.dumps({
                "model_source_sha256": {
                    "experiments/fingerflex_campaign.py": "one",
                    "experiments/run_regression_hilo_clean.py": "two",
                    "train/lr_schedule.py": "three",
                    "models/steegformer/probe.py": "four",
                    "HiLoFuseNet/finger_regression/models/"
                    "nn_regressors.py": "five",
                },
            }))
            base["training"] = {"epochs": 2}
            base["checkpoint_sha256"] = "checkpoint"
            study = {
                "name": "comparison",
                "subjects": ["bp"], "fractions": [1.0],
                "seeds": [0], "comparison": {"test": "paired"},
                "pls": {"max_components": 2},
                "cnn_gru": {"epochs": 2},
            }
            identity = fit_semantics("pls", study, base, legacy)
            study["subjects"].append("jp")
            study["seeds"].append(1)
            self.assertEqual(
                identity, fit_semantics("pls", study, base, legacy),
            )
            roots = {}
            for model in ("corteg", "hilo", "pls", "cnn_gru"):
                root = base_dir / "fits" / model
                root.mkdir(parents=True)
                (root / "index.json").write_text(json.dumps({
                    "semantic_id": model,
                }))
                roots[model] = root
            first = ensure_comparison(study, base, roots)
            study["subjects"].append("mv")
            second = ensure_comparison(study, base, roots)
            self.assertEqual(first, second)
            index = json.loads((second / "index.json").read_text())
            self.assertEqual(index["subjects"], ["bp", "jp", "mv"])

    def test_report_all_six_pairs(self) -> None:
        """Paired tests use subjects after seed averaging, including PLS."""
        models = ("corteg", "hilo", "pls", "cnn_gru")
        subjects = ("bp", "jp", "mv")
        fractions = (0.1, 0.25, 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                model: root / "fits" / model for model in models
            }
            comparison = root / "comparison"
            (comparison / "reports").mkdir(parents=True)
            (comparison / "index.json").write_text(json.dumps({
                "semantic_id": "test",
                "semantics": {"fit_ids": {
                    model: model for model in models
                }},
                "fit_roots": {
                    model: str(path) for model, path in roots.items()
                },
                "subjects": subjects, "fractions": fractions,
                "seeds": (0, 1),
            }))
            for fraction in fractions:
                for subject in subjects:
                    for model in models:
                        seeds = (None,) if model == "pls" else (0, 1)
                        for seed in seeds:
                            repeat = (
                                "deterministic" if seed is None
                                else f"seed{seed}"
                            )
                            path = (
                                roots[model] / "runs"
                                / f"{int(fraction * 100)}pct"
                                / model / subject / repeat
                            )
                            path.mkdir(parents=True)
                            checkpoint = (
                                "best.joblib" if model == "pls"
                                else "best.pt"
                            )
                            (path / checkpoint).write_bytes(b"fit")
                            np.savez(
                                path / "test_predictions.npz",
                                y=np.arange(10, dtype=np.float32),
                                y_hat=np.arange(10, dtype=np.float32),
                            )
                            score = models.index(model) / 10
                            (path / "result.json").write_text(json.dumps({
                                "model": model, "subject": subject,
                                "fraction": fraction, "seed": seed,
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
                                "metrics": {"overall": {
                                    "pearson_r": score,
                                    "mse": 1 - score,
                                    "r2": score - 1,
                                }},
                            }))
            summary = report(comparison)
            self.assertTrue(summary["complete"])
            for entry in summary["fractions"].values():
                self.assertEqual(len(entry["comparisons"]), 6)
                self.assertTrue(all(
                    row["complete"] for row in
                    entry["comparisons"].values()
                ))
                self.assertEqual(
                    entry["subjects"]["bp"]["pls"]["seed_variance"],
                    "not_applicable",
                )
            self.assertTrue(
                (comparison / "reports" / "plots"
                 / "paired_matrix.png").is_file()
            )
            changed = (
                roots["pls"] / "runs" / "100pct"
                / "pls" / "bp" / "deterministic"
                / "test_predictions.npz"
            )
            np.savez(
                changed, y=np.ones(10, dtype=np.float32),
                y_hat=np.ones(10, dtype=np.float32),
            )
            with self.assertRaisesRegex(
                ValueError, "target rows or target values",
            ):
                report(comparison)


if __name__ == "__main__":
    unittest.main()
