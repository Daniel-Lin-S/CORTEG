"""Synthetic checks for Stanford preparation, metrics, scheduling, and resume.

Input: deterministic arrays, temporary local settings and status folders.
Output: assertions only; no raw dataset or GPU access.
"""
from __future__ import annotations

import atexit
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
from data.fingerflex_cache import (
    LEGACY_PREPARATION_SHA256,
    cache_root,
    ensure_cache,
    preparation_id,
    prepare_cached_subject,
    resolve_campaign,
    sha256_file,
)
from experiments.fingerflex_campaign import (
    metric_values,
    normalize,
    parameter_groups,
    split_fit_rows,
)
from scripts.fingerflex_campaign import (
    LEGACY_FIT_SOURCE_SHA256,
    EXTENDED_FIT_SOURCE_SHA256,
    _holm,
    _paired_test,
    _worker,
    job_path,
    jobs_for_subject,
    rank_gpus,
    artifact_root,
    ensure_artifact,
    report,
    semantic_identity,
    source_revision,
)

REPO = Path(__file__).resolve().parents[1]
LOCAL_TEST_DIR = tempfile.TemporaryDirectory()
atexit.register(LOCAL_TEST_DIR.cleanup)
LOCAL_TEST_ROOT = Path(LOCAL_TEST_DIR.name)
(LOCAL_TEST_ROOT / "raw").mkdir()
(LOCAL_TEST_ROOT / "checkpoint.pth").write_bytes(b"checkpoint fixture")
LOCAL_TEST_CONFIG = LOCAL_TEST_ROOT / "fingerflex_local.json"
LOCAL_TEST_CONFIG.write_text(json.dumps({
    "raw_root": str(LOCAL_TEST_ROOT / "raw"),
    "artifact_base": str(LOCAL_TEST_ROOT / "artifacts"),
    "checkpoint": str(LOCAL_TEST_ROOT / "checkpoint.pth"),
    "hilo_source": str(REPO / "experiments" / "fingerflex_campaign.py"),
    "tmux_executable": "git",
}))
CONFIG = resolve_campaign(
    REPO / "configs" / "fingerflex_campaign.json",
    LOCAL_TEST_CONFIG,
)
PROBE = resolve_campaign(
    REPO / "configs" / "fingerflex_delay40ms.json",
    LOCAL_TEST_CONFIG,
)


class FingerflexCampaignTests(unittest.TestCase):
    """Check invariants that can fail without full model training."""

    def test_local_paths_and_checkpoint_hash_are_resolved(self) -> None:
        """Portable configs omit paths and runtime hashes checkpoint bytes."""
        self.assertEqual(
            CONFIG["checkpoint_sha256"],
            sha256_file(Path(CONFIG["checkpoint"])),
        )
        self.assertEqual(
            EXTENDED_FIT_SOURCE_SHA256,
            sha256_file(REPO / "experiments" / "fingerflex_campaign.py"),
        )
        portable = json.loads(
            (REPO / "configs" / "fingerflex_campaign.json").read_text()
        )
        for key in (
            "raw_root", "artifact_base", "checkpoint",
            "hilo_source", "tmux_executable", "checkpoint_sha256",
        ):
            self.assertNotIn(key, portable)
        with self.assertRaisesRegex(
            FileNotFoundError, "Local settings file is missing"
        ):
            resolve_campaign(
                REPO / "configs" / "fingerflex_campaign.json",
                LOCAL_TEST_ROOT / "absent.json",
            )

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

    def test_preparation_cache_independent_of_training(self) -> None:
        """Changing training creates a new run but reuses sample arrays."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            raw_root = base / "raw"
            raw_path = raw_root / "bp" / "bp_fingerflex.mat"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b"registered raw fixture")
            legacy = base / "legacy"
            provenance = legacy / "provenance"
            provenance.mkdir(parents=True)
            prep = {
                key: CONFIG["preprocessing"][key]
                for key in (
                    "ecog_scale", "split", "notch_bands_hz",
                    "notch_filter_order", "bad_channel",
                )
            }
            (provenance / "resolved_config.json").write_text(
                json.dumps({
                    "preprocessing": prep,
                    "delay_subjects": [],
                    "name": CONFIG["name"],
                    "training": CONFIG["training"],
                    "checkpoint_sha256": CONFIG["checkpoint_sha256"],
                })
            )
            (provenance / "semantics.json").write_text(json.dumps({
                "campaign_source_sha256": {
                    "data/fingerflex_campaign.py": (
                        LEGACY_PREPARATION_SHA256
                    ),
                    "experiments/fingerflex_campaign.py": (
                        LEGACY_FIT_SOURCE_SHA256
                    ),
                }
            }))
            (provenance / "input_hashes.json").write_text(json.dumps({
                "raw_sha256": {"bp": sha256_file(raw_path)},
                "corteg_revision": source_revision(
                    Path(__file__).resolve().parents[1]
                ),
                "hilo_revision": source_revision(
                    Path(CONFIG["hilo_source"]).parent
                ),
            }))
            prepared = legacy / "prepared" / "bp"
            for parent, files in (
                (prepared / "common", (
                    "metadata.json", "train_y.npy", "test_y.npy",
                    "train_indices.npy", "test_indices.npy",
                    "xyz_mm.npy",
                )),
                (prepared / "corteg" / "0", (
                    "train_x.npy", "test_x.npy", "train_hi.npy",
                    "test_hi.npy",
                )),
                (prepared / "hilo" / "0", (
                    "train_x.npy", "test_x.npy",
                )),
            ):
                parent.mkdir(parents=True)
                for filename in files:
                    (parent / filename).write_bytes(b"fixture")
            legacy_result = (
                legacy / "runs" / "100pct" / "corteg"
                / "bp" / "seed0" / "result.json"
            )
            legacy_result.parent.mkdir(parents=True)
            legacy_result.write_text(json.dumps({
                "subject": "bp", "model": "corteg",
                "fraction": 1.0, "seed": 0, "delay_ms": 0,
                "test_scored_after_restore": True,
                "metrics": {"overall": {
                    "pearson_r": 0.5, "mse": 0.8, "r2": 0.1,
                }},
            }))
            (legacy_result.parent / "best.pt").write_bytes(b"fixture")
            config = dict(CONFIG)
            config.update({
                "artifact_base": str(base),
                "raw_root": str(raw_root),
                "subjects": ["bp"],
                "delay_ms": 0,
                "models": ["corteg", "hilo"],
            })
            shared = ensure_cache(config)
            self.assertTrue((shared / "bp" / "common").is_symlink())
            with patch("data.fingerflex_cache.prepare_subject") as prepare:
                self.assertTrue(
                    prepare_cached_subject("bp", config, (0,))
                )
            prepare.assert_not_called()
            first = ensure_artifact(config)
            self.assertEqual((first / "prepared").resolve(), shared)
            adopted_result = (
                first / "runs" / "100pct" / "corteg"
                / "bp" / "seed0" / "result.json"
            )
            self.assertTrue(adopted_result.is_file())
            self.assertTrue(adopted_result.parent.is_symlink())
            modified = dict(config)
            modified["training"] = dict(config["training"])
            modified["training"]["base_lr"] = 0.0005
            second = ensure_artifact(modified)
            self.assertNotEqual(first, second)
            self.assertEqual((second / "prepared").resolve(), shared)
            self.assertFalse((
                second / "runs" / "100pct" / "corteg"
                / "bp" / "seed0"
            ).exists())
            self.assertEqual(cache_root(config), cache_root(modified))
            added = dict(config)
            added["seeds"] = [0, 1, 2, 3]
            added["subjects"] = ["bp", "cc"]
            self.assertEqual(artifact_root(added), first)
            changed_prep = dict(config)
            changed_prep["preprocessing"] = dict(
                config["preprocessing"]
            )
            changed_prep["preprocessing"]["bad_channel"] = dict(
                config["preprocessing"]["bad_channel"]
            )
            changed_prep["preprocessing"]["bad_channel"][
                "high_variance_multiplier"
            ] = 4.0
            self.assertNotEqual(
                preparation_id(config), preparation_id(changed_prep)
            )



    def test_probe_report_omits_paired_model_test(self) -> None:
        """One-model reports retain seeds without a paired-model p-value."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for subject in PROBE["subjects"]:
                for job in jobs_for_subject(subject, PROBE):
                    output = job_path(root, job)
                    output.mkdir(parents=True)
                    (output / "result.json").write_text(json.dumps({
                        "metrics": {"overall": {
                            "pearson_r": 0.5, "mse": 0.8, "r2": 0.1,
                        }},
                    }))
            (root / "reports").mkdir()
            summary = report(root, PROBE)
            self.assertEqual(summary["delay_ms"], 40)
            fraction = summary["fractions"]["100pct"]
            self.assertFalse(fraction["paired_complete"])
            self.assertNotIn("paired_test", fraction)
            self.assertEqual(
                fraction["subjects"]["bp"]["corteg"]["seeds"], 3
            )

    def test_report_family_and_job_count(self) -> None:
        """Count approved jobs and correct only the three metrics."""
        total = sum(len(jobs_for_subject(subject, CONFIG))
                    for subject in CONFIG["subjects"])
        self.assertEqual(total, 162)
        probe_total = sum(
            len(jobs_for_subject(subject, PROBE))
            for subject in PROBE["subjects"]
        )
        self.assertEqual(probe_total, 9)
        self.assertEqual(preparation_id(CONFIG), preparation_id(PROBE))
        self.assertNotEqual(
            semantic_identity(CONFIG)[0], semantic_identity(PROBE)[0]
        )
        self.assertTrue(all(
            job["delay"] == 40 and job["model"] == "corteg"
            and job["fraction"] == 1.0
            for subject in PROBE["subjects"]
            for job in jobs_for_subject(subject, PROBE)
        ))
        adjusted = _holm({"pearson_r": 0.01, "mse": 0.2,
                          "r2": 0.03})
        self.assertEqual(adjusted["pearson_r"], 0.03)
        self.assertEqual(adjusted["r2"], 0.06)
        self.assertEqual(_paired_test(np.zeros(9)), 1.0)
        self.assertEqual(_paired_test(np.zeros(3)), 1.0)


if __name__ == "__main__":
    unittest.main()
