"""Check matched regression aggregation and resumable seed jobs.

Input: temporary per-subject JSON scores for multiple experiment seeds.
Output: assertions on matched subjects, test counts, and artifact identity.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.multiseed_regression import (
    evaluate, experiment_root, prepare_jobs, run_campaign,
)


class MultiSeedTests(unittest.TestCase):
    """Check subject pairing and seed summaries."""

    def test_aggregate_uses_all_declared_subjects(self) -> None:
        """Average seeds within subjects and reject a missing split."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for model, offset in (("A", 0.0), ("B", -0.1)):
                sources[model] = {}
                for seed in (1, 2):
                    path = root / f"{model}_{seed}.json"
                    scores = {
                        name: {"corr_mean": offset + seed * 0.1
                               + index * 0.05, "n": 10}
                        for index, name in enumerate(("bp", "cc", "ht"))
                    }
                    path.write_text(json.dumps({
                        "dataset": "Stanford",
                        "per_subject": scores,
                    }))
                    sources[model][str(seed)] = str(path)
            spec = {
                "name": "Finger regression",
                "semantic_config": {"metric": "Pearson r"},
                "tasks": [{
                "name": "finger",
                "semantic_config": {"alignment_ms": 40},
                "dataset": "Stanford",
                "protocol_id": "same_test_grid",
                "subjects": ["bp", "cc", "ht"],
                "models": [
                    {"name": name, "results": paths,
                     "semantic_config": {"model_family": name}}
                    for name, paths in sources.items()
                ],
            }]}
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            report = evaluate(spec_path, root / "out")
            model = report["aggregate"]["tasks"]["finger"][
                "models"]["A"]
            self.assertEqual(model["n_seeds"], 2)
            self.assertEqual(model["n_subjects"], 3)
            self.assertAlmostEqual(model["mean"], 0.2)
            self.assertIsNotNone(model["seed_cohort_sd"])
            comparison = report["comparisons"]["tasks"]["finger"][0]
            self.assertAlmostEqual(
                comparison["mean_difference"], 0.1
            )
            self.assertEqual(comparison["n_subjects"], 3)
            self.assertAlmostEqual(
                comparison["p_raw"], comparison["p_holm"]
            )
            sources["B"]["2"] = str(root / "missing.json")
            spec_path.write_text(json.dumps(spec))
            with self.assertRaises(FileNotFoundError):
                evaluate(spec_path, root / "out")

    def test_nested_result_path_keeps_seed_run_root(self) -> None:
        """Keep save_root above a nested result JSON path."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "name": "Finger regression",
                "semantic_config": {"metric": "Pearson r"},
                "tasks": [{
                "name": "finger",
                "semantic_config": {"alignment_ms": 40},
                "subjects": ["bp"],
                "protocol_id": "same_test_grid",
                "models": [{
                    "name": "A",
                    "semantic_config": {"model_family": "A"},
                    "seeds": [1],
                    "command": ["python", "train.py", "{seed}",
                                "{subject}", "{save_root}"],
                    "result_file": "reports/results_persub.json",
                }],
            }]}
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            jobs = prepare_jobs(spec_path, root / "out")
            artifact, _ = experiment_root(
                root / "out", spec, spec_path.parent
            )
            self.assertEqual(
                jobs[0]["run_root"],
                str(artifact / "finger" / "A" / "subjects"
                    / "bp" / "seed1"),
            )
            self.assertEqual(
                jobs[0]["result_file"],
                str(artifact / "finger" / "A" / "subjects"
                    / "bp" / "seed1"
                    / "reports" / "results_persub.json"),
            )


    def test_resume_extends_subjects_and_seeds_in_same_artifact(self) -> None:
        """Preserve completed runs when adding subjects and random seeds."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = (
                "import json,pathlib,sys;"
                "seed=int(sys.argv[1]);subject=sys.argv[2];"
                "root=pathlib.Path(sys.argv[3]);"
                "path=root/'results_persub.json';"
                "path.write_text(json.dumps({'dataset':'Stanford',"
                "'per_subject':{subject:{'corr_mean':seed/100,"
                "'n':10}},'args':{'seed':seed,"
                "'subjects':subject,'epochs':200}}))"
            )
            spec = {
                "name": "Finger regression study",
                "semantic_config": {
                    "defaults": {"metric": "Pearson r"},
                    "overrides": {},
                },
                "tasks": [{
                    "name": "finger",
                    "dataset": "Stanford",
                    "protocol_id": "one_test_grid",
                    "semantic_config": {"alignment_ms": 40},
                    "subjects": ["bp"],
                    "models": [{
                        "name": "A",
                        "semantic_config": {
                            "defaults": {"width": 16, "dropout": 0.1},
                            "overrides": {"dropout": 0.2},
                        },
                        "seeds": [1],
                        "command": [
                            "python", "-c", writer,
                            "{seed}", "{subject}", "{save_root}",
                        ],
                    }],
                }],
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            first = run_campaign(spec_path, root / "out", 1)
            artifact = Path(first["artifact_root"])
            first_result = (
                artifact / "finger" / "A" / "subjects"
                / "bp" / "seed1" / "results_persub.json"
            )
            original_time = first_result.stat().st_mtime_ns
            spec["tasks"][0]["subjects"].append("cc")
            spec["tasks"][0]["models"][0]["seeds"].append(2)
            spec["tasks"][0]["models"][0]["command"].extend(
                ["--device", "cpu"]
            )
            spec_path.write_text(json.dumps(spec))
            second = run_campaign(spec_path, root / "out", 2)
            self.assertEqual(second["artifact_root"], str(artifact))
            self.assertEqual(first_result.stat().st_mtime_ns, original_time)
            manifest = json.loads(
                (artifact / "run_manifest.json").read_text()
            )
            self.assertEqual(len(manifest["jobs"]), 4)
            self.assertEqual(len(manifest["spec_revisions"]), 2)
            summary = second["aggregate"]["tasks"]["finger"][
                "models"]["A"]
            self.assertEqual(
                summary["effective_runner_args"], {"epochs": 200}
            )
            identity = json.loads(
                (artifact / "experiment_identity.json").read_text()
            )
            resolved = identity["resolved_config"]
            model_config = resolved["tasks"][0]["models"][0][
                "semantic_config"]
            self.assertEqual(model_config["dropout"], 0.2)
            self.assertNotIn("subjects", resolved["tasks"][0])
            spec["tasks"][0]["models"][0][
                "semantic_config"]["overrides"]["dropout"] = 0.3
            changed, _ = experiment_root(
                root / "out", spec, spec_path.parent
            )
            self.assertNotEqual(changed, artifact)




if __name__ == "__main__":
    unittest.main()
