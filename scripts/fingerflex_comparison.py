"""Compare completed model-specific Stanford FingerFlex fit campaigns.

Input is a comparison manifest with four model fit roots. Output is
reports/summary.json plus subject, aggregate, and six-pair comparison PNGs.
The comparison artifact stores references and reports, not model fits.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from data.fingerflex_cache import preparation_id
from data.fingerflex_campaign import digest_array
from scripts.fingerflex_campaign import _holm, _paired_test

METRICS = ("pearson_r", "mse", "r2")
MODELS = ("corteg", "hilo", "pls", "cnn_gru")
COMPARISON_VERSION = "paired_subject_signflip_v1"


def _identity(semantics: dict) -> str:
    """Hash comparison semantics independently of fitted run files."""
    canonical = json.dumps(
        semantics, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def ensure_comparison(
    study: dict,
    base: dict,
    roots: dict[str, Path],
) -> Path:
    """Create or validate a report-only comparison campaign."""
    fit_ids = {
        model: json.loads((root / "index.json").read_text())["semantic_id"]
        for model, root in roots.items()
    }
    semantics = {
        "version": COMPARISON_VERSION,
        "fit_ids": fit_ids,
        "rules": study["comparison"],
        "preprocessing_id": preparation_id(base),
    }
    identifier = _identity(semantics)
    root = (
        Path(base["artifact_base"]) / "comparisons"
        / f"{study['name']}__{identifier}"
    )
    for directory in ("provenance", "reports"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    index = {
        "name": study["name"],
        "semantic_id": identifier,
        "semantics": semantics,
        "fit_roots": {key: str(value) for key, value in roots.items()},
        "subjects": study["subjects"],
        "fractions": study["fractions"],
        "seeds": study["seeds"],
    }
    path = root / "index.json"
    if path.exists():
        saved = json.loads(path.read_text())
        for key in ("semantic_id", "semantics", "fit_roots"):
            if saved[key] != index[key]:
                raise ValueError(f"Comparison identity conflict: {path}.")
        for key in ("subjects", "fractions", "seeds"):
            previous = saved[key]
            saved[key] = list(dict.fromkeys(previous + index[key]))
        index = saved
    path.write_text(json.dumps(index, indent=2) + "\n")
    (root / "provenance" / "resolved_comparison.json").write_text(
        json.dumps(study, indent=2) + "\n"
    )
    return root


def _job_path(
    fit_root: Path,
    model: str,
    fraction: float,
    subject: str,
    seed: int | None,
) -> Path:
    """Locate one completed fit in its model-specific campaign."""
    repeat = "deterministic" if seed is None else f"seed{seed}"
    return (
        fit_root / "runs" / f"{int(fraction * 100)}pct"
        / model / subject / repeat
    )


def _read_fit(path: Path, model: str, subject: str,
              fraction: float, seed: int | None) -> dict | None:
    """Load a complete, finite fit result or report it as pending."""
    result_path = path / "result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    expected = {
        "model": model, "subject": subject, "fraction": fraction,
        "seed": seed, "delay_ms": 0,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Fit identity differs: {result_path}.")
    if not result.get("test_scored_after_restore"):
        raise ValueError(f"Test restore absent: {result_path}.")
    scores = result["metrics"]["overall"]
    if set(scores) != set(METRICS) or not np.isfinite(
        list(scores.values())
    ).all():
        raise ValueError(f"Invalid fit metrics: {result_path}.")
    checkpoint = "best.joblib" if model == "pls" else "best.pt"
    if not (path / checkpoint).is_file():
        raise FileNotFoundError(f"Fit checkpoint absent: {path}.")
    predictions = path / "test_predictions.npz"
    if not predictions.is_file():
        raise FileNotFoundError(f"Test predictions absent: {path}.")
    with np.load(predictions) as saved:
        result["truth_sha256"] = digest_array(saved["y"])
    return result


def _summarize(
    results: list[dict],
    model: str,
) -> dict:
    """Average completed seeds within a subject, retaining PLS N/A SD."""
    scores = np.array([
        [result["metrics"]["overall"][metric] for metric in METRICS]
        for result in results
    ])
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite fitted score.")
    means = scores.mean(axis=0)
    std = (
        scores.std(axis=0, ddof=1) if model != "pls"
        and len(results) > 1 else None
    )
    return {
        "fits": len(results),
        "mean": dict(zip(METRICS, means.tolist())),
        "std": dict(zip(
            METRICS,
            std.tolist() if std is not None else [None] * len(METRICS),
        )),
        "seed_variance": "not_applicable" if model == "pls"
        else "sample_standard_deviation",
    }


def _check_alignment(results: dict[str, list[dict]]) -> None:
    """Require identical fit, validation, and test target identities."""
    keys = (
        "fit_indices_sha256", "validation_indices_sha256",
        "test_grid_sha256", "target_values_sha256",
    )
    references = [
        (result["truth_sha256"],) + tuple(
            result["boundaries"][key] for key in keys
        ) for group in results.values() for result in group
    ]
    if len(set(references)) > 1:
        raise ValueError(
            "Model fits disagree on target rows or target values."
        )


def report(root: Path) -> dict:
    """Aggregate available fits and test six complete paired comparisons."""
    index = json.loads((root / "index.json").read_text())
    fit_roots = {
        model: Path(path) for model, path in index["fit_roots"].items()
    }
    summary = {
        "fit_ids": index["semantics"]["fit_ids"],
        "comparison_id": index["semantic_id"],
        "metrics": list(METRICS),
        "fractions": {},
    }
    all_complete = True
    for fraction in index["fractions"]:
        key = f"{int(fraction * 100)}pct"
        subject_rows = {}
        for subject in index["subjects"]:
            groups: dict[str, list[dict]] = {}
            entries = {}
            for model in MODELS:
                seeds = [None] if model == "pls" else index["seeds"]
                found = [
                    _read_fit(
                        _job_path(
                            fit_roots[model], model, fraction,
                            subject, seed,
                        ),
                        model, subject, fraction, seed,
                    ) for seed in seeds
                ]
                results = [result for result in found if result is not None]
                if results:
                    groups[model] = results
                    entries[model] = _summarize(results, model)
                if len(results) != len(seeds):
                    all_complete = False
            if groups:
                _check_alignment(groups)
            subject_rows[subject] = entries
        comparisons = {}
        for first, second in itertools.combinations(MODELS, 2):
            complete = all(
                subject_rows[subject].get(first, {}).get("fits")
                == (1 if first == "pls" else len(index["seeds"]))
                and subject_rows[subject].get(second, {}).get("fits")
                == (1 if second == "pls" else len(index["seeds"]))
                for subject in index["subjects"]
            )
            pair_key = f"{first}__{second}"
            if not complete:
                comparisons[pair_key] = {"complete": False}
                continue
            difference = {
                metric: np.array([
                    subject_rows[subject][first]["mean"][metric]
                    - subject_rows[subject][second]["mean"][metric]
                    for subject in index["subjects"]
                ]) for metric in METRICS
            }
            raw = {
                metric: _paired_test(values)
                for metric, values in difference.items()
            }
            corrected = _holm(raw)
            comparisons[pair_key] = {
                "complete": True,
                "subjects": len(index["subjects"]),
                "metrics": {
                    metric: {
                        "mean_difference": float(values.mean()),
                        "p_two_sided": raw[metric],
                        "p_holm": corrected[metric],
                    } for metric, values in difference.items()
                },
            }
        summary["fractions"][key] = {
            "subjects": subject_rows,
            "comparisons": comparisons,
        }
    summary["complete"] = all_complete
    output = root / "reports" / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    if all_complete:
        from plotting.fingerflex_comparison import generate_plots
        generate_plots(root)
    return summary
