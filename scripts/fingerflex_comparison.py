"""Compare completed model-specific Stanford FingerFlex fit campaigns.

Input is a portable comparison JSON and model-specific fit artifacts.
Output is a separate comparison root with index.json, reports/summary.json,
and plots/. Fit artifacts and predictions remain in their original roots.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_cache import preparation_id, resolve_campaign
from data.fingerflex_campaign import digest_array
from evaluation.regression import regression_metrics
from scripts.fingerflex_campaign import _holm, _paired_test
from scripts.fingerflex_fit_campaigns import find_legacy, fit_root
from scripts.fingerflex_linear_identity import (
    binned_ridge_identity, ridge_identity,
)

METRICS = ("pearson_r", "mse", "r2")
SUPPORTED_MODELS = (
    "corteg", "hilo", "pls", "cnn_gru",
    "ridge_early", "ridge_late", "ridge_early_binned",
)
DETERMINISTIC_MODELS = (
    "pls", "ridge_early", "ridge_late", "ridge_early_binned",
)
COMPARISON_VERSION = "paired_subject_wilcoxon_v1"
SIGN_FLIP_VERSION = "paired_subject_exact_sign_flip_v1"
EXPECTED_RULES = {
    "metrics": list(METRICS),
    "paired_unit": "subject",
    "test": "wilcoxon_signed_rank",
    "holm_family": "metrics_within_pair_and_fraction",
}
SIGN_FLIP_RULES = {**EXPECTED_RULES, "test": "exact_sign_flip"}
REUSED_FRACTIONS = (0.25, 1.0)
LINEAR_FUSION_MODELS = (
    "corteg", "hilo", "ridge_early", "ridge_late",
)


def _validate_rules(rules: dict) -> None:
    """Accept historical and exact sign-flip comparison rules."""
    if rules not in (EXPECTED_RULES, SIGN_FLIP_RULES):
        raise ValueError(f"Unsupported comparison rules: {rules}.")


def _paired_wilcoxon(difference: np.ndarray) -> float:
    """Return a two-sided signed-rank p-value across paired subjects.

    Parameters
    ----------
    difference : ndarray, shape (subjects,)
        Seed-averaged first-model minus second-model metric values.

    Returns
    -------
    float
        Two-sided Wilcoxon p-value.
    """
    if difference.ndim != 1 or len(difference) < 2 or not np.isfinite(
        difference
    ).all():
        raise ValueError(
            "Wilcoxon needs at least two finite subject differences."
        )
    if np.all(difference == 0):
        return 1.0
    result = wilcoxon(
        difference, alternative="two-sided", zero_method="wilcox",
        method="auto",
    )
    pvalue = float(result.pvalue)
    if not np.isfinite(pvalue):
        raise ValueError("Wilcoxon returned a nonfinite p-value.")
    return pvalue


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
    selected = tuple(study.get("models", roots))
    if tuple(roots) != selected:
        raise ValueError(
            "Comparison models must match the ordered fit-root keys."
        )
    fit_ids = {
        model: json.loads((root / "index.json").read_text())["semantic_id"]
        for model, root in roots.items()
    }
    semantics = {
        "version": (
            SIGN_FLIP_VERSION if study["comparison"] == SIGN_FLIP_RULES
            else COMPARISON_VERSION
        ),
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


def _read_fit(
    path: Path, model: str, subject: str, fraction: float,
    seed: int | None, validate_scores: bool = False,
) -> dict | None:
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
    checkpoint = (
        "best.joblib" if model in DETERMINISTIC_MODELS else "best.pt"
    )
    if not (path / checkpoint).is_file():
        raise FileNotFoundError(f"Fit checkpoint absent: {path}.")
    predictions = path / "test_predictions.npz"
    if not predictions.is_file():
        raise FileNotFoundError(f"Test predictions absent: {path}.")
    with np.load(predictions) as saved:
        truth = saved["y"]
        result["truth_sha256"] = digest_array(truth)
        if validate_scores:
            recalculated = regression_metrics(
                truth, saved["y_hat"],
            )
    if validate_scores:
        for metric in METRICS:
            recorded = scores[metric]
            actual = recalculated["overall"][metric]
            if not np.isclose(recorded, actual, rtol=1e-4, atol=1e-5):
                raise ValueError(
                    f"Saved {metric} differs from predictions: {path}."
                )
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
        scores.std(axis=0, ddof=1) if model not in DETERMINISTIC_MODELS
        and len(results) > 1 else None
    )
    return {
        "fits": len(results),
        "mean": dict(zip(METRICS, means.tolist())),
        "std": dict(zip(
            METRICS,
            std.tolist() if std is not None else [None] * len(METRICS),
        )),
        "seed_variance": "not_applicable" if model in DETERMINISTIC_MODELS
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
    """Aggregate fits and test every complete paired model comparison."""
    index = json.loads((root / "index.json").read_text())
    rules = index.get("semantics", {}).get("rules", {})
    if rules:
        _validate_rules(rules)
    test = rules.get("test", "wilcoxon_signed_rank")
    fit_roots = {
        model: Path(path) for model, path in index["fit_roots"].items()
    }
    models = tuple(fit_roots)
    if len(models) < 2 or set(models) - set(SUPPORTED_MODELS):
        raise ValueError(
            f"Expected at least two supported models, got {models}."
        )
    if set(index["semantics"]["fit_ids"]) != set(models):
        raise ValueError("Comparison fit identities disagree with fit roots.")
    summary = {
        "fit_ids": index["semantics"]["fit_ids"],
        "comparison_id": index["semantic_id"],
        "metrics": list(METRICS),
        "models": list(models),
        "fractions": {},
    }
    all_complete = True
    for fraction in index["fractions"]:
        key = f"{int(fraction * 100)}pct"
        subject_rows = {}
        for subject in index["subjects"]:
            groups: dict[str, list[dict]] = {}
            entries = {}
            for model in models:
                seeds = (
                    [None] if model in DETERMINISTIC_MODELS
                    else index["seeds"]
                )
                found = [
                    _read_fit(
                        _job_path(
                            fit_roots[model], model, fraction,
                            subject, seed,
                        ),
                        model, subject, fraction, seed,
                        validate_scores=bool(rules),
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
        for first, second in itertools.combinations(models, 2):
            complete = all(
                subject_rows[subject].get(first, {}).get("fits")
                == (1 if first in DETERMINISTIC_MODELS
                    else len(index["seeds"]))
                and subject_rows[subject].get(second, {}).get("fits")
                == (1 if second in DETERMINISTIC_MODELS
                    else len(index["seeds"]))
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
                metric: (
                    _paired_test(values) if test == "exact_sign_flip"
                    else _paired_wilcoxon(values)
                )
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



def _linear_fusion_fit_ids(study: dict, base: dict) -> dict[str, str]:
    """Derive fit identities from fitting semantics and stable roots."""
    if set(study["models"]) == {
        "ridge_early", "ridge_early_binned",
    }:
        if "binning" not in study:
            raise ValueError("Binned comparison needs binning settings.")
        return {
            "ridge_early": ridge_identity(base, "ridge_early"),
            "ridge_early_binned": binned_ridge_identity(
                base, study["binning"],
            ),
        }
    if set(study["models"]) != set(LINEAR_FUSION_MODELS):
        raise ValueError(
            "Automatic fit identity is defined for the linear "
            "fusion and binned-ridge studies only."
        )
    historical = {**study, "fractions": REUSED_FRACTIONS}
    legacy = find_legacy(base, historical)
    identities = {}
    for model in ("corteg", "hilo"):
        root = fit_root(model, study, base, legacy)
        identities[model] = root.name.rsplit("__", 1)[1]
    for model in ("ridge_early", "ridge_late"):
        identities[model] = ridge_identity(base, model)
    return identities


def load_comparison_study(path: Path) -> tuple[dict, dict]:
    """Resolve portable comparison selectors and local artifact settings."""
    study = json.loads(path.read_text())
    models = study["models"]
    if len(models) < 2 or len(set(models)) != len(models) or (
        set(models) - set(SUPPORTED_MODELS)
    ):
        raise ValueError(
            f"Comparison requires unique supported models: {models}."
        )
    if "fit_ids" in study and set(study["fit_ids"]) != set(models):
        raise ValueError("Fit IDs must name exactly the selected models.")
    _validate_rules(study["comparison"])
    campaign = Path(study["source_campaign"])
    if campaign.is_absolute():
        raise ValueError("Source campaign must be a repository path.")
    base = resolve_campaign(REPO / campaign)
    if base["delay_ms"] != 0:
        raise ValueError("This comparison requires zero-delay fits.")
    for key in ("subjects", "fractions", "seeds"):
        values = study[key]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Invalid {key} selectors: {values}.")
        if key != "fractions" and not set(values).issubset(base[key]):
            raise ValueError(f"{key} selectors exceed the source campaign.")
    if any(not 0 < value <= 1 for value in study["fractions"]):
        raise ValueError("Comparison fractions must lie in (0, 1].")
    if "fit_ids" not in study:
        study["fit_ids"] = _linear_fusion_fit_ids(study, base)
    return study, base


def find_fit_roots(study: dict, base: dict) -> dict[str, Path]:
    """Resolve model roots by derived or recorded semantic identities."""
    parent = Path(base["artifact_base"]) / "fits"
    expected_preprocessing = preparation_id(base)
    roots = {}
    requested = study["fit_ids"]
    for model in study["models"]:
        matches = []
        for index_path in parent.glob("*/index.json"):
            if index_path.parent.is_symlink():
                continue
            index = json.loads(index_path.read_text())
            if index.get("model") != model:
                continue
            if index.get("semantic_id") != requested[model]:
                continue
            if index.get("preprocessing_id") != expected_preprocessing:
                raise ValueError(
                    f"Fit preprocessing identity differs: {index_path}."
                )
            if index.get("semantics", {}).get("delay_ms") != 0:
                raise ValueError(f"Expected zero-delay fit: {index_path}.")
            matches.append(index_path.parent)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one fit root for {model} with semantic ID "
                f"{requested[model]}; found {len(matches)}."
            )
        roots[model] = matches[0]
    return roots


def main() -> None:
    """Create a model-subset comparison and report completed fits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    study, base = load_comparison_study(args.config)
    roots = find_fit_roots(study, base)
    comparison = ensure_comparison(study, base, roots)
    summary = report(comparison)
    print(comparison.resolve())
    print(f"Complete: {summary['complete']}")


if __name__ == "__main__":
    main()
