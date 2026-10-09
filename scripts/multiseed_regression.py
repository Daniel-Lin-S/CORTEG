#!/usr/bin/env python3
"""Coordinate named regression experiments and matched-subject tests.

Usage
-----
python scripts/multiseed_regression.py run --spec SPEC.json \\
    --output-root OUTPUT_BASE [--max-parallel 1]
python scripts/multiseed_regression.py evaluate --spec SPEC.json \\
    --output-root OUTPUT_BASE

Input
-----
SPEC.json names the experiment and provides semantic_config at experiment,
task, and model levels. Tasks list subjects and models; launch commands use
{subject}, {seed}, and {save_root}. Existing result JSON files may instead be
mapped from seed strings in a model's results field. Result files contain a
per_subject map of subject ID to finite corr_mean and positive test count n.

Output
------
OUTPUT_BASE/<readable-name>__<semantic-hash>/ contains experiment_identity.json
with the exact resolved semantic configuration, immutable spec_versions,
run_manifest.json, aggregate.json, and comparisons.json. Launched runs live
under task/model/subjects/<subject>/seedN/ with run.log and model artifacts.
Completed subject/seed folders are reused without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy.stats import wilcoxon


NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_RESULT_FILE = "results_persub.json"
AGGREGATE_FILE = "aggregate.json"
COMPARISONS_FILE = "comparisons.json"
MANIFEST_FILE = "run_manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[1]



def merge_semantic_defaults(
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Recursively apply semantic overrides to declared defaults.

    Parameters
    ----------
    defaults, overrides : dict[str, Any]
        JSON objects of default and overriding semantic settings.

    Returns
    -------
    dict[str, Any]
        Fully resolved semantic configuration.
    """
    resolved = dict(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f"Unknown semantic override {key!r}.")
        original = defaults[key]
        if isinstance(original, dict) and isinstance(value, dict):
            resolved[key] = merge_semantic_defaults(original, value)
        else:
            resolved[key] = value
    return resolved


def resolve_semantic_value(value: Any, source_dir: Path) -> Any:
    """Resolve nested JSON references and default/override objects.

    Parameters
    ----------
    value : Any
        JSON semantic value or ``{"$json": path}`` reference.
    source_dir : Path
        Directory used for relative referenced JSON paths.

    Returns
    -------
    Any
        Concrete JSON-compatible value with defaults filled.
    """
    if isinstance(value, list):
        return [resolve_semantic_value(row, source_dir) for row in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$json"}:
        source = Path(value["$json"])
        if not source.is_absolute():
            source = source_dir / source
        loaded = json.loads(source.read_text(encoding="utf-8"))
        return resolve_semantic_value(loaded, source.parent)
    if "defaults" in value or "overrides" in value:
        if set(value) != {"defaults", "overrides"}:
            raise ValueError(
                "Semantic default/override objects need exactly "
                "defaults and overrides."
            )
        defaults = value["defaults"]
        overrides = value["overrides"]
        if not isinstance(defaults, dict) or not isinstance(
            overrides, dict
        ):
            raise ValueError(
                "Semantic defaults and overrides must be JSON objects."
            )
        merged = merge_semantic_defaults(defaults, overrides)
        return resolve_semantic_value(merged, source_dir)
    return {
        key: resolve_semantic_value(row, source_dir)
        for key, row in value.items()
    }


def semantic_config(entry: dict[str, Any], source_dir: Path) -> dict:
    """Return and validate an entry's resolved semantic configuration.

    Parameters
    ----------
    entry : dict[str, Any]
        Experiment, task, or model specification entry.
    source_dir : Path
        Directory holding the specification.

    Returns
    -------
    dict
        Nonempty resolved semantic configuration.
    """
    raw = entry.get("semantic_config")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Every experiment, task, and model needs semantic_config."
        )
    resolved = resolve_semantic_value(raw, source_dir)
    if not isinstance(resolved, dict) or not resolved:
        raise ValueError("Resolved semantic_config must be nonempty.")
    forbidden = {
        "seed", "seeds", "random_seed", "device", "gpu",
        "subject", "subjects", "save_root", "output_root",
        "workdir", "data_root", "raw_root", "max_parallel",
    }
    def check_keys(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in forbidden:
                    raise ValueError(
                        f"Invocation setting {key!r} cannot appear in "
                        "semantic_config."
                    )
                check_keys(value)
        elif isinstance(node, list):
            for value in node:
                check_keys(value)
    check_keys(resolved)
    json.dumps(resolved, sort_keys=True, allow_nan=False)
    return resolved



def semantic_command(command: list[str]) -> list[str]:
    """Remove invocation controls from a configured launch template.

    Parameters
    ----------
    command : list[str]
        JSON argument vector before subject and seed substitution.

    Returns
    -------
    list[str]
        Remaining command tokens used in semantic identity.
    """
    invocation_flags = {
        "--seed", "--device", "--gpu", "--cuda_device",
        "--subject", "--subjects", "--save_root",
        "--output_root", "--output-root", "--data_root",
        "--raw_root", "--max_parallel", "--max-parallel",
        "--model_kwargs_json", "--config", "--config_json",
    }
    normalized = []
    index = 0
    while index < len(command):
        token = command[index]
        flag = token.split("=", 1)[0]
        if flag in invocation_flags:
            index += 1 if "=" in token else 2
            continue
        if token in ("{seed}", "{subject}", "{save_root}"):
            index += 1
            continue
        normalized.append(token)
        index += 1
    return normalized



def semantic_payload(spec: dict[str, Any], source_dir: Path) -> dict:
    """Build the identity payload without seeds, subjects, or devices.

    Parameters
    ----------
    spec : dict[str, Any]
        Validated experiment specification.
    source_dir : Path
        Directory holding the specification.

    Returns
    -------
    dict
        Fully resolved semantic settings used for the artifact ID.
    """
    tasks = []
    for task in spec["tasks"]:
        models = []
        for model in task["models"]:
            model_semantics = {
                "semantic_config": semantic_config(model, source_dir)
            }
            if "command" in model:
                model_semantics["command_template"] = (
                    semantic_command(model["command"])
                )
            models.append(model_semantics)
        tasks.append({
            "dataset": task.get("dataset"),
            "protocol_id": task["protocol_id"],
            "semantic_config": semantic_config(task, source_dir),
            "models": sorted(
                models, key=lambda row: json.dumps(row, sort_keys=True)
            ),
        })
    return {
        "schema_version": 1,
        "semantic_config": semantic_config(spec, source_dir),
        "tasks": sorted(
            tasks, key=lambda row: json.dumps(row, sort_keys=True)
        ),
    }


def experiment_root(
    base_root: Path,
    spec: dict[str, Any],
    source_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Derive a named artifact root and cryptographic semantic ID.

    Parameters
    ----------
    base_root : Path
        Parent output directory.
    spec : dict[str, Any]
        Validated experiment specification.
    source_dir : Path
        Directory holding the specification.

    Returns
    -------
    tuple[Path, dict[str, Any]]
        Named root and identity record with resolved configuration.
    """
    name = spec["name"]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    payload = semantic_payload(spec, source_dir)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    identifier = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return base_root / f"{slug}__{identifier[:16]}", {
        "name": name,
        "semantic_id": identifier,
        "resolved_config": payload,
    }


def ensure_identity(root: Path, identity: dict[str, Any]) -> None:
    """Create or verify the immutable semantic provenance record.

    Parameters
    ----------
    root : Path
        Named experiment artifact root.
    identity : dict[str, Any]
        Expected semantic ID and fully resolved configuration.
    """
    path = root / "experiment_identity.json"
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != identity:
            raise ValueError(
                f"Existing experiment identity differs: {path}"
            )
    else:
        if root.is_dir() and any(root.iterdir()):
            raise ValueError(
                f"Artifact root has no identity record: {root}"
            )
        write_json(path, identity)


def save_spec_revision(
    root: Path,
    spec: dict[str, Any],
) -> Path:
    """Keep an immutable copy of each invocation specification.

    Parameters
    ----------
    root : Path
        Experiment artifact root.
    spec : dict[str, Any]
        Validated specification including selected seeds and subjects.

    Returns
    -------
    Path
        Absolute revision file path.
    """
    canonical = json.dumps(
        spec, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = root / "spec_versions" / f"{digest}.json"
    if not path.is_file():
        write_json(path, spec)
    return path



def utc_now() -> str:
    """Return the current UTC timestamp.

    Returns
    -------
    str
        ISO 8601 timestamp in UTC.
    """
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Atomically write a JSON artifact.

    Parameters
    ----------
    path : Path
        Absolute destination path.
    value : Any
        JSON-serializable content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_name(value: Any, field: str) -> str:
    """Validate a task or model name used as a directory component.

    Parameters
    ----------
    value : Any
        Candidate name.
    field : str
        Name of the spec field for errors.

    Returns
    -------
    str
        Safe directory component.
    """
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must contain only letters, digits, dot, dash, or "
            f"underscore; got {value!r}."
        )
    return value


def parse_seeds(model: dict[str, Any]) -> list[int]:
    """Read the nonempty seed list for one model.

    Parameters
    ----------
    model : dict[str, Any]
        Model entry from the experiment specification.

    Returns
    -------
    list[int]
        Unique experiment seeds in specification order.
    """
    if "command" in model:
        values = model.get("seeds")
    else:
        values = list(model.get("results", {}))
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"Model {model.get('name')!r} needs at least one seed."
        )
    seeds = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Invalid seed {value!r}.")
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid seed {value!r}.") from exc
        if str(seed) != str(value):
            raise ValueError(f"Seed must be an integer, got {value!r}.")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError(
            f"Model {model.get('name')!r} repeats an experiment seed."
        )
    return seeds


def validate_model(model: Any, task_name: str) -> None:
    """Validate one model entry and its result contract.

    Parameters
    ----------
    model : Any
        Candidate model entry.
    task_name : str
        Parent task identifier.
    """
    if not isinstance(model, dict):
        raise ValueError(f"Task {task_name!r} has a non-object model.")
    name = safe_name(model.get("name"), "model.name")
    has_command = "command" in model
    has_results = "results" in model
    if has_command == has_results:
        raise ValueError(
            f"Model {name!r} needs exactly one of command or results."
        )
    parse_seeds(model)
    if has_command:
        command = model["command"]
        if not isinstance(command, list) or not command:
            raise ValueError(f"Model {name!r} command must be a list.")
        if not all(isinstance(token, str) for token in command):
            raise ValueError(
                f"Model {name!r} command must contain only strings."
            )
        joined = " ".join(command)
        required = ("{seed}", "{save_root}", "{subject}")
        missing = [token for token in required if token not in joined]
        if missing:
            raise ValueError(
                f"Model {name!r} command needs placeholders "
                f"{missing}."
            )
    else:
        results = model["results"]
        if not isinstance(results, dict) or not results:
            raise ValueError(
                f"Model {name!r} results must be a nonempty mapping."
            )
        if not all(isinstance(path, str) and path for path in
                   results.values()):
            raise ValueError(
                f"Model {name!r} result paths must be nonempty strings."
            )
    result_name = model.get("result_file", DEFAULT_RESULT_FILE)
    if not isinstance(result_name, str) or not result_name:
        raise ValueError(
            f"Model {name!r} result_file must be a relative file path."
        )
    result_path = Path(result_name)
    if (result_path.is_absolute() or ".." in result_path.parts
            or result_path.name in ("", ".")
            or result_path.suffix != ".json"):
        raise ValueError(
            f"Model {name!r} result_file must be a relative file path."
        )


def load_spec(path: Path) -> dict[str, Any]:
    """Load and validate the experiment specification.

    Parameters
    ----------
    path : Path
        Absolute path to the specification JSON.

    Returns
    -------
    dict[str, Any]
        Validated specification with task and model entries.
    """
    spec = json.loads(path.read_text(encoding="utf-8"))
    tasks = spec.get("tasks") if isinstance(spec, dict) else None
    name = spec.get("name") if isinstance(spec, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}: experiment name is required.")
    if not re.sub(r"[^a-z0-9]+", "", name.lower()):
        raise ValueError(f"{path}: experiment name needs letters or digits.")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path}: tasks must be a nonempty list.")
    task_names = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"{path}: every task must be an object.")
        name = safe_name(task.get("name"), "task.name")
        if name in task_names:
            raise ValueError(f"{path}: repeated task name {name!r}.")
        task_names.add(name)
        subjects = task.get("subjects")
        if subjects is not None:
            if (not isinstance(subjects, list) or not subjects
                    or not all(
                        isinstance(subject, str)
                        and NAME_PATTERN.fullmatch(subject) is not None
                        for subject in subjects
                    ) or len(subjects) != len(set(subjects))):
                raise ValueError(
                    f"{path}: task {name!r} subjects must be a "
                    "nonempty list of unique subject IDs."
                )
        protocol = task.get("protocol_id")
        if not isinstance(protocol, str) or not protocol.strip():
            raise ValueError(
                f"{path}: task {name!r} needs protocol_id."
            )
        models = task.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError(
                f"{path}: task {name!r} needs at least one model."
            )
        if any(
            isinstance(model, dict) and "command" in model
            for model in models
        ) and not subjects:
            raise ValueError(
                f"{path}: task {name!r} command models need subjects."
            )
        model_names = set()
        for model in models:
            validate_model(model, name)
            model_name = model["name"]
            if model_name in model_names:
                raise ValueError(
                    f"{path}: task {name!r} repeats model "
                    f"{model_name!r}."
                )
            model_names.add(model_name)
    semantic_payload(spec, path.parent)
    return spec


def result_path(
    spec_path: Path,
    output_root: Path,
    task: dict[str, Any],
    model: dict[str, Any],
    seed: int,
    subject: str | None = None,
) -> Path:
    """Resolve one seed's expected result JSON path.

    Parameters
    ----------
    spec_path : Path
        Absolute specification path.
    output_root : Path
        Absolute coordinated output directory.
    task, model : dict[str, Any]
        Validated specification entries.
    seed : int
        Training seed.
    subject : str or None, optional
        Subject ID for a launched run, default None.

    Returns
    -------
    Path
        Absolute result path.
    """
    if "results" in model:
        source = Path(model["results"][str(seed)])
        if not source.is_absolute():
            source = spec_path.parent / source
        return source.resolve()
    if subject is None:
        raise ValueError(
            f"Model {model['name']!r} needs a subject for "
            "its launched result path."
        )
    run_root = (
        output_root / task["name"] / model["name"]
        / "subjects" / subject / f"seed{seed}"
    )
    return run_root / model.get(
        "result_file", DEFAULT_RESULT_FILE
    )


def load_split_scores(
    path: Path,
    dataset: str | None,
) -> dict[str, dict[str, float | int]]:
    """Read per-subject held-out regression scores from one seed.

    Parameters
    ----------
    path : Path
        Absolute result JSON path.
    dataset : str or None
        Expected dataset tag, if the task specifies one.

    Returns
    -------
    dict[str, dict[str, float | int]]
        Subject IDs mapped to finite correlation and positive test count.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    recorded_dataset = result.get("dataset")
    if (dataset is not None and recorded_dataset is not None
            and recorded_dataset != dataset):
        raise ValueError(
            f"{path}: expected dataset {dataset!r}, "
            f"got {recorded_dataset!r}."
        )
    subjects = result.get("per_subject")
    if not isinstance(subjects, dict) or not subjects:
        raise ValueError(
            f"{path}: per_subject must contain at least one split."
        )
    scores = {}
    for subject, record in subjects.items():
        if not isinstance(record, dict):
            raise ValueError(
                f"{path}: subject {subject!r} needs a result object."
            )
        try:
            score = float(record["corr_mean"])
            raw_count = record["n"]
            count = int(raw_count)
            if isinstance(raw_count, bool) or count != raw_count:
                raise ValueError("test count is not an integer")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: subject {subject!r} needs finite "
                "corr_mean and positive n."
            ) from exc
        if not math.isfinite(score) or count <= 0:
            raise ValueError(
                f"{path}: subject {subject!r} has "
                f"corr_mean={score!r}, n={count!r}."
            )
        scores[str(subject)] = {"score": score, "n": count}
    return scores



def load_seed_scores(
    spec_path: Path,
    root: Path,
    task: dict[str, Any],
    model: dict[str, Any],
    seed: int,
) -> tuple[dict[str, dict[str, float | int]], Any]:
    """Load one seed, joining separately saved subject results.

    Parameters
    ----------
    spec_path : Path
        Absolute experiment specification path.
    root : Path
        Named experiment artifact root.
    task, model : dict[str, Any]
        Validated task and model entries.
    seed : int
        Training seed.

    Returns
    -------
    tuple[dict, Any]
        Subject scores and provenance paths for one seed.
    """
    if "results" in model:
        path = result_path(spec_path, root, task, model, seed)
        return load_split_scores(path, task.get("dataset")), str(path)
    scores = {}
    sources = {}
    for subject in task["subjects"]:
        path = result_path(
            spec_path, root, task, model, seed, subject
        )
        one = load_split_scores(path, task.get("dataset"))
        if set(one) != {subject}:
            raise ValueError(
                f"{path}: expected only subject {subject!r}, "
                f"got {sorted(one)}."
            )
        scores[subject] = one[subject]
        sources[subject] = str(path)
    return scores, sources



def check_splits(
    reference: dict[str, int],
    candidate: dict[str, dict[str, float | int]],
    path: Path,
) -> None:
    """Require exactly the same held-out subjects and test counts.

    Parameters
    ----------
    reference : dict[str, int]
        Subject IDs mapped to expected test-window counts.
    candidate : dict[str, dict[str, float | int]]
        Candidate subject scores and counts.
    path : Path
        Source path for an actionable error.
    """
    actual = {subject: int(row["n"]) for subject, row in
              candidate.items()}
    if actual != reference:
        missing = sorted(set(reference) - set(actual))
        extra = sorted(set(actual) - set(reference))
        changed = {
            key: [reference[key], actual[key]]
            for key in reference.keys() & actual.keys()
            if reference[key] != actual[key]
        }
        raise ValueError(
            f"{path}: held-out splits do not match: "
            f"missing={missing}, extra={extra}, "
            f"test-count changes={changed}."
        )


def sample_sd(values: list[float]) -> float | None:
    """Return sample SD, or None when only one observation exists.

    Parameters
    ----------
    values : list[float]
        Finite observations.

    Returns
    -------
    float or None
        Sample standard deviation for at least two observations.
    """
    if not values:
        raise ValueError("Cannot summarize an empty observation list.")
    return stdev(values) if len(values) > 1 else None


def summarize_model(
    seeds: dict[int, dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    """Aggregate seed scores within each held-out subject.

    Parameters
    ----------
    seeds : dict[int, dict[str, dict[str, float | int]]]
        Seed by subject by scalar score and test count.

    Returns
    -------
    dict[str, Any]
        Per-subject seed means/SDs and cohort mean/SDs.
    """
    if not seeds:
        raise ValueError("Cannot summarize a model without seeds.")
    ordered_seeds = sorted(seeds)
    subjects = sorted(next(iter(seeds.values())))
    per_subject = {}
    for subject in subjects:
        values = [float(seeds[seed][subject]["score"])
                  for seed in ordered_seeds]
        per_subject[subject] = {
            "mean": mean(values),
            "seed_sd": sample_sd(values),
            "by_seed": dict(zip(map(str, ordered_seeds), values)),
            "n_test": int(seeds[ordered_seeds[0]][subject]["n"]),
        }
    subject_means = [per_subject[subject]["mean"]
                     for subject in subjects]
    seed_cohorts = {
        str(seed): mean(
            float(seeds[seed][subject]["score"])
            for subject in subjects
        )
        for seed in ordered_seeds
    }
    return {
        "seeds": ordered_seeds,
        "n_seeds": len(ordered_seeds),
        "n_subjects": len(subjects),
        "mean": mean(subject_means),
        "subject_sd": sample_sd(subject_means),
        "seed_cohort_mean": mean(list(seed_cohorts.values())),
        "seed_cohort_sd": sample_sd(list(seed_cohorts.values())),
        "seed_cohort_scores": seed_cohorts,
        "per_subject": per_subject,
    }


def compare_models(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    """Test paired subject-level differences after averaging seeds.

    Parameters
    ----------
    first, second : dict[str, Any]
        Model summaries with identical held-out subject IDs.

    Returns
    -------
    dict[str, Any]
        Effect size and two-sided paired Wilcoxon result. Subjects are the
        independent test units; seeds are repeated fits, not extra subjects.
    """
    subjects = sorted(first["per_subject"])
    if subjects != sorted(second["per_subject"]):
        raise ValueError("Paired models have different subject IDs.")
    if len(subjects) < 2:
        raise ValueError(
            "Paired testing needs at least two held-out subjects."
        )
    differences = [
        first["per_subject"][subject]["mean"]
        - second["per_subject"][subject]["mean"]
        for subject in subjects
    ]
    if all(value == 0 for value in differences):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    if not math.isfinite(p_value):
        raise ValueError("Paired Wilcoxon returned a non-finite p-value.")
    ordered = sorted(differences)
    midpoint = len(ordered) // 2
    median_difference = (
        ordered[midpoint]
        if len(ordered) % 2
        else mean(ordered[midpoint - 1:midpoint + 1])
    )
    return {
        "n_subjects": len(subjects),
        "mean_difference": mean(differences),
        "median_difference": median_difference,
        "by_subject_difference": dict(zip(subjects, differences)),
        "test": "two-sided paired Wilcoxon",
        "statistic": statistic,
        "p_raw": p_value,
    }


def adjust_holm(rows: list[dict[str, Any]]) -> None:
    """Add Holm-adjusted p-values within one task.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Mutable comparison rows for a single task.
    """
    previous = 0.0
    total = len(rows)
    for rank, row in enumerate(
        sorted(rows, key=lambda item: item["p_raw"])
    ):
        previous = max(
            previous,
            min(1.0, (total - rank) * row["p_raw"]),
        )
        row["p_holm"] = previous



def effective_runner_args(sources: dict[str, Any]) -> dict | None:
    """Check and return the effective runner defaults in job results.

    Parameters
    ----------
    sources : dict[str, Any]
        Seed to subject to absolute result JSON path.

    Returns
    -------
    dict or None
        Stable effective arguments without invocation fields, if saved.
    """
    invocation = {
        "seed", "device", "gpu", "subjects", "subject",
        "save_root", "output_root", "data_root", "raw_root",
        "model_kwargs_json", "max_parallel",
    }
    seen = []
    missing = []
    for by_subject in sources.values():
        for path_text in by_subject.values():
            path = Path(path_text)
            record = json.loads(path.read_text(encoding="utf-8"))
            args = record.get("args")
            if args is None:
                missing.append(str(path))
                continue
            if not isinstance(args, dict):
                raise ValueError(f"{path}: args must be a JSON object.")
            stable = {
                key: value for key, value in args.items()
                if key not in invocation
            }
            seen.append((path, stable))
    if seen and missing:
        raise ValueError(
            "Some launched results omit effective args: "
            f"{missing}"
        )
    if not seen:
        return None
    reference = seen[0][1]
    for path, candidate in seen[1:]:
        if candidate != reference:
            raise ValueError(
                f"Effective runner arguments differ at {path}."
            )
    return reference



def evaluate(spec_path: Path, output_root: Path) -> dict[str, Any]:
    """Aggregate declared splits within a named semantic artifact root.

    Parameters
    ----------
    spec_path : Path
        Absolute specification path.
    output_root : Path
        Parent directory for named experiment artifacts.

    Returns
    -------
    dict[str, Any]
        Named artifact root, descriptive aggregate, and paired tests.
    """
    spec = load_spec(spec_path)
    root, identity = experiment_root(
        output_root, spec, spec_path.parent
    )
    ensure_identity(root, identity)
    save_spec_revision(root, spec)
    aggregate = {
        "created_at": utc_now(),
        "experiment": identity,
        "tasks": {},
    }
    comparisons = {
        "created_at": utc_now(),
        "semantic_id": identity["semantic_id"],
        "tasks": {},
    }
    for task in spec["tasks"]:
        task_name = task["name"]
        reference = None
        task_models = {}
        for model in task["models"]:
            seed_scores = {}
            seed_sources = {}
            for seed in parse_seeds(model):
                scores, sources = load_seed_scores(
                    spec_path, root, task, model, seed
                )
                source_path = Path(
                    sources if isinstance(sources, str)
                    else next(iter(sources.values()))
                )
                if reference is None:
                    if task.get("subjects") is not None:
                        expected = set(task["subjects"])
                        actual = set(scores)
                        if actual != expected:
                            raise ValueError(
                                f"{source_path}: subject split mismatch: "
                                f"missing={sorted(expected - actual)}, "
                                f"extra={sorted(actual - expected)}."
                            )
                    reference = {
                        key: int(row["n"])
                        for key, row in scores.items()
                    }
                else:
                    check_splits(reference, scores, source_path)
                seed_scores[seed] = scores
                seed_sources[str(seed)] = sources
            summary = summarize_model(seed_scores)
            summary["result_files"] = seed_sources
            if "command" in model:
                summary["effective_runner_args"] = (
                    effective_runner_args(seed_sources)
                )
            task_models[model["name"]] = summary
        aggregate["tasks"][task_name] = {
            "dataset": task.get("dataset"),
            "protocol_id": task["protocol_id"],
            "test_counts": reference,
            "models": task_models,
        }
        rows = []
        for first_name, second_name in combinations(task_models, 2):
            result = compare_models(
                task_models[first_name], task_models[second_name]
            )
            result["model_a"] = first_name
            result["model_b"] = second_name
            rows.append(result)
        adjust_holm(rows)
        comparisons["tasks"][task_name] = rows
    write_json(root / AGGREGATE_FILE, aggregate)
    write_json(root / COMPARISONS_FILE, comparisons)
    return {
        "artifact_root": str(root),
        "aggregate": aggregate,
        "comparisons": comparisons,
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Run one seed and capture its complete stdout/stderr.

    Parameters
    ----------
    job : dict[str, Any]
        Absolute paths, argument vector, and identifiers for one seed.

    Returns
    -------
    dict[str, Any]
        Job status and timestamps.
    """
    run_root = Path(job["run_root"])
    run_root.mkdir(parents=True)
    log_path = run_root / "run.log"
    record = dict(job)
    record["started_at"] = utc_now()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            job["command"],
            cwd=job["workdir"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record["ended_at"] = utc_now()
    record["returncode"] = completed.returncode
    record["log_file"] = str(log_path)
    expected = Path(job["result_file"])
    record["status"] = (
        "complete"
        if completed.returncode == 0 and expected.is_file()
        else "failed"
    )
    return record


def prepare_jobs(
    spec_path: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Select only new subject and seed runs for a semantic artifact.

    Parameters
    ----------
    spec_path : Path
        Absolute specification path.
    output_root : Path
        Parent directory for named experiment artifacts.

    Returns
    -------
    list[dict[str, Any]]
        Pending jobs with disjoint subject and seed directories.
    """
    spec = load_spec(spec_path)
    root, identity = experiment_root(
        output_root, spec, spec_path.parent
    )
    identity_path = root / "experiment_identity.json"
    if identity_path.is_file():
        actual = json.loads(identity_path.read_text(encoding="utf-8"))
        if actual != identity:
            raise ValueError(
                f"Existing experiment identity differs: {identity_path}"
            )
    jobs = []
    for task in spec["tasks"]:
        for model in task["models"]:
            if "command" not in model:
                continue
            workdir = Path(model.get("workdir", REPO_ROOT)).resolve()
            if not workdir.is_dir():
                raise NotADirectoryError(
                    f"Model {model['name']!r} workdir is missing: "
                    f"{workdir}"
                )
            for subject in task["subjects"]:
                for seed in parse_seeds(model):
                    expected = result_path(
                        spec_path, root, task, model, seed, subject
                    )
                    run_root = (
                        root / task["name"] / model["name"]
                        / "subjects" / subject / f"seed{seed}"
                    )
                    if run_root.exists():
                        if expected.is_file():
                            scores = load_split_scores(
                                expected, task.get("dataset")
                            )
                            if set(scores) != {subject}:
                                raise ValueError(
                                    f"{expected}: expected only "
                                    f"subject {subject!r}."
                                )
                            continue
                        raise FileExistsError(
                            f"Incomplete run directory: {run_root}. "
                            "Preserve it and use a new attempt path "
                            "after inspecting its artifacts."
                        )
                    command = [
                        token.replace("{seed}", str(seed))
                        .replace("{subject}", subject)
                        .replace("{save_root}", str(run_root))
                        for token in model["command"]
                    ]
                    jobs.append({
                        "task": task["name"],
                        "model": model["name"],
                        "subject": subject,
                        "seed": seed,
                        "command": command,
                        "workdir": str(workdir),
                        "run_root": str(run_root),
                        "result_file": str(expected),
                    })
    return jobs


def run_campaign(
    spec_path: Path,
    output_root: Path,
    max_parallel: int,
) -> dict[str, Any]:
    """Launch missing subject/seed jobs and preserve previous runs.

    Parameters
    ----------
    spec_path : Path
        Absolute specification path.
    output_root : Path
        Parent directory for named experiment artifacts.
    max_parallel : int
        Maximum simultaneous training processes.

    Returns
    -------
    dict[str, Any]
        Reports from the complete declared campaign.
    """
    if max_parallel <= 0:
        raise ValueError(
            f"max_parallel must be positive, got {max_parallel}."
        )
    spec = load_spec(spec_path)
    root, identity = experiment_root(
        output_root, spec, spec_path.parent
    )
    jobs = prepare_jobs(spec_path, output_root)
    ensure_identity(root, identity)
    revision = save_spec_revision(root, spec)
    manifest_path = root / MANIFEST_FILE
    if manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.get("semantic_id") != identity["semantic_id"]:
            raise ValueError(
                f"Run manifest has a different identity: {manifest_path}"
            )
    else:
        manifest = {
            "created_at": utc_now(),
            "semantic_id": identity["semantic_id"],
            "jobs": [],
            "spec_revisions": [],
        }
    if str(revision) not in manifest["spec_revisions"]:
        manifest["spec_revisions"].append(str(revision))
    write_json(manifest_path, manifest)
    new_records = []
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            manifest["jobs"].append(record)
            new_records.append(record)
            write_json(manifest_path, manifest)
            print(
                f"{record['task']}/{record['model']}/"
                f"{record['subject']}/seed{record['seed']}: "
                f"{record['status']} ({record['log_file']})",
                flush=True,
            )
    failures = [
        row for row in new_records
        if row["status"] != "complete"
    ]
    if failures:
        raise RuntimeError(
            f"{len(failures)} seed run(s) failed; inspect "
            f"{manifest_path}."
        )
    return evaluate(spec_path, output_root)


def print_report(report: dict[str, Any]) -> None:
    """Print task summaries and absolute report locations.

    Parameters
    ----------
    report : dict[str, Any]
        Aggregate and comparison reports.
    """
    for task_name, task in report["aggregate"]["tasks"].items():
        for model_name, row in task["models"].items():
            print(
                f"{task_name}/{model_name}: mean={row['mean']:.4f}, "
                f"subject_sd={row['subject_sd']}, "
                f"seed_sd={row['seed_cohort_sd']}, "
                f"subjects={row['n_subjects']}, "
                f"seeds={row['n_seeds']}"
            )
        for row in report["comparisons"]["tasks"][task_name]:
            print(
                f"{task_name}: {row['model_a']} vs "
                f"{row['model_b']}: mean_delta="
                f"{row['mean_difference']:.4f}, "
                f"p={row['p_raw']:.4g}, "
                f"Holm p={row['p_holm']:.4g}"
            )
    root = Path(report["artifact_root"])
    print(root / AGGREGATE_FILE)
    print(root / COMPARISONS_FILE)
    print(root / "experiment_identity.json")


def main() -> None:
    """Parse command-line arguments and run or evaluate a campaign."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("run", "evaluate"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=1)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    output_root = args.output_root.resolve()
    if args.action == "run":
        report = run_campaign(
            spec_path, output_root, args.max_parallel
        )
    else:
        report = evaluate(spec_path, output_root)
    print_report(report)


if __name__ == "__main__":
    try:
        main()
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
