"""Read Stanford split settings and locate its chronological boundary.

Input: ``preprocessed_data/Stanford/resolved_config.json`` (when a prepared
root is given), or ``configs/stanford_preprocessing.json`` for defaults.
Output: a validated split dictionary and the training boundary in samples.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs" / "stanford_preprocessing.json"
)
PROVENANCE_NAME = "resolved_config.json"


def _merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict:
    """Resolve known nested overrides without dropping default values.

    Parameters
    ----------
    defaults : dict
        Default preprocessing settings.
    overrides : dict
        Nested settings to replace.

    Returns
    -------
    dict
        Fully resolved preprocessing settings.
    """
    resolved = defaults.copy()
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f"Unknown Stanford preprocessing override: {key}")
        if isinstance(defaults[key], dict) and isinstance(value, dict):
            resolved[key] = _merge(defaults[key], value)
        else:
            resolved[key] = value
    return resolved


def load_split_config(prepared_root: Path | None = None) -> dict:
    """Load the split rules used by a prepared Stanford dataset.

    Parameters
    ----------
    prepared_root : Path, optional
        Folder with resolved preprocessing provenance. If omitted, resolve
        defaults and overrides from the repository config.

    Returns
    -------
    dict
        Threshold and duration-specific split settings.
    """
    if prepared_root is None:
        source = json.loads(CONFIG_PATH.read_text())
        config = _merge(source["defaults"], source["overrides"])
    else:
        path = Path(prepared_root) / PROVENANCE_NAME
        config = json.loads(path.read_text())
    split = config["split"]
    cutoff = split["duration_cutoff_seconds"]
    train = split["at_or_above_cutoff"]["train_seconds"]
    parts = split["below_cutoff"]["train_parts"]
    total = split["below_cutoff"]["total_parts"]
    settings = (cutoff, train, parts, total)
    if not all(isinstance(value, int) for value in settings):
        raise ValueError("Stanford split settings must be integer values.")
    if not 0 < train < cutoff or not 0 < parts < total:
        raise ValueError("Expected valid long durations and short parts.")
    return split


def split_boundary(
    total_samples: int, sampling_hz: int, split: dict
) -> tuple[int, str]:
    """Find the training boundary with the released chronological rule.

    Parameters
    ----------
    total_samples : int
        Number of samples in the recording.
    sampling_hz : int
        ECoG sample rate in Hz.
    split : dict
        Resolved duration-specific split settings.

    Returns
    -------
    tuple[int, str]
        Exclusive training endpoint and applied rule name.
    """
    if total_samples <= 0 or sampling_hz <= 0:
        raise ValueError("Recording length and sample rate must be positive.")
    if total_samples // sampling_hz >= split["duration_cutoff_seconds"]:
        boundary = (
            split["at_or_above_cutoff"]["train_seconds"]
            * sampling_hz
        )
        rule = "at_or_above_cutoff"
    else:
        below = split["below_cutoff"]
        boundary = (
            total_samples // below["total_parts"]
        ) * below["train_parts"]
        rule = "below_cutoff"
    if boundary <= 0 or boundary >= total_samples:
        raise ValueError(
            f"Invalid Stanford split at sample {boundary} of "
            f"{total_samples}."
        )
    return boundary, rule
