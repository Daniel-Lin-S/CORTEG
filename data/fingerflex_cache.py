"""Resolve Stanford preprocessing and share prepared samples across runs.

Input: ``configs/fingerflex_campaign.json`` referencing
``configs/stanford_preprocessing.json``; registered raw MATLAB files; and,
when available, a completed legacy campaign's ``prepared/`` directory.
Output: ``<artifact_base>/preprocessing/<preprocessing-id>/index.json`` and
``subjects/<subject>/`` feature arrays. Legacy arrays are linked per subject,
not copied. A training artifact may link its ``prepared/`` entry to this
shared ``subjects/`` directory. Raw MATLAB files are read only.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from data.fingerflex_campaign import (
    EC0G_HZ,
    FEATURE_HZ,
    FINGER_HZ,
    MAX_DELAY,
    STRIDE,
    WINDOW,
    prepare_subject,
)
from data.stanford_split import _merge

REPO = Path(__file__).resolve().parents[1]
PREPARATION_VERSION = "fingerflex_fieldtrip_onepass_hgalfs_v1"
LEGACY_PREPARATION_SHA256 = (
    "a9b3f293caf7138f45b571c0e7fb6ff2271b74b301050e16d27f567179276df4"
)
REQUIRED_PREPROCESSING_KEYS = (
    "ecog_scale", "split", "notch_bands_hz", "notch_filter_order",
    "bad_channel",
)
SUPPORTED_FILTER = ("but", "onepass", "all", "avg")
ARRAYS = {
    "common": ("metadata.json", "train_y.npy", "test_y.npy",
               "train_indices.npy", "test_indices.npy", "xyz_mm.npy"),
    "corteg": ("train_x.npy", "test_x.npy", "train_hi.npy", "test_hi.npy"),
    "hilo": ("train_x.npy", "test_x.npy"),
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-memory blocks.

    Parameters
    ----------
    path : Path
        Existing input file.

    Returns
    -------
    str
        SHA256 hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_campaign(path: Path) -> dict:
    """Combine training settings with resolved preprocessing settings.

    Parameters
    ----------
    path : Path
        JSON campaign configuration.

    Returns
    -------
    dict
        Full configuration with ``preprocessing`` resolved from its own file.
    """
    config = json.loads(path.read_text())
    if "preprocessing" in config:
        raise ValueError(
            f"Move preprocessing settings out of training config: {path}."
        )
    source = Path(config["preprocessing_config"])
    if not source.is_absolute():
        source = REPO / source
    if not source.is_file():
        raise FileNotFoundError(f"Preprocessing config is missing: {source}.")
    document = json.loads(source.read_text())
    resolved = _merge(document["defaults"], document["overrides"])
    missing = set(REQUIRED_PREPROCESSING_KEYS) - set(resolved)
    if missing:
        raise ValueError(f"Missing preprocessing settings: {sorted(missing)}.")
    filter_mode = (
        resolved["notch_filter_family"],
        resolved["notch_filter_direction"],
        resolved["reference"]["channels"],
        resolved["reference"]["method"],
    )
    if filter_mode != SUPPORTED_FILTER:
        raise ValueError(
            f"Python preparation requires {SUPPORTED_FILTER}, "
            f"got {filter_mode}."
        )
    config["preprocessing_config"] = str(source.resolve())
    config["preprocessing"] = resolved
    return config


def preparation_semantics(config: dict) -> dict:
    """Return only settings that determine prepared sample values.

    Parameters
    ----------
    config : dict
        Resolved campaign configuration.

    Returns
    -------
    dict
        Stable preprocessing identity, independent of training settings.
    """
    return {
        "version": PREPARATION_VERSION,
        "preprocessing": config["preprocessing"],
        "sample_grid": {
            "ecog_hz": EC0G_HZ,
            "finger_hz": FINGER_HZ,
            "window_samples": WINDOW,
            "stride_samples": STRIDE,
            "max_delay_samples": MAX_DELAY,
            "feature_hz": FEATURE_HZ,
        },
    }


def preparation_id(config: dict) -> str:
    """Hash preprocessing semantics without training or file locations."""
    canonical = json.dumps(
        preparation_semantics(config), sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def cache_root(config: dict) -> Path:
    """Return the shared preprocessing-cache directory."""
    return (
        Path(config["artifact_base"]) / "preprocessing"
        / preparation_id(config)
    )


def _atomic_json(path: Path, contents: dict) -> None:
    """Replace a JSON file only after its full contents are written."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(contents, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive file lock during one cache-index mutation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepared_complete(
    subject_root: Path,
    delays: tuple[int, ...],
) -> bool:
    """Check all arrays required for a subject's selected delays.

    Parameters
    ----------
    subject_root : Path
        Prepared subject directory.
    delays : tuple[int, ...]
        Requested ECoG shifts in samples.

    Returns
    -------
    bool
        Whether every common and feature file is present and nonempty.
    """
    paths = [subject_root / "common" / name for name in ARRAYS["common"]]
    for delay in delays:
        paths.extend(
            subject_root / "corteg" / str(delay) / name
            for name in ARRAYS["corteg"]
        )
    paths.extend(
        subject_root / "hilo" / "0" / name for name in ARRAYS["hilo"]
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _legacy_compatible(candidate: Path, config: dict) -> bool:
    """Verify a completed old artifact before linking its sample arrays."""
    provenance = candidate / "provenance"
    paths = (
        provenance / "resolved_config.json",
        provenance / "semantics.json",
        provenance / "input_hashes.json",
    )
    if not all(path.is_file() for path in paths):
        return False
    old_config = json.loads(paths[0].read_text())
    old_preprocessing = old_config.get("preprocessing", {})
    effective = {
        key: config["preprocessing"][key]
        for key in REQUIRED_PREPROCESSING_KEYS
    }
    if old_preprocessing != effective:
        return False
    semantics = json.loads(paths[1].read_text())
    source_hashes = semantics.get("campaign_source_sha256", {})
    if source_hashes.get("data/fingerflex_campaign.py") != (
        LEGACY_PREPARATION_SHA256
    ):
        return False
    hashes = json.loads(paths[2].read_text()).get("raw_sha256", {})
    if not hashes:
        return False
    for subject in hashes:
        delays = (0, 40) if subject in old_config.get(
            "delay_subjects", []
        ) else (0,)
        if not prepared_complete(
            candidate / "prepared" / subject, delays
        ):
            return False
    return True


def _adopt_legacy(cache: Path, config: dict) -> dict:
    """Link verified legacy arrays into a new shared cache without copying."""
    base = Path(config["artifact_base"])
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir() or candidate.name == "preprocessing":
            continue
        if not _legacy_compatible(candidate, config):
            continue
        inputs = json.loads(
            (candidate / "provenance" / "input_hashes.json").read_text()
        )
        for subject in inputs["raw_sha256"]:
            source = candidate / "prepared" / subject
            target = cache / "subjects" / subject
            (target / "corteg").mkdir(parents=True)
            (target / "hilo").mkdir()
            (target / "common").symlink_to(source / "common")
            for model in ("corteg", "hilo"):
                for delay in (source / model).iterdir():
                    (target / model / delay.name).symlink_to(delay)
        return {
            "adopted_from": str(candidate),
            "raw_sha256": inputs["raw_sha256"],
            "legacy_source_sha256": LEGACY_PREPARATION_SHA256,
        }
    return {"adopted_from": None, "raw_sha256": {}}


def ensure_cache(config: dict) -> Path:
    """Initialize or validate a preparation cache for resolved settings.

    Parameters
    ----------
    config : dict
        Resolved campaign config with preprocessing loaded separately.

    Returns
    -------
    Path
        Shared ``subjects`` directory for prepared data.
    """
    cache = cache_root(config)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "subjects").mkdir(exist_ok=True)
    with _locked(cache / "locks" / "index.lock"):
        index_path = cache / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index["semantics"] != preparation_semantics(config):
                raise ValueError(
                    f"Preprocessing identity conflict: {index_path}."
                )
            current_source = sha256_file(
                REPO / "data" / "fingerflex_campaign.py"
            )
            if index["producer_source_sha256"] != current_source:
                raise ValueError(
                    "Preparation code changed for existing cache "
                    f"{cache}; review the change and bump "
                    "PREPARATION_VERSION if sample values can differ."
                )
        else:
            adoption = _adopt_legacy(cache, config)
            index = {
                "preprocessing_id": preparation_id(config),
                "semantics": preparation_semantics(config),
                "producer_source_sha256": sha256_file(
                    REPO / "data" / "fingerflex_campaign.py"
                ),
                **adoption,
            }
            _atomic_json(index_path, index)
    return cache / "subjects"


def prepare_cached_subject(
    subject: str,
    config: dict,
    delays: tuple[int, ...],
) -> bool:
    """Reuse complete arrays or prepare missing variants once under a lock.

    Parameters
    ----------
    subject : str
        Patient code.
    config : dict
        Resolved campaign configuration.
    delays : tuple[int, ...]
        Requested neural shifts in samples.

    Returns
    -------
    bool
        True if all requested arrays were already present.
    """
    subjects = ensure_cache(config)
    cache = subjects.parent
    source = (
        Path(config["raw_root"]) / subject / f"{subject}_fingerflex.mat"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Raw patient file is missing: {source}.")
    with _locked(cache / "locks" / f"{subject}.lock"):
        index_path = cache / "index.json"
        index = json.loads(index_path.read_text())
        raw_hash = sha256_file(source)
        previous = index["raw_sha256"].get(subject)
        if previous is not None and previous != raw_hash:
            raise ValueError(
                f"Raw file changed for {subject}: {source}."
            )
        output = subjects / subject
        reused = prepared_complete(output, delays)
        if not reused:
            prepare_subject(
                subject, Path(config["raw_root"]), subjects,
                config, delays,
            )
            if not prepared_complete(output, delays):
                raise RuntimeError(
                    f"Preparation did not produce all arrays: {output}."
                )
        if previous is None:
            with _locked(cache / "locks" / "index.lock"):
                index = json.loads(index_path.read_text())
                index["raw_sha256"][subject] = raw_hash
                _atomic_json(index_path, index)
        return reused
