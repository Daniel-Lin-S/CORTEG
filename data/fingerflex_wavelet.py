"""Cache pooled Stanford Morlet features for deterministic PLS fits.

Input is the registered raw MATLAB file and the existing common sample grid
under the shared preparation cache. Output is train/test feature arrays of
shape (windows, retained channels, 10 time bins, 15 frequencies), plus a
recipe manifest under each subject's pls/<recipe-id>/0 directory.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter

from data.fingerflex_campaign import (
    EC0G_HZ,
    MAX_DELAY,
    WINDOW,
    clean_recording,
    digest_array,
)
from data.fingerflex_cache import (
    _locked,
    cache_root,
    prepare_cached_subject,
    sha256_file,
)

FREQ_COUNT = 15
FREQ_LOW = 10.0
FREQ_HIGH = 150.0
FEATURE_HZ = 100
POOLED_BINS = 10
DOWNSAMPLE = 2
LOWPASS_HZ = 200.0
LOWPASS_ORDER = 4
BATCH_SIZE = 16
RECIPE_VERSION = "stanford_pls_morlet_magnitude_v1"


def recipe() -> dict:
    """Return the complete feature recipe used for cache identity."""
    return {
        "version": RECIPE_VERSION,
        "frequencies_hz": np.linspace(
            FREQ_LOW, FREQ_HIGH, FREQ_COUNT,
        ).tolist(),
        "cycles": "frequency / 5",
        "morlet_output": "complex_magnitude",
        "window_samples": WINDOW,
        "lowpass_hz": LOWPASS_HZ,
        "lowpass_order": LOWPASS_ORDER,
        "downsample_factor": DOWNSAMPLE,
        "time_bins": FEATURE_HZ,
        "pooled_bins": POOLED_BINS,
    }


def recipe_id() -> str:
    """Hash the wavelet recipe independently of training choices."""
    payload = json.dumps(recipe(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def feature_root(config: dict, subject: str) -> Path:
    """Locate one subject's wavelet cache without local tracked paths."""
    return (
        cache_root(config) / "subjects" / subject / "pls"
        / recipe_id() / "0"
    )


def _write_features(
    path: Path,
    signal: np.ndarray,
    grid: np.ndarray,
) -> None:
    """Extract windows in batches and atomically save pooled features."""
    frequencies = np.linspace(FREQ_LOW, FREQ_HIGH, FREQ_COUNT)
    cycles = frequencies / 5.0
    n_channels = signal.shape[1]
    shape = (len(grid), n_channels, POOLED_BINS, FREQ_COUNT)
    temporary = path.with_suffix(".tmp.npy")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape,
    )
    window_ds = WINDOW // DOWNSAMPLE
    pool_width = FEATURE_HZ // POOLED_BINS
    try:
        for start in range(0, len(grid), BATCH_SIZE):
            ends = grid[start:start + BATCH_SIZE] // DOWNSAMPLE
            batch = np.stack([
                signal[end - window_ds:end].T for end in ends
            ])
            if batch.shape[-1] != window_ds:
                raise ValueError(
                    f"Expected {window_ds} downsampled samples, "
                    f"got {batch.shape[-1]}."
                )
            wavelet = mne.time_frequency.tfr_array_morlet(
                batch,
                sfreq=EC0G_HZ / DOWNSAMPLE,
                freqs=frequencies,
                n_cycles=cycles,
                output="complex",
                n_jobs=1,
                verbose=False,
            )
            magnitude = np.abs(wavelet)
            first = magnitude.reshape(
                len(ends), n_channels, FREQ_COUNT,
                FEATURE_HZ, window_ds // FEATURE_HZ,
            ).mean(axis=-1)
            pooled = first.reshape(
                len(ends), n_channels, FREQ_COUNT,
                POOLED_BINS, pool_width,
            ).mean(axis=-1)
            output[start:start + len(ends)] = pooled.transpose(
                0, 1, 3, 2,
            )
        output.flush()
        if not np.isfinite(output).all():
            raise ValueError(f"Nonfinite wavelet features for {path}.")
        del output
        temporary.replace(path)
    except Exception:
        del output
        temporary.unlink(missing_ok=True)
        raise


def prepare_wavelet(config: dict, subject: str) -> bool:
    """Reuse or prepare PLS arrays on the shared chronological target grid.

    Parameters
    ----------
    config : dict
        Resolved campaign settings with raw root and preprocessing policy.
    subject : str
        Patient identifier.

    Returns
    -------
    bool
        Whether both complete feature arrays were already cached.
    """
    prepare_cached_subject(subject, config, (0,))
    root = feature_root(config, subject)
    common = root.parents[2] / "common"
    source = (
        Path(config["raw_root"]) / subject / f"{subject}_fingerflex.mat"
    )
    with _locked(cache_root(config) / "locks" / f"{subject}.lock"):
        manifest_path = root / "manifest.json"
        files = [root / f"{part}_x.npy" for part in ("train", "test")]
        raw_hash = sha256_file(source)
        if manifest_path.is_file() and all(
            path.is_file() and path.stat().st_size > 0 for path in files
        ):
            manifest = json.loads(manifest_path.read_text())
            if manifest != {
                "recipe": recipe(),
                "producer_sha256": sha256_file(Path(__file__)),
                "raw_sha256": raw_hash,
                "target_grid_sha256": {
                    part: digest_array(np.load(
                        common / f"{part}_indices.npy"
                    )) for part in ("train", "test")
                },
            }:
                raise ValueError(
                    f"Existing PLS feature manifest differs: {manifest_path}."
                )
            return True
        if root.exists():
            quarantine = (
                cache_root(config) / "quarantine"
                / f"{subject}_pls_{recipe_id()}_{time.time_ns()}"
            )
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            root.rename(quarantine)
        recording = loadmat(source)
        metadata = json.loads((common / "metadata.json").read_text())
        boundary = metadata["split_boundary"]
        clean, kept, _ = clean_recording(
            recording["data"], boundary, config["preprocessing"],
        )
        if kept.tolist() != metadata["kept_channels_zero_based"]:
            raise ValueError(f"PLS channel decisions differ for {subject}.")
        root.mkdir(parents=True, exist_ok=True)
        low_b, low_a = butter(
            LOWPASS_ORDER, LOWPASS_HZ, btype="low", fs=EC0G_HZ,
        )
        grids = {}
        for part, left, right in (
            ("train", 0, boundary),
            ("test", boundary, len(clean)),
        ):
            grid = np.load(common / f"{part}_indices.npy") - left
            if grid.min() < WINDOW or grid.max() + MAX_DELAY >= right - left:
                raise ValueError(f"Invalid {part} target grid for {subject}.")
            segment = lfilter(
                low_b, low_a, clean[left:right].astype(np.float64),
                axis=0,
            )[::DOWNSAMPLE]
            _write_features(root / f"{part}_x.npy", segment, grid)
            grids[part] = digest_array(grid + left)
        manifest_path.write_text(json.dumps({
            "recipe": recipe(),
            "producer_sha256": sha256_file(Path(__file__)),
            "raw_sha256": raw_hash,
            "target_grid_sha256": grids,
        }, indent=2) + "\n")
        return False
