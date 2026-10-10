"""Prepare shared Stanford FingerFlex windows from registered raw MATLAB files.

Input is ``<raw>/<subject>/<subject>_fingerflex.mat`` with ``data``
(time, channel), ``flex`` (time, finger), and ``locs`` (channel, xyz mm).
Output ``prepared/<subject>/common/`` holds cleaned channel decisions,
coordinates, and train/test targets and timestamps. Model features live in
``prepared/<subject>/<model>/<delay>/`` as train/test NumPy arrays.
The raw MATLAB files are read only. Delays shift neural windows, never targets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

NUMBA_CACHE = "/tmp/numba_fingerflex"
os.environ.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE)

import mne
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, hilbert, lfilter

from data.stanford_split import split_boundary

EC0G_HZ = 1000
FINGER_HZ = 25
WINDOW = EC0G_HZ
STRIDE = EC0G_HZ // FINGER_HZ
FEATURE_HZ = 200
MAX_DELAY = 40
FINGERS = 5


def digest_array(values: np.ndarray) -> str:
    """Hash array values with dtype and shape for provenance.

    Parameters
    ----------
    values : np.ndarray
        Array of any shape.

    Returns
    -------
    str
        SHA256 digest.
    """
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def target_grid(length: int, delay: int) -> np.ndarray:
    """Return 25 Hz target indices in one chronological segment.

    Parameters
    ----------
    length : int
        Segment length in 1 kHz samples.
    delay : int
        Neural-window shift in samples, either 0 or 40.

    Returns
    -------
    np.ndarray
        Target indices, shape (N,).
    """
    if delay not in (0, MAX_DELAY):
        raise ValueError(f"Delay must be 0 or {MAX_DELAY}, got {delay}.")
    grid = np.arange(WINDOW, length - MAX_DELAY, STRIDE)
    if len(grid) == 0:
        raise ValueError(f"Segment has no valid target rows: {length}.")
    return grid.astype(np.int64)


def clean_recording(
    ecog: np.ndarray,
    boundary: int,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one-pass notch filtering, training-channel rule, and CAR.

    Parameters
    ----------
    ecog : np.ndarray
        Raw ECoG, shape (samples, channels).
    boundary : int
        Exclusive chronological training endpoint.
    config : dict
        Resolved Stanford preprocessing settings.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Cleaned (samples, retained channels), kept channel indices, and
        pre-reference training variance for every raw channel.
    """
    if ecog.ndim != 2 or ecog.shape[1] < 2:
        raise ValueError(f"Expected (samples, channels>=2), got {ecog.shape}.")
    if not 1 < boundary < len(ecog):
        raise ValueError(f"Invalid train boundary {boundary} of {len(ecog)}.")
    data = ecog.astype(np.float64) * config["ecog_scale"]
    if not np.isfinite(data).all():
        raise ValueError("Raw ECoG has nonfinite values.")
    for low, high in config["notch_bands_hz"]:
        b, a = butter(
            config["notch_filter_order"],
            [low, high],
            btype="bandstop",
            fs=EC0G_HZ,
        )
        channel_mean = data.mean(axis=0, keepdims=True)
        demeaned = data - channel_mean
        offset = demeaned[:1]
        dc_gain = b.sum() / a.sum()
        filtered = lfilter(b, a, demeaned - offset, axis=0)
        data = filtered + dc_gain * offset + channel_mean
    variance = data[:boundary].var(
        axis=0,
        ddof=1 - config["bad_channel"]["variance_normalization"],
    )
    median = float(np.median(variance))
    if not np.isfinite(variance).all() or median <= 0:
        raise ValueError("Training channel variance is nonfinite or zero.")
    rule = config["bad_channel"]
    kept = np.flatnonzero(
        (variance >= median * rule["low_variance_multiplier"])
        & (variance <= median * rule["high_variance_multiplier"])
    )
    if len(kept) < 2:
        raise ValueError(f"Bad-channel rule retained {len(kept)} channels.")
    data = data[:, kept]
    data -= data.mean(axis=1, keepdims=True)
    return data.astype(np.float32), kept, variance


def _corteg_features(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract CORTEG broadband and HGA arrays from (1000, C) window.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Broadband (C, 128) and HGA/LFS (C, 200, 2).
    """
    channels = window.T.astype(np.float64)
    filtered = mne.filter.filter_data(
        channels,
        sfreq=EC0G_HZ,
        l_freq=70,
        h_freq=200,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        verbose=False,
    )
    envelope = np.abs(hilbert(filtered, axis=1))
    hga = envelope.reshape(len(channels), FEATURE_HZ, 5).mean(axis=2)
    lfs = mne.filter.resample(channels, down=5, npad="auto")
    broadband = mne.filter.resample(
        channels,
        up=16,
        down=125,
        npad="auto",
    )
    return broadband.astype(np.float32), np.stack(
        (hga, lfs), axis=-1,
    ).astype(np.float32)


def _hilo_continuous(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply source HiLoFuseNet causal HGA/LFS filters to (T, C)."""
    b_hga, a_hga = butter(4, [70, 200], btype="bandpass", fs=EC0G_HZ)
    b_lfs, a_lfs = butter(4, 100, btype="low", fs=EC0G_HZ)
    signal = data.astype(np.float64)
    return (
        lfilter(b_hga, a_hga, signal, axis=0),
        lfilter(b_lfs, a_lfs, signal, axis=0),
    )


def _hilo_feature(
    hga: np.ndarray,
    lfs: np.ndarray,
    end: int,
) -> np.ndarray:
    """Use source HiLo windowed Hilbert/binning for one (C,200,2) row."""
    channels = hga[end - WINDOW:end].T
    envelope = np.abs(hilbert(channels, axis=1))
    high = envelope.reshape(len(channels), FEATURE_HZ, 5).mean(axis=2)
    low = lfs[end - WINDOW:end:5].T
    return np.stack((high, low), axis=-1).astype(np.float32)


def _write_array(path: Path, values: np.ndarray) -> None:
    """Atomically write a nonempty NumPy array."""
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Empty or nonfinite prepared array: {path}.")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    temporary.replace(path)


def prepare_subject(
    subject: str,
    raw_root: Path,
    prepared_root: Path,
    config: dict,
    delays: tuple[int, ...],
) -> None:
    """Prepare one subject and all requested model/delay feature caches.

    Parameters
    ----------
    subject : str
        Stanford patient code.
    raw_root : Path
        Read-only directory of raw patient folders.
    prepared_root : Path
        Campaign ``prepared`` directory.
    config : dict
        Resolved semantic campaign configuration.
    delays : tuple[int, ...]
        Neural shifts in samples.
    """
    subject_root = prepared_root / subject
    common = subject_root / "common"
    expected = [
        subject_root / model / str(delay) / f"{part}.npy"
        for model in ("corteg", "hilo")
        for delay in delays if model == "corteg" or delay == 0
        for part in (("train_x", "test_x", "train_hi", "test_hi")
                     if model == "corteg" else ("train_x", "test_x"))
    ]
    if (common / "metadata.json").exists() and all(
        path.exists() for path in expected
    ):
        return
    raw_path = raw_root / subject / f"{subject}_fingerflex.mat"
    recording = loadmat(raw_path)
    ecog = recording["data"]
    flex = recording["flex"]
    locs = recording["locs"]
    if ecog.shape[0] != flex.shape[0] or flex.shape[1] != FINGERS:
        raise ValueError(f"Invalid ECoG/flex shapes in {raw_path}.")
    if locs.shape != (ecog.shape[1], 3):
        raise ValueError(f"Invalid electrode coordinates in {raw_path}.")
    boundary, rule = split_boundary(
        len(ecog), EC0G_HZ, config["preprocessing"]["split"],
    )
    clean, kept, variance = clean_recording(
        ecog, boundary, config["preprocessing"],
    )
    common.mkdir(parents=True, exist_ok=True)
    _write_array(common / "xyz_mm.npy", locs[kept].astype(np.float32))
    for part, start, stop in (
        ("train", 0, boundary),
        ("test", boundary, len(clean)),
    ):
        grid = target_grid(stop - start, 0)
        targets = flex[start + grid].astype(np.float32)
        _write_array(common / f"{part}_y.npy", targets)
        _write_array(common / f"{part}_indices.npy", start + grid)
    metadata = {
        "raw_path": str(raw_path),
        "raw_samples": len(ecog),
        "split_boundary": boundary,
        "split_rule": rule,
        "kept_channels_zero_based": kept.tolist(),
        "bad_channels_zero_based": np.setdiff1d(
            np.arange(ecog.shape[1]), kept,
        ).tolist(),
        "channel_variance": variance.tolist(),
        "median_channel_variance": float(np.median(variance)),
        "target_grid_sha256": {
            part: digest_array(np.load(common / f"{part}_indices.npy"))
            for part in ("train", "test")
        },
        "target_values_sha256": {
            part: digest_array(np.load(common / f"{part}_y.npy"))
            for part in ("train", "test")
        },
    }
    (common / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for part, start, stop in (
        ("train", 0, boundary),
        ("test", boundary, len(clean)),
    ):
        segment = clean[start:stop]
        grid = target_grid(len(segment), 0)
        hga, lfs = _hilo_continuous(segment)
        for model in ("corteg", "hilo"):
            for delay in delays:
                if model == "hilo" and delay != 0:
                    continue
                out = subject_root / model / str(delay)
                out.mkdir(parents=True, exist_ok=True)
                if (out / f"{part}_x.npy").exists():
                    continue
                if model == "corteg":
                    raw = []
                    features = []
                    for end in grid:
                        left = int(end + delay)
                        broadband, high = _corteg_features(
                            segment[left - WINDOW:left]
                        )
                        raw.append(broadband)
                        features.append(high)
                    _write_array(out / f"{part}_x.npy", np.stack(raw))
                    _write_array(out / f"{part}_hi.npy", np.stack(features))
                else:
                    features = [
                        _hilo_feature(hga, lfs, int(end)) for end in grid
                    ]
                    _write_array(out / f"{part}_x.npy", np.stack(features))
