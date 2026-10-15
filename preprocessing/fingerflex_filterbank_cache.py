"""Cache naplib-style HGA envelopes for Stanford FingerFlex windows.

Input: raw MATLAB recordings and the completed shared preprocessing split.
Output: one subject directory with train/test band arrays (N,C,11,200),
mean arrays (N,C,200), center frequencies, and a final index.json marker.
Each cleaned one-second window is processed independently.
"""
from __future__ import annotations

import fcntl
import json
import uuid
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from data.fingerflex_campaign import (
    EC0G_HZ, WINDOW, clean_recording, digest_array, target_grid,
)
from preprocessing.naplib_filterbank import (
    analytic_kernels, center_frequencies, window_amplitudes,
)


def _build_subject(
    subject: str, base: dict, prepared: Path, root: Path,
    settings: dict,
) -> Path:
    """Create and validate one subject cache after split alignment.

    Parameters
    ----------
    subject : str
        Stanford subject code.
    base : dict
        Resolved source campaign including raw root and preprocessing.
    prepared : Path
        Shared prepared subject directory.
    root : Path
        Semantic filterbank cache root.
    settings : dict
        Filter limits and output timepoint count.

    Returns
    -------
    Path
        Complete subject cache directory.
    """
    output = root / "subjects" / subject
    marker = output / "index.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved["settings"] != settings:
            raise ValueError(f"Filterbank settings differ: {marker}.")
        for part in ("train", "test"):
            for suffix in ("bands", "average"):
                if not (output / f"{part}_{suffix}.npy").is_file():
                    raise FileNotFoundError(
                        f"Filterbank cache is incomplete: {output}."
                    )
        return output
    if output.exists():
        quarantine = root / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        output.rename(quarantine / f"{subject}_{uuid.uuid4().hex}")
    output.mkdir(parents=True)
    metadata = json.loads((prepared / "common/metadata.json").read_text())
    raw = Path(base["raw_root"]) / subject / f"{subject}_fingerflex.mat"
    recording = loadmat(raw)
    data = recording["data"]
    boundary = metadata["split_boundary"]
    clean, kept, _ = clean_recording(
        data, boundary, base["preprocessing"],
    )
    if kept.tolist() != metadata["kept_channels_zero_based"]:
        raise ValueError(f"Cleaned channels differ from prepared: {subject}.")
    centers = center_frequencies(
        settings["low_hz"], settings["high_hz"],
    )
    if len(centers) != settings["components"]:
        raise ValueError(
            f"Expected {settings['components']} components, "
            f"got {len(centers)}."
        )
    kernels = analytic_kernels(WINDOW, EC0G_HZ, centers)
    np.save(output / "center_hz.npy", centers)
    for part, start, stop in (
        ("train", 0, boundary), ("test", boundary, len(clean)),
    ):
        segment = clean[start:stop]
        grid = target_grid(len(segment), 0)
        indices = np.load(prepared / "common" / f"{part}_indices.npy")
        if digest_array(start + grid) != digest_array(indices):
            raise ValueError(
                f"Filterbank rows differ from prepared {subject}/{part}."
            )
        shape = (
            len(grid), len(kept), len(centers),
            settings["timepoints"],
        )
        bands = np.lib.format.open_memmap(
            output / f"{part}_bands.npy", mode="w+",
            dtype=np.float32, shape=shape,
        )
        average = np.lib.format.open_memmap(
            output / f"{part}_average.npy", mode="w+",
            dtype=np.float32,
            shape=(len(grid), len(kept), settings["timepoints"]),
        )
        for index, end in enumerate(grid):
            window = segment[int(end) - WINDOW:int(end)]
            value = window_amplitudes(
                window, kernels, settings["timepoints"],
            )
            bands[index] = value
            average[index] = value.mean(axis=1)
        bands.flush()
        average.flush()
        del bands, average
    marker.write_text(json.dumps({
        "subject": subject,
        "settings": settings,
        "center_hz": centers.tolist(),
        "prepared_target_sha256": metadata["target_values_sha256"],
        "prepared_grid_sha256": metadata["target_grid_sha256"],
    }, indent=2) + "\n")
    return output


def load_filterbank(cache: Path, part: str, resolved: bool) -> np.ndarray:
    """Open the averaged or channel-component HGA design by row."""
    suffix = "bands" if resolved else "average"
    values = np.load(
        cache / f"{part}_{suffix}.npy", mmap_mode="r",
    )
    if resolved:
        rows, channels, bands, timepoints = values.shape
        values = values.reshape(rows, channels * bands, timepoints)
    if values.ndim != 3 or len(values) < 1:
        raise ValueError(f"Invalid {part} filterbank shape: {values.shape}.")
    return values


def cache_subject(
    subject: str, base: dict, prepared: Path, root: Path,
    settings: dict,
) -> Path:
    """Serialize each subject's cache creation across coordinators."""
    locks = root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{subject}.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            return _build_subject(
                subject, base, prepared, root, settings,
            )
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
