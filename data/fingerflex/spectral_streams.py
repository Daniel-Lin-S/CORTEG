"""Load prepared broadband and high-gamma FingerFlex feature streams.

Input: ``prepared/<subject>/common/`` raw targets and
``prepared/<subject>/corteg/0/`` broadband and HGA arrays. Output is
memory-mapped feature arrays with shared validated target rows. The
``corteg/0`` directory names the feature source, not a fitted estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.fingerflex.targets import TargetRows, load_target_rows

BROADBAND_SAMPLES = 128
HIGH_GAMMA_SAMPLES = 200


@dataclass(frozen=True)
class PreparedSpectralStreams:
    """Shared targets and two model-independent spectral streams."""

    targets: TargetRows
    train_broadband: np.ndarray
    test_broadband: np.ndarray
    train_high_gamma: np.ndarray
    test_high_gamma: np.ndarray


def load_spectral_streams(
    prepared: Path,
    subject: str,
) -> PreparedSpectralStreams:
    """Read the two existing prepared streams without copying data.

    Parameters
    ----------
    prepared : Path
        Shared prepared-subject root.
    subject : str
        Stanford subject code.

    Returns
    -------
    PreparedSpectralStreams
        Broadband has shape ``(rows, channels, 128)``; HGA has shape
        ``(rows, channels, 200)``. Targets have shape ``(rows, 5)``.
    """
    subject_root = prepared / subject
    targets = load_target_rows(subject_root / "common")
    features = subject_root / "corteg" / "0"
    broadband = tuple(
        np.load(features / f"{part}_x.npy", mmap_mode="r")
        for part in ("train", "test")
    )
    high_gamma = tuple(
        np.load(features / f"{part}_hi.npy", mmap_mode="r")[..., 0]
        for part in ("train", "test")
    )
    for part, target, low, high in zip(
        ("train", "test"),
        (targets.train_y, targets.test_y),
        broadband, high_gamma,
    ):
        if low.ndim != 3 or high.ndim != 3 or (
            low.shape[:2] != high.shape[:2]
        ) or low.shape[0] != len(target):
            raise ValueError(
                f"{subject} {part} feature and target shapes differ: "
                f"{low.shape}, {high.shape}, {target.shape}."
            )
        if low.shape[2] != BROADBAND_SAMPLES or (
            high.shape[2] != HIGH_GAMMA_SAMPLES
        ):
            raise ValueError(
                f"{subject} {part} spectral lengths differ from "
                f"{BROADBAND_SAMPLES}/{HIGH_GAMMA_SAMPLES}: "
                f"{low.shape}, {high.shape}."
            )
    return PreparedSpectralStreams(
        targets, broadband[0], broadband[1],
        high_gamma[0], high_gamma[1],
    )
