"""Load the prepared Stanford FingerFlex arrays for linear fits.

Input: prepared/<subject>/common/{train,test}_{y,indices}.npy and
prepared/<subject>/corteg/0/{train,test}_{x,hi}.npy. Arrays contain
five finger targets, (N,C,128) broadband, and (N,C,200,2) spectral
features; channel zero of the last axis is CORTEG's high gamma input.
Output: validated memory-mapped arrays and target-boundary identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.fingerflex_campaign import digest_array

FINGERS = 5
BROADBAND_SAMPLES = 128
HIGH_GAMMA_SAMPLES = 200


@dataclass(frozen=True)
class FingerflexLinearArrays:
    """Prepared input and target arrays for one Stanford subject."""

    common: Path
    train_y: np.ndarray
    test_y: np.ndarray
    train_broadband: np.ndarray
    test_broadband: np.ndarray
    train_high_gamma: np.ndarray
    test_high_gamma: np.ndarray

    def boundaries(self, fit: slice, validation: slice) -> dict:
        """Describe saved fit, reference-validation, and test rows."""
        indices = np.load(self.common / "train_indices.npy")
        test_indices = np.load(self.common / "test_indices.npy")
        return {
            "fit_indices_sha256": digest_array(indices[fit]),
            "validation_indices_sha256": digest_array(
                indices[validation]
            ),
            "test_grid_sha256": digest_array(test_indices),
            "target_values_sha256": digest_array(self.test_y),
            "fit_window_count": len(indices[fit]),
            "validation_window_count": len(indices[validation]),
            "test_window_count": len(test_indices),
            "fit_first_target": int(indices[fit][0]),
            "fit_last_target": int(indices[fit][-1]),
            "validation_first_target": int(indices[validation][0]),
            "test_first_target": int(test_indices[0]),
        }


def load_fingerflex_linear(
    prepared: Path,
    subject: str,
) -> FingerflexLinearArrays:
    """Read one subject's existing CORTEG inputs without copying files."""
    common = prepared / subject / "common"
    features = prepared / subject / "corteg" / "0"
    train_y = np.load(common / "train_y.npy")
    test_y = np.load(common / "test_y.npy")
    train_broadband = np.load(
        features / "train_x.npy", mmap_mode="r",
    )
    test_broadband = np.load(
        features / "test_x.npy", mmap_mode="r",
    )
    train_high = np.load(
        features / "train_hi.npy", mmap_mode="r",
    )[..., 0]
    test_high = np.load(
        features / "test_hi.npy", mmap_mode="r",
    )[..., 0]
    for name, target, broadband, high in (
        ("train", train_y, train_broadband, train_high),
        ("test", test_y, test_broadband, test_high),
    ):
        if target.ndim != 2 or target.shape[1] != FINGERS:
            raise ValueError(
                f"{subject} {name} target needs (N,{FINGERS}), "
                f"got {target.shape}."
            )
        if broadband.shape[:2] != high.shape[:2] or (
            broadband.shape[0] != len(target)
        ):
            raise ValueError(
                f"{subject} {name} feature/target row or channel "
                f"mismatch: {broadband.shape}, {high.shape}, "
                f"{target.shape}."
            )
        if broadband.shape[2] != BROADBAND_SAMPLES or (
            high.shape[2] != HIGH_GAMMA_SAMPLES
        ):
            raise ValueError(
                f"{subject} {name} spectral lengths differ from "
                f"{BROADBAND_SAMPLES}/{HIGH_GAMMA_SAMPLES}: "
                f"{broadband.shape}, {high.shape}."
            )
        if len(target) < 2 or not np.isfinite(target).all():
            raise ValueError(
                f"{subject} {name} targets are empty or nonfinite."
            )
    return FingerflexLinearArrays(
        common, train_y, test_y, train_broadband,
        test_broadband, train_high, test_high,
    )
