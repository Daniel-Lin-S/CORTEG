"""Load shared Stanford FingerFlex targets and target-row identities.

Input: ``prepared/<subject>/common/{train,test}_{y,indices}.npy``.
Output: raw five-finger targets and 1 kHz target sample indices in memory,
plus hashes describing selected fit, validation, and test rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.fingerflex_campaign import digest_array

FINGER_COUNT = 5


@dataclass(frozen=True)
class TargetRows:
    """Prepared raw targets and their 1 kHz sample indices."""

    train_y: np.ndarray
    train_indices: np.ndarray
    test_y: np.ndarray
    test_indices: np.ndarray


def load_target_rows(common: Path) -> TargetRows:
    """Load nonempty, aligned raw targets from a prepared subject.

    Parameters
    ----------
    common : Path
        Subject's ``prepared/<subject>/common`` directory.

    Returns
    -------
    TargetRows
        Target matrices with shape ``(train/test rows, 5)`` and
        corresponding sample-index vectors with shape ``(rows,)``.
    """
    arrays = TargetRows(*(
        np.load(common / filename)
        for filename in (
            "train_y.npy", "train_indices.npy",
            "test_y.npy", "test_indices.npy",
        )
    ))
    for part in ("train", "test"):
        target = getattr(arrays, f"{part}_y")
        indices = getattr(arrays, f"{part}_indices")
        if target.ndim != 2 or target.shape[1] != FINGER_COUNT:
            raise ValueError(
                f"Expected {part} target shape (rows, {FINGER_COUNT}) "
                f"in {common}, got {target.shape}."
            )
        if indices.ndim != 1 or len(indices) != len(target):
            raise ValueError(
                f"Expected one {part} index per target in {common}."
            )
        if len(target) < 2 or not np.isfinite(target).all():
            raise ValueError(
                f"Too few or nonfinite {part} targets in {common}."
            )
        if not np.issubdtype(indices.dtype, np.integer) or not np.all(
            np.diff(indices) > 0
        ):
            raise ValueError(
                f"Expected increasing integer {part} target indices "
                f"in {common}."
            )
    return arrays


def target_boundaries(
    rows: TargetRows,
    fit: slice,
    validation: slice,
) -> dict:
    """Describe selected fit, validation, and test row identities.

    Parameters
    ----------
    rows : TargetRows
        Shared prepared target arrays.
    fit, validation : slice
        Chronological fit and validation row selections.

    Returns
    -------
    dict
        Counts, first/last samples, and SHA256 hashes for all row groups.
    """
    selected = rows.train_indices[fit]
    heldout = rows.train_indices[validation]
    if not len(selected) or not len(heldout):
        raise ValueError("Fit and validation target rows must be nonempty.")
    return {
        "fit_indices_sha256": digest_array(selected),
        "validation_indices_sha256": digest_array(heldout),
        "test_grid_sha256": digest_array(rows.test_indices),
        "target_values_sha256": digest_array(rows.test_y),
        "fit_window_count": len(selected),
        "validation_window_count": len(heldout),
        "test_window_count": len(rows.test_indices),
        "fit_first_target": int(selected[0]),
        "fit_last_target": int(selected[-1]),
        "validation_first_target": int(heldout[0]),
        "test_first_target": int(rows.test_indices[0]),
    }
