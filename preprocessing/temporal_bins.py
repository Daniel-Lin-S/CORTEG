"""Average fixed windows into equal-duration temporal bins.

Input: a finite or memory-mapped array shaped (rows, channels, samples)
from a one-second window and an output bin count. Output: float32 array
shaped (rows, channels, bins). Both streams use the same bin count,
so each output bin spans the same fraction of the one-second window.
"""
from __future__ import annotations

import numpy as np


def mean_time_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Average sample positions assigned to equal-duration intervals.

    Parameters
    ----------
    values : ndarray, shape (rows, channels, samples)
        Input windows sampled uniformly in time.
    bins : int
        Number of nonempty, equal-duration output intervals.

    Returns
    -------
    ndarray, shape (rows, channels, bins)
        Mean value in each interval, in float32.
    """
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(
            f"Expected nonempty (rows, channels, samples), got "
            f"{values.shape}."
        )
    samples = values.shape[2]
    if type(bins) is not int or not 1 <= bins <= samples:
        raise ValueError(
            f"Expected bin count in [1, {samples}], got {bins}."
        )
    starts = np.ceil(
        np.arange(bins, dtype=np.float64) * samples / bins
    ).astype(np.int64)
    widths = np.diff(np.append(starts, samples))
    if (widths <= 0).any():
        raise ValueError("Temporal binning produced an empty interval.")
    result = np.add.reduceat(
        values, starts, axis=2, dtype=np.float64,
    ) / widths.reshape(1, 1, -1)
    output = result.astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Temporal bins contain nonfinite values.")
    return output
