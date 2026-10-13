"""Define blocked CV, alpha selection, and convex output weighting.

Input: fit-row count, fold score matrix, and aligned out-of-fold
predictions/targets. Output: purged fold indices, selected alpha index,
and one convex weight per output. No model or preprocessing is invoked.
"""
from __future__ import annotations

import numpy as np

FOLDS = 5
PURGE_ROWS = 25
ALPHA_PER_ROW = tuple(10.0 ** exponent for exponent in range(-4, 3))


def _folds(
    count: int, folds: int = FOLDS, purge_rows: int = PURGE_ROWS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition rows into five contiguous held-out blocks with purging."""
    if folds < 2 or purge_rows < 0 or (
        count < folds * (2 * purge_rows + 2)
    ):
        raise ValueError(
            f"Need at least {folds * (2 * purge_rows + 2)} fit rows, "
            f"got {count}."
        )
    partitions = np.array_split(np.arange(count), folds)
    result = []
    for held_out in partitions:
        first, last = int(held_out[0]), int(held_out[-1]) + 1
        keep = np.ones(count, dtype=bool)
        keep[max(0, first - purge_rows):min(count, last + purge_rows)] = False
        training = np.flatnonzero(keep)
        if len(training) < 2:
            raise ValueError(
                f"Fold {first}:{last} leaves {len(training)} train rows."
            )
        result.append((training, held_out))
    return result


def choose_alpha(fold_scores: np.ndarray) -> int:
    """Choose the largest alpha on a tied mean-fold score."""
    expected = (len(ALPHA_PER_ROW), FOLDS)
    if fold_scores.shape != expected or (
        not np.isfinite(fold_scores).all()
    ):
        raise ValueError(
            f"Expected finite CV scores with shape {expected}, "
            f"got {fold_scores.shape}."
        )
    mean_scores = fold_scores.mean(axis=1)
    return max(
        range(len(ALPHA_PER_ROW)),
        key=lambda index: (mean_scores[index], index),
    )


def choose_bin_alpha(
    scores: np.ndarray, bins: tuple[int, ...],
    alphas: tuple[float, ...], folds: int,
) -> tuple[int, int]:
    """Select the best bin/alpha pair by mean held-out correlation.

    Parameters
    ----------
    scores : ndarray, shape (bin candidates, alpha candidates, folds)
        Finite mean-finger Pearson correlations.
    bins : tuple[int, ...]
        Output bin counts in candidate order.
    alphas : tuple[float, ...]
        Positive alpha-per-training-row values in candidate order.
    folds : int
        Number of contiguous CV folds.

    Returns
    -------
    tuple[int, int]
        Bin and alpha candidate indices. Exact ties favor stronger
        regularization, then fewer temporal bins.
    """
    expected = (len(bins), len(alphas), folds)
    if scores.shape != expected or not np.isfinite(scores).all():
        raise ValueError(
            f"Expected finite CV scores shaped {expected}, "
            f"got {scores.shape}."
        )
    if not bins or len(set(bins)) != len(bins) or any(
        type(value) is not int or value < 1 for value in bins
    ):
        raise ValueError(f"Invalid temporal bin candidates: {bins}.")
    if not alphas or any(
        not np.isfinite(value) or value <= 0 for value in alphas
    ):
        raise ValueError(f"Invalid L2 candidates: {alphas}.")
    mean = scores.mean(axis=2)
    return max(
        np.ndindex(mean.shape),
        key=lambda pair: (
            mean[pair], alphas[pair[1]], -bins[pair[0]],
        ),
    )


def convex_weights(
    first: np.ndarray,
    second: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Fit independent convex output weights to out-of-fold predictions."""
    if first.shape != target.shape or second.shape != target.shape:
        raise ValueError(
            f"Expected prediction shapes {target.shape}, got "
            f"{first.shape} and {second.shape}."
        )
    delta = first - second
    denominator = np.sum(delta.astype(np.float64) ** 2, axis=0)
    if (denominator <= 0).any():
        raise ValueError(
            "Late-fusion streams give identical out-of-fold predictions "
            "for at least one output."
        )
    numerator = np.sum(delta * (target - second), axis=0)
    weights = np.clip(numerator / denominator, 0.0, 1.0)
    if not np.isfinite(weights).all():
        raise ValueError("Late-fusion weights are nonfinite.")
    return weights
