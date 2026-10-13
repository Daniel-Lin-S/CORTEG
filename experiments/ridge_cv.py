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


def _folds(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition rows into five contiguous held-out blocks with purging."""
    if count < FOLDS * (2 * PURGE_ROWS + 2):
        raise ValueError(
            f"Need at least {FOLDS * (2 * PURGE_ROWS + 2)} fit rows, "
            f"got {count}."
        )
    partitions = np.array_split(np.arange(count), FOLDS)
    result = []
    for held_out in partitions:
        first, last = int(held_out[0]), int(held_out[-1]) + 1
        keep = np.ones(count, dtype=bool)
        keep[max(0, first - PURGE_ROWS):min(count, last + PURGE_ROWS)] = False
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
