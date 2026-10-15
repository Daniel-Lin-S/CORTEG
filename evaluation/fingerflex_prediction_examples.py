"""Select test intervals and restore saved FingerFlex predictions.

Input: aligned prepared targets and saved standardized predictions. Output:
reproducible subject/row selections and original-value prediction arrays.
No fitting, artifact loading, plotting, or file writing occurs here.
"""
from __future__ import annotations

import numpy as np

from data.fingerflex_campaign import FINGER_HZ
from data.fingerflex.saved_predictions import SavedPrediction
from data.fingerflex.targets import TargetRows


def select_intervals(
    row_counts: dict[str, int],
    subject_count: int,
    intervals_per_subject: int,
    duration_seconds: int,
    random_seed: int,
) -> dict[str, list[tuple[int, int]]]:
    """Sample subjects and nonoverlapping 25 Hz test intervals.

    Parameters
    ----------
    row_counts : dict[str, int]
        Available test rows per subject.
    subject_count : int
        Number of subjects to draw without replacement.
    intervals_per_subject : int
        Number of disjoint intervals per selected subject.
    duration_seconds : int
        Duration of each interval in seconds.
    random_seed : int
        Reproducible NumPy generator seed.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Half-open test-row intervals for each sampled subject.
    """
    if not isinstance(subject_count, int) or not 0 < subject_count <= len(
        row_counts
    ):
        raise ValueError("Subject count exceeds available subjects.")
    if not isinstance(intervals_per_subject, int) or (
        intervals_per_subject < 1
    ):
        raise ValueError("Intervals per subject must be positive.")
    if not isinstance(duration_seconds, int) or duration_seconds < 1:
        raise ValueError("Interval duration must be positive seconds.")
    length = duration_seconds * FINGER_HZ
    if any(
        count < length * intervals_per_subject
        for count in row_counts.values()
    ):
        raise ValueError(
            "At least one subject has too few rows for the requested "
            "nonoverlapping intervals."
        )
    generator = np.random.default_rng(random_seed)
    subjects = generator.choice(
        sorted(row_counts), size=subject_count, replace=False,
    )
    selected = {}
    for subject in subjects:
        count = row_counts[str(subject)]
        starts = np.arange(count - length + 1)
        windows = []
        for _ in range(intervals_per_subject):
            if not len(starts):
                raise ValueError(
                    f"No disjoint test intervals remain for {subject}."
                )
            start = int(generator.choice(starts))
            stop = start + length
            windows.append((start, stop))
            starts = starts[
                (starts + length <= start) | (starts >= stop)
            ]
        selected[str(subject)] = sorted(windows)
    return selected


def restore_original_values(
    saved: SavedPrediction,
    rows: TargetRows,
    fit_rows: slice,
    path: str,
) -> np.ndarray:
    """Undo fit-row target scaling and verify saved standardized truth.

    Parameters
    ----------
    saved : SavedPrediction
        Standardized held-out arrays with shape ``(test rows, 5)``.
    rows : TargetRows
        Original glove targets with shape ``(train/test rows, 5)``.
    fit_rows : slice
        Rows selected for model fitting, excluding validation tail.
    path : str
        Fit path included in any diagnostic error.

    Returns
    -------
    np.ndarray
        Original-value predictions with shape ``(test rows, 5)``.
    """
    fit_y = rows.train_y[fit_rows]
    if not len(fit_y):
        raise ValueError(f"No fit target rows for {path}.")
    mean = fit_y.mean(axis=0, keepdims=True)
    std = fit_y.std(axis=0, keepdims=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (
        std <= 0
    ).any():
        raise ValueError(f"Invalid fit target normalization for {path}.")
    if saved.truth.shape != rows.test_y.shape:
        raise ValueError(
            f"Expected test shape {rows.test_y.shape}, "
            f"got {saved.truth.shape} for {path}."
        )
    expected = ((rows.test_y - mean) / std).astype(saved.truth.dtype)
    if not np.allclose(saved.truth, expected, rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"Saved standardized truth differs from prepared "
            f"test targets for {path}."
        )
    prediction = saved.prediction * std + mean
    if not np.isfinite(prediction).all():
        raise ValueError(f"Restored predictions are nonfinite for {path}.")
    return prediction


def matched_limits(
    truth: np.ndarray,
    prediction_groups: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """Return shared five-finger plot bounds across comparison groups.

    Parameters
    ----------
    truth : ndarray, shape (interval rows, 5)
        Original-value ground truth.
    prediction_groups : list[dict[str, ndarray]]
        One prediction dictionary per comparison root; arrays match truth.

    Returns
    -------
    ndarray, shape (5, 2)
        Lower and upper limits shared by corresponding figures.
    """
    traces = [truth]
    for group in prediction_groups:
        traces.extend(group.values())
    if not traces or any(
        trace.shape != truth.shape or not np.isfinite(trace).all()
        for trace in traces
    ):
        raise ValueError("Prediction traces differ in shape or finiteness.")
    values = np.stack(traces)
    minimum = values.min(axis=(0, 1))
    maximum = values.max(axis=(0, 1))
    span = maximum - minimum
    if (span <= 0).any():
        raise ValueError("A finger has no value range for plotting.")
    padding = span * 0.06
    return np.column_stack((minimum - padding, maximum + padding))
