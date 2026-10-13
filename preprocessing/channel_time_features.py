"""Scale channel-time feature streams without dataset or model assumptions.

Input: matched training and prediction streams shaped (N,C,T). Output:
concatenated flattened float32 design matrices and per-stream channel
mean/std statistics, fitted only on training rows and time positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_STD = 1e-6


@dataclass(frozen=True)
class ChannelTimeScaler:
    """Fit per-channel moments across rows and time positions."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, training: np.ndarray) -> "ChannelTimeScaler":
        """Compute moments from a nonempty (N,C,T) training array."""
        if training.ndim != 3 or len(training) < 2:
            raise ValueError(
                f"Expected (N>=2,C,T) training array, got "
                f"{training.shape}."
            )
        if not np.isfinite(training).all():
            raise ValueError("Training features contain nonfinite values.")
        mean = training.mean(axis=(0, 2), keepdims=True)
        std = training.std(axis=(0, 2), keepdims=True)
        if (std <= 0).any():
            raise ValueError(
                "A channel has zero variance across training rows/time."
            )
        return cls(mean, np.maximum(std, MIN_STD))

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply saved moments and flatten to (N,C*T)."""
        if values.ndim != 3 or values.shape[1] != self.mean.shape[1]:
            raise ValueError(
                f"Expected (N,{self.mean.shape[1]},T), "
                f"got {values.shape}."
            )
        result = (
            (values - self.mean) / self.std
        ).astype(np.float32).reshape(len(values), -1)
        if not np.isfinite(result).all():
            raise ValueError("Scaled channel-time features are nonfinite.")
        return result


def transform_streams(
    training: tuple[np.ndarray, ...],
    prediction: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[ChannelTimeScaler, ...]]:
    """Fit separate stream scalers and concatenate their design matrices."""
    if not training or len(training) != len(prediction):
        raise ValueError(
            f"Expected matched nonempty stream groups, got "
            f"{len(training)} and {len(prediction)}."
        )
    if len({len(array) for array in training}) != 1 or (
        len({len(array) for array in prediction}) != 1
    ):
        raise ValueError("Stream row counts differ within a split.")
    scalers = tuple(ChannelTimeScaler.fit(array) for array in training)
    train_parts = [
        scaler.transform(array)
        for scaler, array in zip(scalers, training)
    ]
    predict_parts = [
        scaler.transform(array)
        for scaler, array in zip(scalers, prediction)
    ]
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(predict_parts, axis=1),
        scalers,
    )
