"""Ridge model composition for generic two-stream regression.

Input: flattened numeric feature matrices with matching row counts and
multi-output predictions from fitted ridge estimators. Output: predictions
from an early concatenated ridge or convex late fusion; no dataset I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

RIDGE_TOL = 1e-4
RIDGE_MAX_ITER = 1000
STREAMS = ("first", "second")
MODES = ("ridge_early", "ridge_late")


def make_ridge(alpha: float) -> Ridge:
    """Build the configured multi-output L2 estimator."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"Ridge alpha must be finite and positive: {alpha}.")
    return Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver="lsqr",
        tol=RIDGE_TOL,
        max_iter=RIDGE_MAX_ITER,
    )


@dataclass(frozen=True)
class RidgeFusion:
    """Compose already fitted stream estimators without joint training."""

    mode: str
    estimators: dict[str, Ridge]
    weights: np.ndarray | None = None

    def predict(self, designs: dict[str, np.ndarray]) -> np.ndarray:
        """Predict from one joint or two separately transformed matrices."""
        if self.mode == "ridge_early":
            if set(self.estimators) != {"joint"} or self.weights is not None:
                raise ValueError("Early fusion requires one joint estimator.")
            return self.estimators["joint"].predict(designs["joint"])
        if self.mode != "ridge_late":
            raise ValueError(f"Unknown fusion mode: {self.mode}.")
        if set(self.estimators) != set(STREAMS) or self.weights is None:
            raise ValueError(
                "Late fusion requires two estimators and weights."
            )
        first = self.estimators[STREAMS[0]].predict(designs[STREAMS[0]])
        second = self.estimators[STREAMS[1]].predict(designs[STREAMS[1]])
        if first.shape != second.shape or (
            first.shape[1] != len(self.weights)
        ):
            raise ValueError(
                f"Late predictions have shapes {first.shape}, "
                f"{second.shape}; weights have {self.weights.shape}."
            )
        return (
            self.weights * first + (1 - self.weights) * second
        )
