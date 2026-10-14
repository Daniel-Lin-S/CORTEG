"""Define high-gamma regression models without dataset or CV policy.

Input: scaled (batch, channels, timepoints) HGA tensors or flattened
feature matrices. Output: five continuous finger predictions. The MLP
topology follows HiLoFuseNet's comparison MLP, with configurable width:
https://github.com/hisunjiang/HiLoFuseNet/blob/main/
finger_regression/models/nn_regressors.py
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR
from torch import nn

FINGERS = 5


class FingerMLP(nn.Module):
    """Two-hidden-layer flattened-window finger regressor.

    Parameters
    ----------
    channels : int
        HGA channels.
    timepoints : int
        Timepoints per HGA window.
    hidden_size : int
        First hidden-layer width; the second has half this width.
    dropout : float
        Dropout probability after each hidden activation.
    """

    def __init__(
        self, channels: int, timepoints: int,
        hidden_size: int, dropout: float,
    ) -> None:
        super().__init__()
        if min(channels, timepoints, hidden_size) < 1 or (
            hidden_size % 2 or not 0 <= dropout < 1
        ):
            raise ValueError(
                "MLP channels, timepoints, and even hidden width must "
                "be positive; dropout must lie in [0, 1)."
            )
        self.channels = channels
        self.timepoints = timepoints
        self.network = nn.Sequential(
            nn.Linear(channels * timepoints, hidden_size),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, FINGERS),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predict five outputs from (batch, channels, timepoints)."""
        if values.ndim != 3 or values.shape[1:] != (
            self.channels, self.timepoints
        ):
            raise ValueError(
                f"Expected (batch, {self.channels}, {self.timepoints}), "
                f"got {tuple(values.shape)}."
            )
        return self.network(values.reshape(len(values), -1))


class SharedKernelSVR:
    """Fit five independent SVRs from one exact RBF kernel matrix.

    Parameters
    ----------
    c_value, gamma : float
        SVR penalty and coefficient of squared feature distance.
    epsilon : float
        Epsilon in standardized target units.
    cache_mib : float
        Per-estimator libsvm cache size in MiB.
    """

    def __init__(
        self, c_value: float, gamma: float,
        epsilon: float, cache_mib: float,
    ) -> None:
        self.c_value = c_value
        self.gamma = gamma
        self.epsilon = epsilon
        self.cache_mib = cache_mib

    def fit(
        self, design: np.ndarray, targets: np.ndarray,
    ) -> "SharedKernelSVR":
        """Fit five outputs from (N,D) features and (N,5) targets."""
        if design.ndim != 2 or len(design) < 2 or (
            not np.isfinite(design).all()
        ):
            raise ValueError(
                f"Expected finite (N>=2, features) design, got "
                f"{design.shape}."
            )
        if targets.shape != (len(design), FINGERS) or (
            not np.isfinite(targets).all()
        ):
            raise ValueError(
                "Expected finite standardized targets with shape "
                f"({len(design)},{FINGERS}), got {targets.shape}."
            )
        self.training_design_ = np.asarray(
            design, dtype=np.float64, order="C",
        )
        kernel = np.ascontiguousarray(
            rbf_kernel(self.training_design_, gamma=self.gamma),
        )
        self.estimators_ = [
            SVR(
                kernel="precomputed", C=self.c_value,
                epsilon=self.epsilon, cache_size=self.cache_mib,
            ).fit(kernel, targets[:, finger])
            for finger in range(FINGERS)
        ]
        return self

    def predict(self, design: np.ndarray) -> np.ndarray:
        """Predict (M,5) targets from scaled (M,D) features."""
        if not hasattr(self, "estimators_"):
            raise ValueError("SharedKernelSVR must be fitted before predict.")
        if design.ndim != 2 or len(design) < 1 or (
            design.shape[1] != self.training_design_.shape[1]
        ) or not np.isfinite(design).all():
            raise ValueError(
                "Expected finite prediction design with shape "
                f"(M,{self.training_design_.shape[1]}), "
                f"got {design.shape}."
            )
        kernel = np.ascontiguousarray(rbf_kernel(
            np.asarray(design, dtype=np.float64, order="C"),
            self.training_design_, gamma=self.gamma,
        ))
        prediction = np.column_stack([
            estimator.predict(kernel)
            for estimator in self.estimators_
        ])
        if not np.isfinite(prediction).all():
            raise ValueError("Shared-kernel SVR predictions are nonfinite.")
        return prediction


def make_svr(
    design: np.ndarray, c_value: float,
    gamma_factor: float, epsilon: float,
    cache_mib: float,
) -> SharedKernelSVR:
    """Create independent exact RBF SVRs sharing one parameter tuple.

    Parameters
    ----------
    design : ndarray, shape (training rows, features)
        Fold-scaled input used to compute scikit-learn's gamma scale.
    c_value, gamma_factor, epsilon : float
        Positive SVR C, multiplier of gamma scale, and epsilon.
    cache_mib : float
        Machine-local per-finger kernel cache limit.

    Returns
    -------
    SharedKernelSVR
        Unfitted five-output exact RBF SVR.
    """
    if design.ndim != 2 or len(design) < 2 or (
        not np.isfinite(design).all()
    ):
        raise ValueError(
            f"Expected finite (N>=2, features) design, got "
            f"{design.shape}."
        )
    parameters = np.array(
        [c_value, gamma_factor, epsilon, cache_mib], dtype=float,
    )
    if not np.isfinite(parameters).all() or (
        min(c_value, gamma_factor, cache_mib) <= 0 or epsilon < 0
    ):
        raise ValueError(
            "SVR C, gamma factor, epsilon, and cache must be finite; "
            "C, gamma factor, and cache must be positive."
        )
    variance = float(design.var())
    if variance <= 0:
        raise ValueError("SVR design variance is zero.")
    gamma = gamma_factor / (design.shape[1] * variance)
    return SharedKernelSVR(
        c_value, gamma, epsilon, cache_mib,
    )
