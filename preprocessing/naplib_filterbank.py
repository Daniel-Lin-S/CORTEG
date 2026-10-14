"""Compute naplib-style Gaussian filterbank HGA envelopes.

Input: cleaned ECoG windows shaped (1000, channels), sampling frequency,
and high-gamma bounds. Output: center frequencies and envelopes shaped
(channels, frequency components, 200 timepoints). No dataset I/O occurs.

The Gaussian spacing and analytic FFT response follow naplib-python's
``filterbank_hilbert`` implementation:
https://github.com/naplab/naplib-python/blob/main/
naplib/preprocessing/filter_hilbert.py
Copyright (c) the naplib-python developers; MIT notice is retained in
``preprocessing/naplib_LICENSE.txt``.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft

BASE_FREQUENCY_HZ = 0.018
LINEAR_LIMIT_HZ = 4.0
OCTAVE_STEP = 1.0 / 7.0
SIGMA_SCALE = 0.39
SIGMA_EXPONENT = 0.5
SIGMA_TO_SD = np.sqrt(2.0)
OUTPUT_HZ = 200


def center_frequencies(low_hz: float, high_hz: float) -> np.ndarray:
    """Return naplib filter centers within one frequency range.

    Parameters
    ----------
    low_hz, high_hz : float
        Positive lower and larger upper band limits, in Hz.

    Returns
    -------
    ndarray, shape (frequency components,)
        Strictly increasing filter center frequencies.
    """
    if not np.isfinite([low_hz, high_hz]).all() or (
        low_hz <= 0 or high_hz <= low_hz
    ):
        raise ValueError(
            f"Expected finite 0 < low < high, got {low_hz}, {high_hz}."
        )
    centers = [BASE_FREQUENCY_HZ]
    while centers[-1] < high_hz:
        current = centers[-1]
        if current < LINEAR_LIMIT_HZ:
            sigma = SIGMA_SCALE * current ** SIGMA_EXPONENT
            centers.append(current + sigma)
        else:
            centers.append(current * 2.0 ** OCTAVE_STEP)
    selected = np.asarray(
        [value for value in centers if low_hz <= value <= high_hz],
        dtype=np.float64,
    )
    if len(selected) == 0:
        raise ValueError(
            f"No Gaussian filters have centers in {low_hz}-{high_hz} Hz."
        )
    return selected


def analytic_kernels(
    samples: int, sample_hz: int, centers: np.ndarray,
) -> np.ndarray:
    """Build naplib's one-sided Gaussian Hilbert responses.

    Parameters
    ----------
    samples : int
        Samples in each FFT window.
    sample_hz : int
        ECoG sampling rate in Hz.
    centers : ndarray, shape (frequency components,)
        Filter centers in Hz.

    Returns
    -------
    ndarray, shape (samples, frequency components)
        Real-valued FFT multipliers, including the Hilbert factor.
    """
    if samples < 2 or sample_hz <= 0 or centers.ndim != 1 or (
        len(centers) == 0 or not np.isfinite(centers).all()
    ):
        raise ValueError(
            f"Invalid FFT geometry: {samples} samples, {sample_hz} Hz, "
            f"centers shaped {centers.shape}."
        )
    if centers[-1] >= sample_hz / 2:
        raise ValueError("Filter centers must lie below Nyquist.")
    frequency = np.arange(samples // 2 + 1) * sample_hz / samples
    sigma = SIGMA_SCALE * centers ** SIGMA_EXPONENT
    standard_deviation = sigma * SIGMA_TO_SD
    distance = (
        frequency[:, None] - centers[None, :]
    ) / standard_deviation[None, :]
    kernels = np.zeros((samples, len(centers)), dtype=np.float32)
    kernels[:len(frequency)] = np.exp(-0.5 * distance ** 2)
    kernels[0] = 0.0
    if samples % 2 == 0:
        kernels[1:samples // 2] *= 2.0
    else:
        kernels[1:(samples + 1) // 2] *= 2.0
    return kernels


def window_amplitudes(
    window: np.ndarray, kernels: np.ndarray,
    output_hz: int = OUTPUT_HZ,
) -> np.ndarray:
    """Return filterbank amplitudes averaged into output timepoints.

    Parameters
    ----------
    window : ndarray, shape (samples, channels)
        One cleaned ECoG window.
    kernels : ndarray, shape (samples, frequency components)
        Analytic FFT multipliers from :func:`analytic_kernels`.
    output_hz : int, optional
        Output timepoints per one-second window; default 200.

    Returns
    -------
    ndarray, shape (channels, frequency components, output_hz)
        Mean analytic amplitude in each equal sample interval.
    """
    if window.ndim != 2 or window.shape[0] != kernels.shape[0]:
        raise ValueError(
            f"Expected window (samples, channels) with "
            f"{kernels.shape[0]} samples, got {window.shape}."
        )
    if output_hz < 1 or len(window) % output_hz:
        raise ValueError(
            f"Window length {len(window)} is not divisible by "
            f"output rate {output_hz}."
        )
    if not np.isfinite(window).all() or not np.isfinite(kernels).all():
        raise ValueError("Filterbank window or kernels are nonfinite.")
    spectrum = fft(window.astype(np.float32), axis=0)
    analytic = ifft(
        spectrum[:, :, None] * kernels[:, None, :], axis=0,
    )
    amplitude = np.abs(analytic).astype(np.float32)
    result = amplitude.reshape(
        output_hz, len(window) // output_hz,
        window.shape[1], kernels.shape[1],
    ).mean(axis=1).transpose(1, 2, 0)
    if not np.isfinite(result).all():
        raise ValueError("Filterbank amplitudes are nonfinite.")
    return result
