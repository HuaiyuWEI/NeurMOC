"""Vectorized Butterworth low-pass filtering along the time axis.

The filter uses a batched `sosfiltfilt` call. Permanently masked (all-NaN)
columns remain masked. Partly missing series are rejected rather than
misidentified as filtered data.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import FilterConfig


def mask_incomplete_series(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Mask partly missing target series and count newly masked columns."""
    values = np.asarray(data)
    flat = values.reshape(values.shape[0], -1)
    missing = np.isnan(flat)
    partial = missing.any(axis=0) & ~missing.all(axis=0)
    if not partial.any():
        return values, 0
    masked = values.astype(float, copy=True)
    masked.reshape(masked.shape[0], -1)[:, partial] = np.nan
    return masked, int(partial.sum())


def lowpass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    order: int = 5,
    sampling_rate: float = 1.0,
    padding_length: int | None = None,
) -> np.ndarray:
    """Low-pass filter `data` along axis 0 (time).

    Accepts 1-D `[time]`, 2-D `[time, feature]`, or N-D `[time, ...]` input.
    The series is reflect-padded by `padding_length` samples on both ends
    before the zero-phase filter is applied, to limit edge effects.
    All-NaN columns remain NaN. Partly missing columns and infinities raise;
    callers must make an explicit scientific choice about gap filling before
    filtering them.
    """
    sos = butter(order, cutoff_freq, btype="low", output="sos", fs=sampling_rate)
    padding_length = int(padding_length or 0)

    data = np.asarray(data)
    squeeze = data.ndim == 1
    flat = data.reshape(data.shape[0], -1) if not squeeze else data[:, None]

    if np.isinf(flat).any():
        raise ValueError("lowpass_filter: input contains infinite values")
    missing = np.isnan(flat)
    partly_missing = missing.any(axis=0) & ~missing.all(axis=0)
    if partly_missing.any():
        examples = np.flatnonzero(partly_missing)[:5].tolist()
        raise ValueError(
            "lowpass_filter: "
            f"{int(partly_missing.sum())} series contain only some missing "
            f"months (flattened columns {examples}); fill or mask those gaps "
            "explicitly before filtering"
        )
    clean = ~missing.any(axis=0)
    filtered = flat.astype(float, copy=True)

    if clean.any():
        block = flat[:, clean]
        if padding_length:
            block = np.pad(block, ((padding_length, padding_length), (0, 0)), mode="reflect")
        block = sosfiltfilt(sos, block, axis=0)
        if padding_length:
            block = block[padding_length:-padding_length]
        filtered[:, clean] = block

    return filtered[:, 0] if squeeze else filtered.reshape(data.shape)


def lowpass(data: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    """Convenience wrapper taking a `FilterConfig`."""
    return lowpass_filter(
        data,
        cutoff_freq=cfg.cutoff_freq,
        order=cfg.order,
        sampling_rate=cfg.sampling_rate,
        padding_length=cfg.padding_length,
    )


def lowpass_by_realization(
    data: np.ndarray, n_realizations: int, cfg: FilterConfig
) -> np.ndarray:
    """Filter each realization block of a concatenated `[time, ...]` array.

    Training/testing arrays concatenate several ensemble members along the
    time axis; filtering across the joins would mix unrelated series, so
    each block is filtered independently.
    """
    n_samples = data.shape[0]
    if n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    if n_samples % n_realizations:
        raise ValueError(
            f"{n_samples} samples cannot be divided into {n_realizations} equal "
            "realization blocks; pass explicit member boundaries instead")
    per_block = n_samples // n_realizations
    chunks = []
    for i in range(n_realizations):
        start = i * per_block
        stop = (i + 1) * per_block
        chunks.append(lowpass(data[start:stop], cfg))
    return np.concatenate(chunks, axis=0)


def std_by_realization(data: np.ndarray, n_realizations: int) -> np.ndarray:
    """Mean of within-realization temporal standard deviations."""
    n_samples = data.shape[0]
    if n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    if n_samples % n_realizations:
        raise ValueError(
            f"{n_samples} samples cannot be divided into {n_realizations} equal "
            "realization blocks")
    per_block = n_samples // n_realizations
    return np.mean([
        data[i * per_block:(i + 1) * per_block].std(axis=0)
        for i in range(n_realizations)
    ], axis=0)


def trend_by_realization(
    data: np.ndarray, n_realizations: int, samples_per_year: float = 12.0
) -> np.ndarray:
    """Average per-realization least-squares slopes in units per year."""
    n_samples = data.shape[0]
    if n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    if n_samples % n_realizations:
        raise ValueError(
            f"{n_samples} samples cannot be divided into {n_realizations} equal "
            "realization blocks")
    per_block = n_samples // n_realizations
    tc = np.arange(per_block, dtype=float) / samples_per_year
    tc -= tc.mean()
    ss_t = (tc**2).sum()
    trend = None
    for i in range(n_realizations):
        block = data[i * per_block:(i + 1) * per_block]
        flat = block.reshape(per_block, -1)
        slope = (tc[:, None] * (flat - flat.mean(axis=0))).sum(axis=0) / ss_t
        slope = slope.reshape(block.shape[1:]) / n_realizations
        trend = slope if trend is None else trend + slope
    return trend


def detrend_by_realization(data: np.ndarray, n_realizations: int) -> np.ndarray:
    """Remove a linear trend from each realization block (axis 0 = time)."""
    out = np.empty_like(data, dtype=float)
    n_samples = data.shape[0]
    if n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    if n_samples % n_realizations:
        raise ValueError(
            f"{n_samples} samples cannot be divided into {n_realizations} equal "
            "realization blocks; pass explicit member boundaries instead")
    per_block = n_samples // n_realizations
    for i in range(n_realizations):
        start = i * per_block
        stop = (i + 1) * per_block
        block = data[start:stop]
        flat = block.reshape(block.shape[0], -1)
        tc = np.arange(flat.shape[0], dtype=float)
        tc -= tc.mean()
        slope = (tc[:, None] * (flat - flat.mean(axis=0))).sum(axis=0) / (tc**2).sum()
        trend = tc[:, None] * slope + flat.mean(axis=0)
        out[start:stop] = (flat - trend).reshape(block.shape)
    return out
