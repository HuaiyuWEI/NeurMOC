"""Shared monthly time axis and decimal-year convention.

For example, April 2002 maps to 2002 + 4/12.
"""

from __future__ import annotations

import numpy as np


def _strip_string_padding(values: np.ndarray) -> np.ndarray:
    """Remove fixed-width padding without coercing non-string timestamps.

    MATLAB character arrays loaded through SciPy can preserve the nominal
    NumPy Unicode width as trailing spaces (for example ``"2002-04   "``).
    NumPy's datetime parser rejects that otherwise valid monthly label.
    """
    if values.dtype.kind in {"U", "S"}:
        return np.char.strip(values.astype("U"))
    if values.dtype.kind == "O":
        stripped = values.copy()
        flat = stripped.reshape(-1)
        for index, value in enumerate(flat):
            if isinstance(value, bytes):
                flat[index] = value.decode().strip()
            elif isinstance(value, str):
                flat[index] = value.strip()
        return stripped
    return values


def _as_datetime_months(values) -> np.ndarray:
    """Convert values to monthly datetimes after normalizing text padding."""
    raw = _strip_string_padding(np.asarray(values))
    return raw.astype("datetime64[M]")


def decimal_year(months) -> np.ndarray:
    """Decimal years of a monthly coordinate, as `year + month_number / 12`.

    Accepts anything convertible to ``datetime64[M]``: normalized `YYYY-MM`
    strings (the pipeline's `time_month` arrays), `datetime64` values of any
    unit, or pandas timestamps.
    """
    m = _as_datetime_months(months)
    if m.size and np.isnat(m).any():
        raise ValueError("decimal_year: monthly coordinate contains NaT")
    months_since_epoch = m.astype(np.int64)  # months since 1970-01
    return 1970.0 + (months_since_epoch + 1) / 12.0


def normalize_month_axis(time, n_samples: int, label: str) -> np.ndarray:
    """Return a strict, contiguous, unique ``datetime64[M]`` coordinate.

    The shared gate for every monthly time axis entering the pipeline
    (stage-12 satellite products, stage-14 observation files): one
    timestamp per sample, all convertible, no NaT, strictly consecutive
    months. `label` names the offending record in error messages.
    """
    raw = np.asarray(time).reshape(-1)
    if raw.size != n_samples:
        raise RuntimeError(
            f"{label}: {raw.size} timestamps for {n_samples} samples")
    try:
        months = _as_datetime_months(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label}: invalid monthly timestamps") from exc
    if np.isnat(months).any():
        raise RuntimeError(f"{label}: time coordinate contains NaT")
    if months.size > 1:
        steps = np.diff(months).astype(int)
        if np.any(steps != 1):
            bad = int(np.flatnonzero(steps != 1)[0])
            raise RuntimeError(
                f"{label}: monthly coordinate is not contiguous at "
                f"{months[bad]} -> {months[bad + 1]}")
    return months
