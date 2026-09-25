"""MOC cell-core extraction, skill metrics, and trend diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

# Two-sided Gaussian tail probability beyond two standard deviations.
TWO_SIGMA_ALPHA = float(2.0 * norm.sf(2.0))          # 0.0455


def find_nearest_index(array: np.ndarray, value: float) -> int:
    """Index of the element of `array` closest to `value`."""
    return int(np.argmin(np.abs(np.asarray(array) - value)))


def moving_smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average with edge shrinking (MATLAB `smooth(...,'moving')`).

    NaN entries are ignored within each window (and stay NaN where the
    whole window is NaN), so gaps do not drag their neighbours."""
    y = np.asarray(y, dtype=float)
    half = window // 2
    out = np.empty_like(y)
    n = y.size
    for i in range(n):
        k = min(i, n - 1 - i, half)
        block = y[i - k : i + k + 1]
        valid = block[~np.isnan(block)]
        out[i] = valid.mean() if valid.size else np.nan
    return out


# ---------------------------------------------------------------------------
# Masked flatten / unflatten of the (density, latitude) plane
# ---------------------------------------------------------------------------
def flatten_valid(field_tzy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten `[time, n_lev, n_lat]` -> `[time, n_valid]` dropping NaN columns.

    Returns `(flat, mask)` where `mask` is the boolean validity over the
    flattened `n_lev * n_lat` plane (True where the water column exists).
    """
    nt = field_tzy.shape[0]
    flat = field_tzy.reshape(nt, -1)
    mask = ~np.isnan(flat).any(axis=0)
    return flat[:, mask], mask


def unflatten(values: np.ndarray, mask: np.ndarray, n_lev: int, n_lat: int) -> np.ndarray:
    """Inverse of `flatten_valid` for `[time, n_valid]` or `[n_valid]` arrays."""
    values = np.asarray(values)
    if values.ndim == 1:
        full = np.full(n_lev * n_lat, np.nan)
        full[mask] = values
        return full.reshape(n_lev, n_lat)
    full = np.full((values.shape[0], n_lev * n_lat), np.nan)
    full[:, mask] = values
    return full.reshape(values.shape[0], n_lev, n_lat)


# ---------------------------------------------------------------------------
# Cell-core ("MOC strength") extraction
# ---------------------------------------------------------------------------
@dataclass
class CellCores:
    """Density indices/values of the mid-depth and abyssal overturning cells.

    The mid-depth core is the density level of the time-mean maximum at each
    latitude; the abyssal core is the level of the time-mean minimum. Both
    index curves are smoothed with a 5-point moving average.
    """

    mid_index: np.ndarray       # [n_lat] density index of the mid-depth core
    abyssal_index: np.ndarray   # [n_lat]
    mid_sigma2: np.ndarray      # [n_lat] sigma2 at the mid-depth core (NaN S of -55.5)
    abyssal_sigma2: np.ndarray  # [n_lat]


def locate_cell_cores(
    moc_mean_zy: np.ndarray,
    lat: np.ndarray,
    sigma2: np.ndarray,
    smooth_window: int = 5,
    mid_cutoff_lat: float = -55.5,
) -> CellCores:
    """Locate the mid-depth and abyssal cell cores from a time-mean MOC.

    `moc_mean_zy` is `[n_lev, n_lat]`; `sigma2` are the density levels
    (already offset by -1000 or not - returned values follow the input).
    Latitudes with no valid water column (all-NaN, e.g. the Southern Ocean
    in an Atlantic-only cross-model test) get index 0 and a NaN core density.
    """
    n_lev = moc_mean_zy.shape[0]
    empty = np.isnan(moc_mean_zy).all(axis=0)   # latitudes with no valid column
    filled = np.where(empty, 0.0, moc_mean_zy)  # placeholder for argmax/argmin

    def smoothed_index(raw: np.ndarray) -> np.ndarray:
        # Exclude empty columns from smoothing near domain edges.
        curve = raw.astype(float)
        curve[empty] = np.nan
        curve = moving_smooth(curve, smooth_window)
        curve[np.isnan(curve)] = 0.0
        return np.clip(np.round(curve), 0, n_lev - 1).astype(int)

    mid = smoothed_index(np.nanargmax(filled, axis=0))
    aby = smoothed_index(np.nanargmin(filled, axis=0))

    sigma2 = np.asarray(sigma2, dtype=float)
    mid_sigma2 = sigma2[mid].copy()
    mid_sigma2[np.asarray(lat) < mid_cutoff_lat] = np.nan
    mid_sigma2[empty] = np.nan
    abyssal_sigma2 = sigma2[aby].copy()
    abyssal_sigma2[empty] = np.nan

    return CellCores(mid_index=mid, abyssal_index=aby,
                     mid_sigma2=mid_sigma2, abyssal_sigma2=abyssal_sigma2)


def extract_at_cores(field_tzy: np.ndarray, level_index: np.ndarray) -> np.ndarray:
    """Sample `[time, n_lev, n_lat]` at one density level per latitude."""
    n_lat = field_tzy.shape[2]
    return field_tzy[:, level_index, np.arange(n_lat)]


# ---------------------------------------------------------------------------
# Skill metrics on the (density, latitude) plane
# ---------------------------------------------------------------------------
def pointwise_r(truth_tzy: np.ndarray, pred_tzy: np.ndarray) -> np.ndarray:
    """Signed Pearson correlation at every (density, latitude) point."""
    ta = truth_tzy - truth_tzy.mean(axis=0)
    pa = pred_tzy - pred_tzy.mean(axis=0)
    num = (ta * pa).sum(axis=0)
    den = np.sqrt((ta**2).sum(axis=0) * (pa**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r


def pointwise_r2(truth_tzy: np.ndarray, pred_tzy: np.ndarray) -> np.ndarray:
    """Squared Pearson correlation at every (density, latitude) point."""
    r = pointwise_r(truth_tzy, pred_tzy)
    return r**2


def pointwise_rmse(truth_tzy: np.ndarray, pred_tzy: np.ndarray) -> np.ndarray:
    """RMSE at every (density, latitude) point."""
    return np.sqrt(np.mean((truth_tzy - pred_tzy) ** 2, axis=0))


def fill_down_columns(field_zy: np.ndarray) -> np.ndarray:
    """Fill NaN cells from the nearest finite level ABOVE in each column.

    Used where a held-out model's density grid ends before the reconstruction
    grid. Entirely missing columns remain NaN.
    """
    field = np.asarray(field_zy, dtype=float)
    n_lev = field.shape[0]
    lev_of_finite = np.where(np.isfinite(field),
                             np.arange(n_lev)[:, None], -1)
    src = np.maximum.accumulate(lev_of_finite, axis=0)
    filled = np.take_along_axis(np.nan_to_num(field, nan=0.0),
                                np.maximum(src, 0), axis=0)
    return np.where(src >= 0, filled, np.nan)


# ---------------------------------------------------------------------------
# Trend estimation with serial-correlation-aware uncertainty
# ---------------------------------------------------------------------------
@dataclass
class TrendResult:
    """Pointwise OLS trends and their uncertainty components."""

    slope_mean: np.ndarray       # [n_lev, n_lat] Sv / yr (OLS estimate)
    #: [2, n_lev, n_lat] slope +/- 2*sigma_total (95.45% if Gaussian).
    slope_interval_2sigma: np.ndarray
    slope_pval: np.ndarray       # [n_lev, n_lat] two-sided z-test p-value
    intercept_mean: np.ndarray   # [n_lev, n_lat]
    sigma_serial: np.ndarray     # [n_lev, n_lat] serial-correlation term
    #: [n_lev, n_lat] spread of network-member trends.
    sigma_eps: np.ndarray
    sigma_map: np.ndarray      # [n_lev, n_lat] mapping trend error
    sigma_sate: np.ndarray       # [n_lev, n_lat] satellite-product spread
    sigma_total: np.ndarray      # [n_lev, n_lat] quadrature combination
    method: str                  # "mbb" (circular moving-block bootstrap)
    #: [n_lev, n_lat] propagated GRACE input-error trend spread, if available.
    sigma_grace: np.ndarray | None = None

    def is_significant(self, n_sigma: float = 2.0) -> np.ndarray:
        """Mask where |slope| exceeds `n_sigma * sigma_total`."""
        with np.errstate(invalid="ignore"):
            return np.abs(self.slope_mean) > n_sigma * self.sigma_total

    def is_significant_fdr(
        self,
        alpha_fdr: float = 2.0 * TWO_SIGMA_ALPHA,
        *,
        n_sigma: float = 2.0,
        test_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Benjamini-Hochberg mask intersected with the local threshold.

        P-values use a Gaussian approximation. `test_mask` defines the
        tested cells; the result always passes `is_significant(n_sigma)`.
        """
        alpha_fdr = float(alpha_fdr)
        n_sigma = float(n_sigma)
        if not np.isfinite(alpha_fdr) or not 0.0 < alpha_fdr <= 1.0:
            raise ValueError("alpha_fdr must be finite and in (0, 1]")
        if not np.isfinite(n_sigma) or n_sigma <= 0.0:
            raise ValueError("n_sigma must be finite and positive")
        p = self.slope_pval
        finite = np.isfinite(p)
        if test_mask is not None:
            test_mask = np.asarray(test_mask, dtype=bool)
            if test_mask.shape != p.shape:
                raise ValueError(
                    f"test_mask shape {test_mask.shape} does not match "
                    f"p-value shape {p.shape}"
                )
            finite &= test_mask
        out = np.zeros(p.shape, dtype=bool)
        if not finite.any():
            return out
        vals = np.sort(p[finite].ravel())
        ok = vals <= np.arange(1, vals.size + 1) / vals.size * alpha_fdr
        if not ok.any():
            return out
        bh_thresh = vals[np.flatnonzero(ok).max()]
        local_alpha = float(2.0 * norm.sf(n_sigma))
        thresh = min(bh_thresh, local_alpha)
        out[finite] = p[finite] <= thresh
        # Preserve the local threshold after FDR adjustment.
        return out & self.is_significant(n_sigma)


def robust_trend(
    pred_tzy: np.ndarray,
    t_years: np.ndarray,
    method: str = "mbb",
    block_months: int = 48,
    n_boot: int = 1000,
    seed: int | None = 0,
    sigma_map: np.ndarray | None = None,
    sigma_sate: np.ndarray | None = None,
    sigma_eps: np.ndarray | None = None,
    sigma_grace: np.ndarray | None = None,
) -> TrendResult:
    """Fit pointwise OLS trends and combine uncertainty components.

    The serial-sampling uncertainty is estimated with a circular
    moving-block bootstrap of the regression residuals (`method="mbb"`);
    `block_months=0` resamples individual months.
    The optional `sigma_*` inputs are uncertainties estimated in trend space;
    missing components contribute zero. Their variances are summed with
    serial variance. Transfer bias is not included in the symmetric spread.
    """
    rng = np.random.default_rng(seed)
    nt, n_lev, n_lat = pred_tzy.shape
    t = np.asarray(t_years, dtype=float)
    y = pred_tzy.reshape(nt, -1)                     # [nt, P]

    xc = t - t.mean()
    a = xc / (xc**2).sum()          # [nt]; slope = sum_t a_t y_t, all points
    slope = a @ y
    intercept = y.mean(axis=0) - slope * t.mean()
    resid = y - (intercept + slope * t[:, None])
    # An absent ensemble term contributes zero; no surrogate is estimated.
    s_eps = (np.zeros(n_lev * n_lat) if sigma_eps is None
             else np.asarray(sigma_eps, dtype=float).reshape(-1))

    if method == "mbb":
        # Zero selects independent-month resampling.
        requested_block = int(block_months)
        block = 1 if requested_block == 0 else min(requested_block, nt)
        n_blocks = int(np.ceil(nt / block))
        deltas = np.empty((n_boot, y.shape[1]))
        offsets = np.arange(block)
        for b in range(n_boot):
            starts = rng.integers(0, nt, n_blocks)
            idx = (starts[:, None] + offsets).ravel()[:nt] % nt
            deltas[b] = a @ resid[idx]
        sigma_serial = deltas.std(axis=0, ddof=1)
    else:
        raise ValueError(f"unknown trend method {method!r}; only \"mbb\" is implemented")

    s_map = (np.zeros(n_lev * n_lat) if sigma_map is None
             else np.asarray(sigma_map, dtype=float).reshape(-1))
    s_sate = (np.zeros(n_lev * n_lat) if sigma_sate is None
              else np.asarray(sigma_sate, dtype=float).reshape(-1))
    s_grace = (np.zeros(n_lev * n_lat) if sigma_grace is None
               else np.asarray(sigma_grace, dtype=float).reshape(-1))
    sigma_total = np.sqrt(sigma_serial**2 + s_eps**2
                          + s_map**2 + s_sate**2 + s_grace**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = slope / sigma_total
    pval = 2.0 * norm.sf(np.abs(z))
    ci = np.stack([slope - 2.0 * sigma_total, slope + 2.0 * sigma_total])

    def reshape(arr):
        return arr.reshape(arr.shape[:-1] + (n_lev, n_lat))

    return TrendResult(
        slope_mean=reshape(slope), slope_interval_2sigma=reshape(ci),
        slope_pval=reshape(pval), intercept_mean=reshape(intercept),
        sigma_serial=reshape(sigma_serial), sigma_eps=reshape(s_eps),
        sigma_map=reshape(s_map), sigma_sate=reshape(s_sate),
        sigma_total=reshape(sigma_total), method=method,
        sigma_grace=reshape(s_grace))


def sliding_window_trends(
    field_tzy: np.ndarray, window_months: int, step_months: int = 12
) -> np.ndarray:
    """OLS slopes of sliding monthly windows, returned as `[n_win, ...]`."""
    nt = field_tzy.shape[0]
    if window_months > nt:
        raise ValueError(f"window {window_months} exceeds record {nt}")
    x = np.arange(window_months) / 12.0
    xc = x - x.mean()
    a = xc / (xc**2).sum()
    starts = np.arange(0, nt - window_months + 1, step_months)
    flat = field_tzy.reshape(nt, -1)
    out = np.stack([a @ flat[s:s + window_months] for s in starts])
    return out.reshape((starts.size,) + field_tzy.shape[1:])


def simple_trend(
    series_t_xy: np.ndarray, t_years: np.ndarray, dof: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares trend (slope, r, p) of `[time, ...]` series, vectorized."""
    nt = series_t_xy.shape[0]
    flat = series_t_xy.reshape(nt, -1)
    t = np.asarray(t_years, dtype=float)
    tc = t - t.mean()
    ss_t = (tc**2).sum()

    ya = flat - flat.mean(axis=0)
    slope = (tc[:, None] * ya).sum(axis=0) / ss_t
    fit = tc[:, None] * slope
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (fit * ya).sum(axis=0) / np.sqrt((fit**2).sum(axis=0) * (ya**2).sum(axis=0))
    if dof is None:
        dof = nt / 10.0
    r_clip = np.clip(r, -0.999999, 0.999999)
    tval = r_clip * np.sqrt(dof / (1.0 - r_clip**2))
    p = 2.0 * (1.0 - student_t.cdf(np.abs(tval), dof))

    shape = series_t_xy.shape[1:]
    return slope.reshape(shape), r.reshape(shape), p.reshape(shape)
