"""RAPID 26.5N comparison with consistent reference and month alignment.

Reference to 2004-2009 precedes edge trimming. Statistics use the exactly
matched months after removing each series' mean over that match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LPF_OBS, RAPID_DIR
from .filtering import lowpass
from .io_utils import load_npz_or_mat

# Decimal-year window for the 2009-2010 dip statistic.
DIP_WINDOW = (2009.0, 2011.5)
# Observed dip after the standard reference and edge trim, in Sv.
OBSERVED_DIP = -3.51


def reference_anomaly(series: np.ndarray, t_years: np.ndarray) -> np.ndarray:
    """Subtract the January 2004-December 2009 mean along time."""
    ref = (t_years > 2004.0 + 1e-6) & (t_years <= 2010.0 + 1e-6)
    return series - np.nanmean(series[ref], axis=0)


@dataclass(frozen=True)
class RapidRecord:
    """The referenced, edge-trimmed RAPID 26.5N transport record."""

    anomaly: np.ndarray            # [T] Sv, 2004-2009-referenced
    t_years: np.ndarray            # [T] decimal years
    time_month: np.ndarray | None  # [T] datetime64[M] (None if absent)
    uncertainty: np.ndarray | None  # [T] filtered deployment-era band


def load_rapid(edge_months: int = 12,
               with_uncertainty: bool = False) -> RapidRecord:
    """Load referenced RAPID transport with optional filtered uncertainty."""
    # An explicit slice avoids an empty result when edge_months is zero.
    sl = slice(edge_months, -edge_months) if edge_months else slice(None)
    data = load_npz_or_mat(RAPID_DIR / "Rapid_LPF")
    series_full = np.asarray(data["RAPID_monthly_LPF"]).squeeze()
    t_full = np.asarray(data["t_year"]).squeeze()
    months = (np.asarray(data["time_month"]).astype("datetime64[M]")[sl]
              if "time_month" in data else None)

    uncertainty = None
    if with_uncertainty:
        unc = np.full(t_full.size, 0.9)
        unc[(t_full > 2005) & (t_full <= 2006)] = 1.0
        unc[(t_full > 2007) & (t_full <= 2008)] = 1.3
        uncertainty = lowpass(unc, LPF_OBS)[sl]

    return RapidRecord(
        anomaly=reference_anomaly(series_full, t_full)[sl],
        t_years=t_full[sl],
        time_month=months,
        uncertainty=uncertainty,
    )


def dip_in_window(series: np.ndarray, t_years: np.ndarray,
                  window: tuple[float, float] = DIP_WINDOW) -> float:
    """Minimum of a series inside the dip window (NaN-safe)."""
    sel = (np.asarray(t_years) >= window[0]) & (np.asarray(t_years) <= window[1])
    return float(np.nanmin(np.asarray(series)[sel]))


@dataclass(frozen=True)
class RapidStats:
    """Zero-bias comparison statistics over the exactly matched months."""

    r: float          # Pearson correlation (offset-invariant)
    rmse: float       # Sv, after demeaning both series over the match
    dip: float        # Sv, min of the zero-biased prediction in DIP_WINDOW
    n_matched: int    # number of common months


def zero_bias_pair(pred: np.ndarray, pred_months: np.ndarray,
                   rapid: RapidRecord):
    """Return full series demeaned over their matched months and match indices."""
    if rapid.time_month is None:
        raise ValueError("RapidRecord has no time_month labels")
    pm = np.asarray(pred_months).astype("datetime64[M]")
    _, i_pred, i_obs = np.intersect1d(pm, rapid.time_month,
                                      return_indices=True)
    if i_pred.size == 0:
        raise ValueError("prediction and RAPID share no common months")
    pred = np.asarray(pred, dtype=float)
    pred_zb = pred - np.nanmean(pred[i_pred])
    rapid_zb = rapid.anomaly - np.nanmean(rapid.anomaly[i_obs])
    return pred_zb, rapid_zb, i_pred, i_obs


def stats_vs_rapid(pred: np.ndarray, pred_months: np.ndarray,
                   rapid: RapidRecord,
                   t_pred: np.ndarray | None = None,
                   dip_window: tuple[float, float] = DIP_WINDOW) -> RapidStats:
    """r / RMSE / dip of a predicted 26.5N series against the record.

    `pred_months` are the prediction's month labels (anything castable to
    datetime64[M]). The dip needs the prediction's own time axis `t_pred`
    (full series, not just matched months); NaN when omitted.
    """
    pred_zb, rapid_zb, i_pred, i_obs = zero_bias_pair(pred, pred_months, rapid)
    r = float(np.corrcoef(rapid_zb[i_obs], pred_zb[i_pred])[0, 1])
    rmse = float(np.sqrt(np.mean((rapid_zb[i_obs] - pred_zb[i_pred]) ** 2)))
    dip = (dip_in_window(pred_zb, t_pred, dip_window)
           if t_pred is not None else float("nan"))
    return RapidStats(r=r, rmse=rmse, dip=dip, n_matched=int(i_pred.size))
