"""Write the reconstruction and its trend statistics to NPZ and NetCDF files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .moc_utils import TWO_SIGMA_ALPHA, TrendResult
from .plotting.timeseries import GRACE_GAP_MONTHS


def save_realworld_trend_npz(
    rw,
    trend: TrendResult,
    path: Path | str,
    *,
    n_sigma: float = 2.0,
    alpha_fdr: float = 2.0 * TWO_SIGMA_ALPHA,
    block_months: int = 48,
    n_boot: int = 1000,
    seed: int = 0,
    edge_months: int = 12,
    extra_fields: dict | None = None,
) -> Path:
    """Save the trend fit, its uncertainty components, and significance masks.

    The file holds the values and test settings used by the later stages and
    figures; all trend statistics come from :func:`robust_trend`.
    """
    slope = np.asarray(trend.slope_mean)
    expected = (np.asarray(rw.sigma2).size, np.asarray(rw.lat).size)
    if slope.shape != expected:
        raise ValueError(
            f"trend shape {slope.shape} does not match coordinates {expected}"
        )

    pval = np.asarray(trend.slope_pval)
    pred = np.asarray(rw.pred)
    if pred.ndim != 3 or pred.shape[1:] != expected:
        raise ValueError(
            f"reconstruction shape {pred.shape} is incompatible with {expected}"
        )
    # A finite p-value alone is insufficient for structural-zero cells.
    testable = np.isfinite(pval) & (np.std(pred, axis=0, ddof=0) > 0)
    sig = trend.is_significant(n_sigma).astype("float32")
    sig_fdr = trend.is_significant_fdr(
        alpha_fdr, n_sigma=n_sigma, test_mask=testable
    ).astype("float32")
    sig[~testable] = np.nan
    sig_fdr[~testable] = np.nan

    if rw.time_month is None:
        time_month_int = np.empty(0, dtype="int64")
    else:
        time_month_int = (
            np.asarray(rw.time_month).astype("datetime64[M]").astype("int64")
        )

    payload = {
        "schema_version": np.asarray(1, dtype="int64"),
        "slope_mean": slope,
        "intercept_mean": np.asarray(trend.intercept_mean),
        "slope_interval_2sigma": np.asarray(trend.slope_interval_2sigma),
        "slope_pval": pval,
        "sigma_serial": np.asarray(trend.sigma_serial),
        "sigma_eps": np.asarray(trend.sigma_eps),
        "sigma_map": np.asarray(trend.sigma_map),
        "sigma_sate": np.asarray(trend.sigma_sate),
        "sigma_grace": (
            np.zeros(expected, dtype=float)
            if trend.sigma_grace is None
            else np.asarray(trend.sigma_grace)
        ),
        "sigma_total": np.asarray(trend.sigma_total),
        "significant": sig,
        "significant_fdr": sig_fdr,
        "testable": testable.astype("uint8"),
        "n_sigma": np.asarray(float(n_sigma)),
        "alpha_fdr": np.asarray(float(alpha_fdr)),
        "block_months": np.asarray(int(block_months), dtype="int64"),
        "n_boot": np.asarray(int(n_boot), dtype="int64"),
        "seed": np.asarray(int(seed), dtype="int64"),
        "edge_months": np.asarray(int(edge_months), dtype="int64"),
        "method": np.asarray(str(trend.method)),
        "method_code": np.asarray(
            {"mbb": 1}.get(str(trend.method), 0),
            dtype="int64",
        ),
        "lat": np.asarray(rw.lat),
        "sigma2": np.asarray(rw.sigma2),
        "t_years": np.asarray(rw.t_years),
        "time_month_int": time_month_int,
    }
    if rw.moc_baseline is not None:
        payload["moc_baseline"] = np.asarray(rw.moc_baseline)
    for key in ("run_id", "cmip_dataset_id", "satellite_dataset_id"):
        value = getattr(rw, key, None)
        if value is not None:
            payload[key] = np.asarray(str(value))
    if extra_fields:
        overlap = set(payload).intersection(extra_fields)
        if overlap:
            raise ValueError(
                "extra_fields must not replace existing fields: "
                + ", ".join(sorted(overlap))
            )
        payload.update(extra_fields)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    print("saved", path)
    return path


def save_neurmoc_netcdf(
    rw,
    trend: TrendResult,
    path: Path | str,
    n_sigma: float = 2.0,
    alpha_fdr: float = 2.0 * TWO_SIGMA_ALPHA,
    extra_attrs: dict | None = None,
) -> Path:
    """Write the reconstruction (`RealWorldReconstruction`) and trends to NetCDF."""
    import xarray as xr

    # Untestable cells (undefined trend uncertainty) are NaN, distinguishing
    # them from tested, non-significant cells.
    testable = (
        np.isfinite(trend.slope_pval)
        & (np.std(np.asarray(rw.pred), axis=0, ddof=0) > 0)
    )
    sig = trend.is_significant(n_sigma).astype("float32")
    sig_fdr = trend.is_significant_fdr(
        alpha_fdr, n_sigma=n_sigma, test_mask=testable
    ).astype("float32")
    sig[~testable] = np.nan
    sig_fdr[~testable] = np.nan

    months = np.asarray(rw.time_month).astype("datetime64[M]")
    # Report departures from the record mean at each latitude-density point.
    pred = np.asarray(rw.pred, dtype=float)
    moc = pred - pred.mean(axis=0, keepdims=True)
    two_sigma = f"{n_sigma:g} x trend_uncertainty_total"
    data_vars = {
        "moc": (
            ("time", "sigma2", "lat"), moc.astype("float32"),
            {"long_name": "reconstructed meridional overturning streamfunction anomaly",
             "units": "Sv",
             "comment": (
                 f"Departure from the {months[0]} to {months[-1]} mean at each "
                 "latitude-density point. The network reconstructs anomalies from "
                 "satellite inputs that are anomalies relative to their January "
                 "2004-December 2009 means; the record mean of the reconstruction "
                 "is then removed. The absolute real-ocean mean circulation is not "
                 "reconstructed.")},
        ),
        "moc_uncertainty": (
            ("time", "sigma2", "lat"), rw.total_uncertainty.astype("float32"),
            {"long_name": "one-sigma monthly reconstruction uncertainty", "units": "Sv",
             "comment": ("cross-model mapping, satellite-product, GRACE input, and "
                         "network-ensemble uncertainties combined in quadrature")},
        ),
        "trend": (
            ("sigma2", "lat"), trend.slope_mean.astype("float32"),
            {"long_name": (f"linear trend of the reconstruction, "
                           f"{months[0]} to {months[-1]}"),
             "units": "Sv yr-1", "comment": "ordinary least-squares fit with intercept"},
        ),
        "trend_ci_lower": (
            ("sigma2", "lat"), trend.slope_interval_2sigma[0].astype("float32"),
            {"long_name": f"trend minus {two_sigma}", "units": "Sv yr-1"},
        ),
        "trend_ci_upper": (
            ("sigma2", "lat"), trend.slope_interval_2sigma[1].astype("float32"),
            {"long_name": f"trend plus {two_sigma}", "units": "Sv yr-1"},
        ),
        "trend_significant": (
            ("sigma2", "lat"), sig,
            {"long_name": f"pointwise trend significance: |trend| > {two_sigma}",
             "comment": "1 = significant, 0 = not significant, NaN = not testable"},
        ),
        "trend_significant_fdr": (
            ("sigma2", "lat"), sig_fdr,
            {"long_name": ("trend significance under both the pointwise criterion and "
                           "Benjamini-Hochberg false discovery rate control "
                           f"(q = {alpha_fdr:.3f})"),
             "comment": "1 = significant, 0 = not significant, NaN = not testable"},
        ),
        "trend_uncertainty_total": (
            ("sigma2", "lat"), trend.sigma_total.astype("float32"),
            {"long_name": "one-sigma trend uncertainty", "units": "Sv yr-1",
             "comment": ("serial-sampling (circular moving-block bootstrap), "
                         "cross-model mapping, satellite-product, GRACE input, and "
                         "network-ensemble uncertainties combined in quadrature")},
        ),
    }
    if rw.moc_baseline is not None:
        data_vars["moc_baseline"] = (
            ("sigma2", "lat"), np.asarray(rw.moc_baseline, dtype="float32"),
            {"long_name": "ACCESS-ESM1.5 ensemble-mean MOC, January 2004-December 2009",
             "units": "Sv",
             "comment": ("training-model reference state of the network's anomalies; "
                         "used to locate the cell cores and to set the sign convention "
                         "of the abyssal cells; not an estimate of the real-ocean mean")},
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", months.astype("datetime64[ns]"),
                     {"long_name": "time (first day of each month)"}),
            "sigma2": ("sigma2", rw.sigma2,
                       {"long_name": "potential density anomaly referenced to 2000 dbar",
                        "units": "kg m-3", "positive": "down"}),
            "lat": ("lat", rw.lat,
                    {"long_name": "latitude", "units": "degrees_north"}),
        },
        attrs={
            "title": ("NeurMOC: satellite-based reconstruction of Atlantic and Southern "
                      "Ocean meridional overturning circulation changes"),
            "version": __version__,
            "source": (f"dual-branch neural network trained on "
                       f"{getattr(rw, 'trained_on', 'CMIP6 simulations')}; inputs: "
                       f"{', '.join(getattr(rw, 'input_sources', ()))}"),
            "time_coverage_start": f"{months[0]}",
            "time_coverage_end": f"{months[-1]}",
            "anomaly_definition": (f"moc is the departure from its {months[0]} to "
                                   f"{months[-1]} mean at each latitude-density point"),
            "grace_gap_start_month": GRACE_GAP_MONTHS[0],
            "grace_gap_end_month": GRACE_GAP_MONTHS[1],
            "grace_gap_comment": ("GRACE/GRACE-FO months in this range are missing and "
                                  "were linearly interpolated in the bottom-pressure input"),
            "Conventions": "CF-1.8",
            "history": f"created {datetime.now(timezone.utc):%Y-%m-%d}",
            **(extra_attrs or {}),
        },
    )

    path = Path(path)
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    print("saved", path)
    return path
