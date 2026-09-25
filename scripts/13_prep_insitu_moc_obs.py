"""Stage 13: prepare in-situ and state-estimate comparison time series.

Process observational overturning records used to compare against the
satellite-based reconstruction. These records are not used for training.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import (
    ECCO_V4R3_OBS_DIR,
    ECCO_V4R3_OVERTURNING_DIR,
    LPF_OBS,
    OSNAP_DIR,
    OSNAP_RAW_DIR,
    RAPID_DIR,
    RAPID_RAW_DIR,
)
from neurmoc.filtering import lowpass
from neurmoc.io_utils import ensure_dir, load_mat, require_file
from neurmoc.timeaxis import decimal_year, normalize_month_axis

# ========== User settings ==========
RUN = {"rapid": True, "rapid_full_depth": True, "osnap": True,
       "ecco_v4r3": True}

OSNAP_PRODUCT_SCHEMA_VERSION = 2
OSNAP_SOURCE_VARIABLE = "MOC_ALL"
OSNAP_UNCERTAINTY_SOURCE_VARIABLE = "MOC_ALL_ERR"
OSNAP_MOC_SOURCE = (
    OSNAP_RAW_DIR / "OSNAP_MOC_MHT_MFT_TimeSeries_201408_202207_2025.nc"
)

# Preserve ECCO's native reference for the observational comparison.


def month_labels(values) -> np.ndarray:
    """Portable `YYYY-MM` coordinate for NPZ products."""
    return np.asarray(values).astype("datetime64[M]").astype("U7")


def validate_osnap_monthly_series(
    moc, uncertainty, time_values
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the provider's official OSNAP MOC and monthly time axis.

    ``MOC_ALL`` is the Monte-Carlo mean of the maximum overturning in each
    realization.  It must not be reconstructed as ``max(T_ALL.mean)``:
    taking a maximum and taking an ensemble mean do not commute.
    """
    monthly = np.asarray(moc, dtype=float).squeeze()
    monthly_uncertainty = np.asarray(uncertainty, dtype=float).squeeze()
    if monthly.ndim != 1 or monthly_uncertainty.ndim != 1:
        raise RuntimeError(
            "OSNAP MOC_ALL and MOC_ALL_ERR must each be one-dimensional"
        )
    if monthly.size != monthly_uncertainty.size:
        raise RuntimeError(
            "OSNAP MOC_ALL and MOC_ALL_ERR have different time-axis lengths"
        )
    if not np.isfinite(monthly).all():
        raise RuntimeError("OSNAP MOC_ALL contains non-finite values")
    if (
        not np.isfinite(monthly_uncertainty).all()
        or np.any(monthly_uncertainty < 0)
    ):
        raise RuntimeError(
            "OSNAP MOC_ALL_ERR must contain finite, non-negative values"
        )

    months = month_labels(normalize_month_axis(
        time_values, monthly.size, "OSNAP TIME"
    ))
    return monthly, monthly_uncertainty, months


def prep_rapid():
    import xarray as xr

    print("=== RAPID (26.5N transport) ===")
    ds = xr.open_dataset(require_file(RAPID_RAW_DIR / "moc_transports.nc", "RAPID"))
    monthly_da = ds["moc_mar_hc10"].resample(time="1M").mean()
    monthly = monthly_da.values
    monthly_lpf = lowpass(monthly, LPF_OBS)
    months = month_labels(monthly_da.time.values)
    t_year = decimal_year(months)
    out_dir = ensure_dir(RAPID_DIR)
    np.savez(out_dir / "Rapid_LPF.npz", RAPID_monthly=monthly,
             RAPID_monthly_LPF=monthly_lpf, t_year=t_year, time_month=months,
             source_file=str(RAPID_RAW_DIR / "moc_transports.nc"))
    print("saved Rapid_LPF.npz", monthly.shape)


def prep_rapid_full_depth():
    import xarray as xr

    print("=== RAPID (full-depth stream functions) ===")
    ds = xr.open_dataset(require_file(
        RAPID_RAW_DIR / "meridional_transports.nc", "RAPID"))
    out = {}
    for var in ["stream_depth", "stream_sigma2", "stream_sigma0"]:
        monthly = ds[var].resample(time="1M").mean().values
        out[f"{var}_LPF"] = lowpass(monthly, LPF_OBS)
    for coord in ["depth", "sigma2", "sigma0"]:
        out[coord] = ds[coord].values
    n_t = out["stream_depth_LPF"].shape[0]
    months = month_labels(ds["stream_depth"].resample(time="1M").mean().time.values)
    if months.size != n_t:
        raise RuntimeError("RAPID full-depth monthly coordinate length mismatch")
    out["time_month"] = months
    out["t_year"] = decimal_year(months)
    out["source_file"] = str(RAPID_RAW_DIR / "meridional_transports.nc")
    out_dir = ensure_dir(RAPID_DIR)
    np.savez(out_dir / "Rapid_FullDepth_LPF.npz", **out)
    print("saved Rapid_FullDepth_LPF.npz")


def prep_osnap():
    import xarray as xr

    print("=== OSNAP ===")
    source = require_file(OSNAP_MOC_SOURCE, "OSNAP")
    with xr.open_dataset(source) as ds:
        missing = [
            name for name in (
                "TIME", OSNAP_SOURCE_VARIABLE,
                OSNAP_UNCERTAINTY_SOURCE_VARIABLE,
            )
            if name not in ds
        ]
        if missing:
            raise RuntimeError(
                f"OSNAP source {source} is missing variables {missing}"
            )
        monthly, monthly_uncertainty, months = validate_osnap_monthly_series(
            ds[OSNAP_SOURCE_VARIABLE].values,
            ds[OSNAP_UNCERTAINTY_SOURCE_VARIABLE].values,
            ds["TIME"].values,
        )
        source_definition = str(
            ds[OSNAP_SOURCE_VARIABLE].attrs.get("comment", "")
        )
        uncertainty_definition = str(
            ds[OSNAP_UNCERTAINTY_SOURCE_VARIABLE].attrs.get("comment", "")
        )
    monthly_lpf = lowpass(monthly, LPF_OBS)
    t_year = decimal_year(months)
    out_dir = ensure_dir(OSNAP_DIR)
    np.savez(
        out_dir / "OSNAP_LPF.npz",
        OSNAP_monthly=monthly,
        OSNAP_monthly_LPF=monthly_lpf,
        # Provider uncertainty applies to unfiltered monthly MOC only.
        OSNAP_monthly_uncertainty=monthly_uncertainty,
        t_year=t_year,
        time_month=months,
        product_schema_version=np.asarray(
            OSNAP_PRODUCT_SCHEMA_VERSION, dtype=np.int64
        ),
        source_file=str(source),
        source_variable=OSNAP_SOURCE_VARIABLE,
        source_definition=source_definition,
        uncertainty_source_variable=OSNAP_UNCERTAINTY_SOURCE_VARIABLE,
        uncertainty_definition=uncertainty_definition,
    )
    print(
        "saved OSNAP_LPF.npz",
        monthly.shape,
        f"from {OSNAP_SOURCE_VARIABLE}",
    )


def prep_ecco_v4r3():
    print("=== ECCO v4r3 ===")
    source = require_file(
        ECCO_V4R3_OVERTURNING_DIR / "PSItot_AtlOnly.mat", "ECCO v4r3"
    )
    data = load_mat(source, ["PSItot", "lat"])
    moc = np.asarray(data["PSItot"]).T / 1e6  # -> [time, rho, lat], Sv
    lat = np.asarray(data["lat"]).squeeze().astype(float)
    if lat.ndim != 1 or lat.size != moc.shape[-1]:
        raise RuntimeError(
            f"ECCO v4r3 latitude/MOC mismatch: lat {lat.shape}, MOC {moc.shape}"
        )
    moc_lpf = lowpass(moc, LPF_OBS)
    months = np.arange(np.datetime64("1992-01"),
                       np.datetime64("1992-01") + moc.shape[0],
                       dtype="datetime64[M]").astype("U7")
    t_year = decimal_year(months)
    out_dir = ensure_dir(ECCO_V4R3_OBS_DIR)
    np.savez(out_dir / "PSI_LPF.npz", MOC_ecco=moc, MOC_ecco_LPF=moc_lpf,
             lat_ecco=lat, t_year_ecco=t_year, time_month=months,
             bottom_referenced=np.bool_(False),  # native; consumers choose
             source_file=str(source))
    print("saved PSI_LPF.npz", moc.shape)


def main() -> None:
    steps = {
        "rapid": prep_rapid, "rapid_full_depth": prep_rapid_full_depth,
        "osnap": prep_osnap,
        "ecco_v4r3": prep_ecco_v4r3,
    }
    for name, fn in steps.items():
        if RUN.get(name):
            fn()


if __name__ == "__main__":
    main()
