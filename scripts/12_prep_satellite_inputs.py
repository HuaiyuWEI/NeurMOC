"""Stage 12: prepare satellite inputs for real-ocean reconstruction.

Area-average GRACE bottom pressure, altimetric sea surface height, and 10-m
wind onto the mascon grid, then form anomalies and apply the two-year low-pass
filter. Outputs in ``OBS_MASCON_ROOT/ASMOC`` include a monthly ``time_month``
coordinate and source metadata. GRACE/GRACE-FO missing months are linearly
interpolated before filtering. Optional QC plots can be redrawn from saved
products without repeating preprocessing.
"""

import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import (
    BASELINE_YEARS,
    BASINMASK_DIR,
    CCMP_RAW_DIR,
    DUACS_CACHE_DIR,
    DUACS_DAILY_DIR,
    ERA5_WIND_NC,
    GRACE_MASCON_NC,
    GSFC_MASCON_NC,
    LPF_OBS,
    NASASSH_GRID_DIR,
    OBS_MASCON_ROOT,
)
from neurmoc.cmip_io import (BULK_AIR_DENSITY, BULK_DRAG_LAW,
                            bulk_wind_stress, wind_stress_curl)
from neurmoc.filtering import lowpass
from neurmoc.grids import (
    MasconAverager,
    MasconGeometry,
    cell_area_weights,
    load_grace_land,
)
from neurmoc.io_utils import ensure_dir, load_npz_or_mat, require_dir, require_file
from neurmoc.plotting import CMAP_AMPLITUDE, CMAP_DIVERGING, apply_style, save_figure, shade_gap
from neurmoc.timeaxis import normalize_month_axis

# ========== User settings ==========
RUN_GRACE = True
RUN_DUACS = True
RUN_CCMP = True
RUN_ERA5 = True
#: Optional stress and curl inputs derived from the same 10-m wind products.
RUN_CCMP_STRESS = True
RUN_ERA5_STRESS = True
RUN_NASASSH = True
#: Build the alternative GRACE products (CSR, GSFC) as obp_GRACE_<tag>.npz.
RUN_GRACE_VARIANTS = True
#: Draw QC figures from the saved products.
QC_PLOTS = True
T0_YEARS = 2002 + 4 / 12  # all three records start April 2002 (GRACE era)
OBSERVATION_SCHEMA_VERSION = 2

#: Constant gravitational acceleration (m s^-2), matching the model convention.
GRAVITY = 9.806

#: Provider-specific reference density (kg m^-3) for equivalent water height
#: conversion: bottom pressure equals g * rho_provider * height.
EWH_DENSITY_EXPECTED = {
    "GRACE": 1000.0,          # JPL RL06 mascons
    "GRACE_CSR": 1025.0,      # CSR RL0603 mascons (seawater-equivalent)
    "GRACE_GSFC": 1000.0,     # GSFC RL06 v2.0 mascons
}

_EWH_DENSITY_ATTR = re.compile(
    r"water\s+density\s+used\s+to\s+convert\s+to\s+equivalent\s+water\s+"
    r"height:\s*([0-9]+(?:\.[0-9]+)?)\s*kg\s*/?\s*m\^?3", re.I)

OUT_DIR = OBS_MASCON_ROOT


def ewh_density(nc, tag: str) -> float:
    """Read the provider's equivalent-water-height density from the file."""
    found = None
    for attr in nc.ncattrs():
        match = _EWH_DENSITY_ATTR.search(str(getattr(nc, attr)))
        if match is not None:
            found = float(match.group(1))
    if found is None:
        raise ValueError(f"{tag}: no equivalent-water-height density in the file")
    if found != EWH_DENSITY_EXPECTED[tag]:
        raise ValueError(
            f"{tag}: file density {found} kg m^-3 differs from the expected "
            f"{EWH_DENSITY_EXPECTED[tag]} kg m^-3")
    return found


def load_geometry():
    data = load_npz_or_mat(BASINMASK_DIR / "Mascon_AtlSO")
    geometry = MasconGeometry.from_dict(data)
    return geometry, np.asarray(data["mascon_ID"]), \
        np.asarray(data["lon_mascon"]), np.asarray(data["lat_mascon"])


def shift_lon_halves(arr: np.ndarray, axis: int) -> np.ndarray:
    """Recenter a 0..360-longitude array to -180..180 (Atlantic centered)."""
    n = arr.shape[axis]
    lower = np.take(arr, range(n // 2, n), axis=axis)
    upper = np.take(arr, range(0, n // 2), axis=axis)
    return np.concatenate([lower, upper], axis=axis)


def baseline_mask(months: np.ndarray, label: str) -> np.ndarray:
    """Select and validate the complete Jan-2004 through Dec-2009 baseline."""
    start = np.datetime64(f"{BASELINE_YEARS[0]}-01", "M")
    stop = np.datetime64(f"{BASELINE_YEARS[1]}-12", "M")
    mask = (months >= start) & (months <= stop)
    expected = 12 * (BASELINE_YEARS[1] - BASELINE_YEARS[0] + 1)
    if mask.sum() != expected:
        raise RuntimeError(
            f"{label}: expected {expected} baseline months "
            f"({start}..{stop}), found {mask.sum()}")
    return mask


def atlso_lpf_save(name, data, time, geometry, out_file, demean=True,
                   extra_fields=None, static_fields=None, metadata=None,
                   nan_to_zero=False):
    """Low-pass filter and save the Atlantic/Southern Ocean mascon series.

    With ``demean``, the spatial mean over the Atlantic-Southern Ocean mascon
    domain is removed at each month.
    """
    data = np.asarray(data)
    months = normalize_month_axis(time, data.shape[0], name)
    atlso = geometry.basin_id == 1
    data = data[:, atlso]
    lon = geometry.lon_center[atlso]
    lat = geometry.lat_center[atlso]

    if demean:
        data = data - np.nanmean(data, axis=1, keepdims=True)
    if nan_to_zero:
        data = np.nan_to_num(data, nan=0.0)
    data_lpf = lowpass(data, LPF_OBS)

    payload = {
        name: data, f"{name}_LPF_ALL": data_lpf,
        f"{name}_lon": lon, f"{name}_lat": lat,
        "time_month": np.datetime_as_string(months, unit="M"),
        "time_calendar": np.str_("proleptic_gregorian"),
        "time_frequency": np.str_("monthly"),
        "observation_schema_version": np.int64(OBSERVATION_SCHEMA_VERSION),
    }
    if extra_fields:
        for key, values in extra_fields.items():
            values = values[:, atlso]
            if demean:
                values = values - np.nanmean(values, axis=1, keepdims=True)
            payload[key] = values
            payload[f"{key}_LPF_ALL"] = lowpass(values, LPF_OBS)
    if static_fields:
        for key, values in static_fields.items():
            values = np.asarray(values)
            if values.ndim == 1 and values.shape[0] == atlso.size:
                values = values[atlso]
            payload[key] = values
    if metadata:
        payload.update(metadata)
    np.savez(out_file, **payload)
    print("saved", out_file, data.shape,
          f"({payload['time_month'][0]}..{payload['time_month'][-1]})")


# ---------------------------------------------------------------------------
# GRACE ocean bottom pressure
# ---------------------------------------------------------------------------
def prep_grace(geometry, mascon_id_grid):
    import netCDF4

    print("=== GRACE OBP ===")
    with netCDF4.Dataset(require_file(GRACE_MASCON_NC, "GRACE mascon NetCDF")) as nc:
        days = np.asarray(nc["time"][:])                    # days since 2002-01-01
        lwe = np.ma.filled(nc["lwe_thickness"][:], np.nan) / 100.0   # [t, lat, lon] m
        gad = np.ma.filled(nc["GAD"][:], np.nan) / 100.0
        lat_grid = np.asarray(nc["lat"][:])
        rho = ewh_density(nc, "GRACE")

    print(f"  EWH density {rho:g} kg m^-3 (from the granule) "
          f"-> {GRAVITY * rho:.1f} Pa per equivalent-water m")
    obp = shift_lon_halves(lwe, axis=2) * (GRAVITY * rho)
    gad = shift_lon_halves(gad, axis=2) * (GRAVITY * rho)

    # drop polar rows outside the mascon band
    polar = (lat_grid < -75) | (lat_grid > 64.5)
    obp[:, polar, :] = np.nan
    gad[:, polar, :] = np.nan

    # uniform 0.5-deg grid: cell area is proportional to cos(latitude)
    weights = np.repeat(np.cos(np.deg2rad(lat_grid))[:, None],
                        mascon_id_grid.shape[1], axis=1)
    averager = MasconAverager.from_mascon_ids(mascon_id_grid, geometry.mascon_ids,
                                              weights=weights)
    obp_mascon = averager(obp)   # [t, n_mascon]
    gad_mascon = averager(gad)

    # strictly monthly time axis: first GRACE sample, then the 16th of each month
    from scipy.interpolate import interp1d

    t0 = np.datetime64("2002-01-01")
    t_samples = t0 + days.astype("timedelta64[D]")
    # End the monthly axis at the granule's last available month.
    last_month = t_samples[-1].astype("datetime64[M]")
    months = np.arange(np.datetime64("2002-05"), last_month + 1,
                       np.timedelta64(1, "M"))
    monthly = np.array([np.datetime64(str(m) + "-16") for m in months])
    t_monthly = np.concatenate([[t_samples[0]], monthly])

    x = (t_samples - t0) / np.timedelta64(1, "D")
    x_new = (t_monthly - t0) / np.timedelta64(1, "D")
    # Fill missing GRACE months by linear interpolation before filtering.
    # Gap-adjacent values inherit information from the interpolated segment;
    # this step does not quantify uncertainty in the missing observations.
    obp_monthly = interp1d(x, obp_mascon, axis=0, bounds_error=False,
                           fill_value=np.nan)(x_new)
    gad_monthly = interp1d(x, gad_mascon, axis=0, bounds_error=False,
                           fill_value=np.nan)(x_new)
    print(f"  {obp_monthly.shape[0]} monthly samples "
          f"({t_monthly[0]} .. {t_monthly[-1]})")

    # Re-reference each mascon to the 2004-2009 mean after resampling, which
    # can otherwise introduce a small offset. Apply the same convention to
    # the default and alternative GRACE products.
    lbl = normalize_month_axis(t_monthly, obp_monthly.shape[0], "obp_GRACE")
    base = baseline_mask(lbl, "obp_GRACE")
    obp_monthly = obp_monthly - np.nanmean(obp_monthly[base], axis=0)
    # GAD shares the OBP baseline so their difference remains an anomaly.
    gad_monthly = gad_monthly - np.nanmean(gad_monthly[base], axis=0)

    atlso_lpf_save(
        "obp_GRACE", obp_monthly, t_monthly, geometry,
        OUT_DIR / "obp_GRACE.npz",
        demean=True, extra_fields={"GAD_GRACE": gad_monthly},
        metadata={
            "anomaly": np.bool_(True),
            "baseline_period": np.asarray(BASELINE_YEARS, dtype=np.int16),
            "baseline_spec": np.str_(
                "JPL RL06.3 source anomaly (native 2004.000-2009.999 static "
                "field); explicit 2004-01..2009-12 per-mascon re-referencing "
                "applied to OBP and GAD; Atlantic-Southern Ocean monthly spatial mean "
                "subsequently removed"),
            "source_product": np.str_(Path(GRACE_MASCON_NC).name),
        },
    )


# ---------------------------------------------------------------------------
# GRACE variant OBP products (CSR mascons, JPL/CSR SH ocean grids)
# ---------------------------------------------------------------------------
def _load_csr_cm(nc_path=None, tag="GRACE_CSR"):
    """Block-average CSR mascons to 0.5 degrees and check the EWH density."""
    import netCDF4

    from neurmoc.config import CSR_MASCON_NC

    nc_path = nc_path or CSR_MASCON_NC
    with netCDF4.Dataset(require_file(nc_path, "CSR mascon NetCDF")) as nc:
        units = str(getattr(nc["lwe_thickness"], "units",
                            getattr(nc["lwe_thickness"], "Units", ""))).strip()
        if "cm" not in units:
            raise ValueError(f"unexpected CSR lwe units {units!r}")
        days = np.asarray(nc["time"][:], dtype=float)
        lwe = np.ma.filled(nc["lwe_thickness"][:], np.nan)
        # CSR mascons use seawater-equivalent thickness (1025 kg m^-3).
        rho = ewh_density(nc, tag)
    nt = lwe.shape[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice")
        lwe = np.nanmean(lwe.reshape(nt, 360, 2, 720, 2), axis=(2, 4))
    return lwe, days, Path(nc_path).name, rho


def _load_gsfc_cm():
    """Load GSFC 0.5-degree ocean mascons and mask non-ocean values."""
    import netCDF4

    with netCDF4.Dataset(
        require_file(GSFC_MASCON_NC, "GSFC mascon NetCDF")
    ) as nc:
        units = str(getattr(nc["lwe_thickness"], "units",
                            getattr(nc["lwe_thickness"], "Units", ""))).strip()
        if "cm" not in units:
            raise ValueError(f"unexpected GSFC lwe units {units!r}")
        lat = np.asarray(nc["lat"][:], dtype=float)
        lon = np.asarray(nc["lon"][:], dtype=float)
        days = np.asarray(nc["time"][:], dtype=float)
        lwe = np.ma.filled(nc["lwe_thickness"][:], np.nan)
        land = np.asarray(nc["land_mask"][:]) > 0.5
        rho = ewh_density(nc, "GRACE_GSFC")
    # Confirm the native GSFC grid matches the expected 0.5-degree grid.
    expected_lat = np.arange(-89.75, 90.0, 0.5)
    expected_lon = np.arange(0.25, 360.0, 0.5)
    if not (np.allclose(lat, expected_lat) and np.allclose(lon, expected_lon)):
        raise ValueError(
            "GSFC granule is not on the production 0.5-deg grid "
            f"(lat {lat[0]}..{lat[-1]}, lon {lon[0]}..{lon[-1]})"
        )
    if land.shape != lwe.shape[1:]:
        raise ValueError("GSFC land_mask does not match the field grid")
    lwe[:, land] = np.nan
    return lwe, days, Path(GSFC_MASCON_NC).name, rho


def _grace_monthly(mascon_series: np.ndarray, days: np.ndarray):
    """Interpolate solutions to monthly 16th dates, clamping the last month."""
    from scipy.interpolate import interp1d

    t0 = np.datetime64("2002-01-01")
    t_samples = t0 + np.asarray(days).astype("timedelta64[D]")
    last_month = t_samples[-1].astype("datetime64[M]")
    months = np.arange(np.datetime64("2002-05"), last_month + 1,
                       np.timedelta64(1, "M"))
    monthly = np.array([np.datetime64(str(m) + "-16") for m in months])
    t_monthly = np.concatenate([[t_samples[0]], monthly])
    x = (t_samples - t0) / np.timedelta64(1, "D")
    x_new = (t_monthly - t0) / np.timedelta64(1, "D")
    out = interp1d(x, mascon_series, axis=0, bounds_error=False,
                   fill_value=(mascon_series[0], mascon_series[-1]))(x_new)
    return out, t_monthly


def prep_grace_variants(geometry, mascon_id_grid, selected_tags):
    """obp_GRACE_<tag>.npz for each alternative GRACE solution.

    Same treatment as prep_grace end to end: cm -> Pa, longitude recenter,
    polar band, cos-lat mascon weights, monthly-16th axis, Atl+SO
    reduction with per-month basin demeaning, 2-year LPF. All products are
    natively ~2004-2009 anomalies on the mascons' GAD-restored full-OBP
    convention."""
    print("=== GRACE variant OBP products ===")
    lat_grid = np.arange(-89.75, 90.0, 0.5)
    weights = np.repeat(np.cos(np.deg2rad(lat_grid))[:, None],
                        mascon_id_grid.shape[1], axis=1)
    averager = MasconAverager.from_mascon_ids(mascon_id_grid,
                                              geometry.mascon_ids,
                                              weights=weights)
    polar = (lat_grid < -75) | (lat_grid > 64.5)
    for tag, loader, spec in [
        ("GRACE_CSR", _load_csr_cm,
         "CSR RL0603 source anomaly (native ~2004-2009 static field); "
         "explicit 2004-01..2009-12 per-mascon re-referencing applied; "
         "Atlantic-Southern Ocean monthly spatial mean subsequently removed"),
        ("GRACE_GSFC", _load_gsfc_cm,
         "GSFC RL06 v2.0 OBP mascons, ICE6G-D GIA removed and GAD "
         "restored over ocean pixels (native 2004.000-2009.999 static "
         "field, i.e. already the pipeline window); explicit "
         "2004-01..2009-12 per-mascon re-referencing applied; Atlantic "
         "monthly spatial mean subsequently removed"),
    ]:
        if tag not in selected_tags:
            continue
        lwe_cm, days, source, rho = loader()
        if rho != EWH_DENSITY_EXPECTED[tag]:
            raise RuntimeError(
                f"{tag}: loader returned EWH density {rho} kg m^-3, expected "
                f"{EWH_DENSITY_EXPECTED[tag]}")
        obp = shift_lon_halves(lwe_cm / 100.0, axis=2) * (GRAVITY * rho)
        obp[:, polar, :] = np.nan
        obp_monthly, t_monthly = _grace_monthly(averager(obp), days)
        # Re-reference all GRACE variants to the same 2004-2009 baseline.
        lbl = normalize_month_axis(t_monthly, obp_monthly.shape[0], tag)
        obp_monthly = obp_monthly - np.nanmean(
            obp_monthly[baseline_mask(lbl, tag)], axis=0)
        print(f"  {tag}: {obp_monthly.shape[0]} monthly samples "
              f"({t_monthly[0]} .. {t_monthly[-1]}); EWH density "
              f"{rho:g} kg m^-3 -> {GRAVITY * rho:.1f} Pa per m")
        atlso_lpf_save(
            f"obp_{tag}", obp_monthly, t_monthly, geometry,
            OUT_DIR / f"obp_{tag}.npz", demean=True,
            metadata={
                "anomaly": np.bool_(True),
                "baseline_period": np.asarray(BASELINE_YEARS, dtype=np.int16),
                "baseline_spec": np.str_(spec),
                "source_product": np.str_(source),
            },
        )


# ---------------------------------------------------------------------------
# DUACS absolute dynamic topography (SSH)
# ---------------------------------------------------------------------------
def duacs_monthly_means():
    """Aggregate daily DUACS files to monthly means (cached as .npz).

    The cache records the daily chunk files it was built from and rebuilds
    itself when that set changes (e.g. a new chunk is downloaded).
    Incomplete trailing months (a chunk ending mid-month) are dropped: a
    monthly mean over a partial month would be biased."""
    import netCDF4

    files = sorted(require_dir(DUACS_DAILY_DIR, "DUACS daily data").glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No DUACS NetCDF files in {DUACS_DAILY_DIR}")
    file_names = np.array([p.name for p in files])

    cache = ensure_dir(DUACS_CACHE_DIR) / "DUACS_monthly_means.npz"
    if cache.is_file():
        with np.load(cache) as fh:
            if ("source_files" in fh.files
                    and np.array_equal(np.asarray(fh["source_files"]),
                                       file_names)):
                return fh["adt_monthly"], fh["months"], fh["lon"], fh["lat"]
        print("  daily chunk set changed - rebuilding the monthly-mean cache")

    sums, counts, day_counts = {}, {}, {}
    lon = lat = None
    for path in files:
        print("  aggregating", path.name)
        with netCDF4.Dataset(path) as nc:
            days = np.asarray(nc["time"][:])  # days since 1950-01-01
            if lon is None:
                lon = np.asarray(nc["longitude"][:])
                lat = np.asarray(nc["latitude"][:])
            dates = np.datetime64("1950-01-01") + days.astype("timedelta64[D]")
            month_id = np.datetime_as_string(dates, unit="M")
            # one month per read: a whole multi-year chunk unpacks to
            # >14 GiB of float64 and does not fit in memory
            for m in np.unique(month_id):
                sel = np.flatnonzero(month_id == m)
                if m in sums:
                    raise RuntimeError(f"Month {m} appears in multiple files")
                if not np.array_equal(sel, np.arange(sel[0], sel[-1] + 1)):
                    raise RuntimeError(
                        f"{path.name}: days of {m} are not contiguous")
                adt = np.ma.filled(nc["adt"][sel[0]:sel[-1] + 1], np.nan)
                sums[m] = np.nansum(adt, axis=0)
                counts[m] = np.isfinite(adt).sum(axis=0)
                day_counts[m] = int(sel.size)

    def calendar_days(m: str) -> int:
        m64 = np.datetime64(m, "M")
        span = (m64 + 1).astype("datetime64[D]") - m64.astype("datetime64[D]")
        return int(span / np.timedelta64(1, "D"))

    months = np.array(sorted(sums))
    complete = np.array([day_counts[m] >= calendar_days(m) for m in months])
    if not complete.all():
        dropped = [f"{m} ({day_counts[m]}/{calendar_days(m)} days)"
                   for m in months[~complete]]
        print("  dropping incomplete months:", ", ".join(dropped))
        months = months[complete]

    adt_monthly = np.stack([
        np.divide(sums[m], counts[m], out=np.full(sums[m].shape, np.nan),
                  where=counts[m] > 0)
        for m in months
    ])
    np.savez(cache, adt_monthly=adt_monthly, months=months, lon=lon, lat=lat,
             source_files=file_names)
    print("  cached", cache)
    return adt_monthly, months, lon, lat


def prep_duacs(geometry):
    print("=== DUACS SSH ===")
    adt_monthly, months, lon, lat = duacs_monthly_means()
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    averager = MasconAverager(lon_grid, lat_grid, geometry,
                              weights=cell_area_weights(lon_grid, lat_grid))
    adt_mascon = averager(adt_monthly)  # [t, n_mascon]

    time_month = normalize_month_axis(months, adt_mascon.shape[0], "DUACS")
    baseline = baseline_mask(time_month, "DUACS")
    ssh_baseline = np.nanmean(adt_mascon[baseline], axis=0)
    ssh_mascon = adt_mascon - ssh_baseline

    # near Antarctica some months are ice-covered: fill with the local time mean
    col_mean = np.nanmean(ssh_mascon, axis=0)
    nan_mask = np.isnan(ssh_mascon)
    ssh_filled = ssh_mascon.copy()
    ssh_filled[nan_mask] = np.broadcast_to(col_mean, ssh_mascon.shape)[nan_mask]

    # Sea-ice sensitivity test: ice-blanked months set to zero anomaly.
    ssh_zero_ice = ssh_mascon.copy()
    ssh_zero_ice[nan_mask] = 0.0

    base_metadata = {
        "anomaly": np.bool_(True),
        "baseline_period": np.asarray(BASELINE_YEARS, dtype=np.int16),
        "source_product": np.str_("DUACS daily ADT"),
    }
    #: Report missing-data share over the network's Atlantic/Southern domain.
    in_domain = geometry.basin_id == 1
    domain_nan = nan_mask[:, in_domain]
    print(f"  ice-blanked mascon-months: {int(domain_nan.sum())} over "
          f"{int(domain_nan.any(axis=0).sum())} mascons "
          f"({100 * domain_nan.mean():.2f}% of the Atlantic/Southern Ocean "
          f"input domain; {int(nan_mask.sum())} over the full mascon grid)")

    products = [
        ("ssh_DUACS", ssh_filled,
         "DUACS own per-mascon mean, 2004-01..2009-12; ice-blanked "
         "mascon-months filled with the local full-record mean anomaly; "
         "Atlantic-Southern Ocean monthly spatial mean subsequently removed"),
        ("ssh_ZeroIce", ssh_zero_ice,
         "DUACS own per-mascon mean, 2004-01..2009-12; SEA-ICE SENSITIVITY: "
         "ice-blanked mascon-months set to zero anomaly instead of the local "
         "full-record mean; Atlantic-Southern Ocean monthly spatial mean subsequently removed"),
    ]

    for name, values, spec in products:
        atlso_lpf_save(
            name, values, time_month, geometry,
            OUT_DIR / f"{name}.npz", demean=True,
            static_fields={"ssh_2004_2009_mean": ssh_baseline},
            metadata={**base_metadata, "baseline_spec": np.str_(spec)},
        )


# ---------------------------------------------------------------------------
# CCMP 10-m zonal wind
# ---------------------------------------------------------------------------
def prep_ccmp(geometry):
    import netCDF4

    print("=== CCMP wind ===")
    files = sorted(require_dir(CCMP_RAW_DIR, "CCMP data").glob(
        "CCMP_Wind_Analysis_*_monthly_mean_V03.1_L4.nc"))
    if not files:
        raise FileNotFoundError(f"No CCMP monthly files in {CCMP_RAW_DIR}")

    u10_list, hours = [], []
    lon = lat = None
    for path in files:
        with netCDF4.Dataset(path) as nc:
            u10_list.append(np.ma.filled(nc["u"][:], np.nan))  # [1, lat, lon]
            hours.append(np.asarray(nc["time"][:]).ravel())
            if lon is None:
                lon = np.asarray(nc["longitude"][:])
                lat = np.asarray(nc["latitude"][:])
    u10 = np.concatenate(u10_list, axis=0)
    hours = np.concatenate(hours)
    order = np.argsort(hours)
    u10, hours = u10[order], hours[order]
    dates = np.datetime64("1987-01-01") + hours.astype("timedelta64[h]")

    lon = lon.copy()
    lon[lon > 180] -= 360
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    averager = MasconAverager(lon_grid, lat_grid, geometry,
                              weights=cell_area_weights(lon_grid, lat_grid))
    u10_mascon = averager(u10)

    time_month = normalize_month_axis(dates, u10_mascon.shape[0], "CCMP")
    keep = time_month >= np.datetime64("2002-04", "M")
    u10_mascon = u10_mascon[keep]
    time_month = time_month[keep]
    wind_baseline_sel = baseline_mask(time_month, "CCMP")
    wind_baseline = np.nanmean(u10_mascon[wind_baseline_sel], axis=0)
    u10_mascon = u10_mascon - wind_baseline
    print(f"  {u10_mascon.shape[0]} monthly anomaly samples from "
          f"{time_month[0]}")

    atlso_lpf_save(
        "uas_CCMP", u10_mascon, time_month, geometry,
        OUT_DIR / "uas_CCMP.npz", demean=False, nan_to_zero=True,
        static_fields={"wind_2004_2009_mean": wind_baseline},
        metadata={
            "wind_source": np.str_("CCMP V03.1 monthly zonal 10-m wind"),
            "wind_anomaly": np.bool_(True),
            "wind_convention": np.str_("anomaly_2004_2009"),
            "baseline_period": np.asarray(BASELINE_YEARS, dtype=np.int16),
            "baseline_spec": np.str_(
                "CCMP own per-mascon mean, 2004-01..2009-12"),
            "source_product": np.str_("CCMP V03.1 L4 monthly mean"),
            "missing_value_fill": np.str_(
                "NaN anomalies filled with 0 before low-pass filtering"),
        },
    )


# ---------------------------------------------------------------------------
# NASA-SSH simple gridded SSHA (alternative SSH source; stage 14 SSH_SOURCE)
# ---------------------------------------------------------------------------
def prep_nasassh(geometry):
    import netCDF4

    print("=== NASA-SSH (simple grid v1.1) ===")
    files = sorted(require_dir(NASASSH_GRID_DIR, "NASA-SSH grids").glob(
        "NASA-SSH_alt_ref_simple_grid_v1_1_*.nc"))
    if not files:
        raise FileNotFoundError(f"No NASA-SSH NetCDF files in {NASASSH_GRID_DIR}")

    # Form monthly means weighted by each weekly granule's observation counts.
    sums, wsums, n_granules = {}, {}, {}
    lon = lat = None
    for path in files:
        with netCDF4.Dataset(path) as nc:
            tvar = nc["time"]
            t = np.ravel(netCDF4.num2date(
                tvar[:], tvar.units, only_use_cftime_datetimes=False))[0]
            ssha = np.ma.filled(nc["ssha"][:], np.nan).squeeze()  # [lat, lon]
            counts = np.ma.filled(nc["counts"][:], 0).squeeze().astype(float)
            if lon is None:
                lon = np.asarray(nc["longitude"][:])
                lat = np.asarray(nc["latitude"][:])
        m = f"{t.year:04d}-{t.month:02d}"
        w = np.where(np.isfinite(ssha) & (counts > 0), counts, 0.0)
        if m not in sums:
            sums[m] = np.zeros(ssha.shape)
            wsums[m] = np.zeros(ssha.shape)
            n_granules[m] = 0
        sums[m] += np.where(w > 0, ssha, 0.0) * w
        wsums[m] += w
        n_granules[m] += 1

    months = np.array(sorted(sums))
    # a first/last month covered by fewer than 3 weekly granules is a
    # partial month at a record edge: a mean over it would be biased
    complete = np.array([n_granules[m] >= 3 for m in months])
    if not complete.all():
        dropped = [f"{m} ({n_granules[m]} granules)" for m in months[~complete]]
        print("  dropping incomplete months:", ", ".join(dropped))
        months = months[complete]
    ssh_monthly = np.stack([
        np.divide(sums[m], wsums[m], out=np.full(sums[m].shape, np.nan),
                  where=wsums[m] > 0)
        for m in months
    ])

    lon = lon.copy()
    lon[lon > 180] -= 360
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    averager = MasconAverager(lon_grid, lat_grid, geometry,
                              weights=cell_area_weights(lon_grid, lat_grid))
    ssh_mascon = averager(ssh_monthly)  # [t, n_mascon]

    time_month = normalize_month_axis(months.astype("datetime64[M]"),
                                      ssh_mascon.shape[0], "NASA-SSH")
    keep = time_month >= np.datetime64("2002-04", "M")
    ssh_mascon, time_month = ssh_mascon[keep], time_month[keep]
    baseline = baseline_mask(time_month, "NASA-SSH")
    ssh_baseline = np.nanmean(ssh_mascon[baseline], axis=0)
    ssh_mascon = ssh_mascon - ssh_baseline

    # Fill ice-covered months with local means and wholly unobserved mascons
    # with zero anomalies.
    col_mean = np.nanmean(ssh_mascon, axis=0)
    n_empty = int(np.isnan(col_mean).sum())
    nan_mask = np.isnan(ssh_mascon)
    ssh_mascon[nan_mask] = np.broadcast_to(
        np.nan_to_num(col_mean, nan=0.0), ssh_mascon.shape)[nan_mask]
    print(f"  {ssh_mascon.shape[0]} monthly anomaly samples from "
          f"{time_month[0]} | mascons without any data (zero-filled): "
          f"{n_empty}")

    atlso_lpf_save(
        "ssh_NASASSH", ssh_mascon, time_month, geometry,
        OUT_DIR / "ssh_NASASSH.npz", demean=True,
        static_fields={"ssh_2004_2009_mean": ssh_baseline},
        metadata={
            "anomaly": np.bool_(True),
            "baseline_period": np.asarray(BASELINE_YEARS, dtype=np.int16),
            "baseline_spec": np.str_(
                "NASA-SSH own per-mascon mean, 2004-01..2009-12; "
                "Atlantic-Southern Ocean monthly spatial mean subsequently removed"),
            "source_product": np.str_(
                "NASA-SSH simple gridded SSHA v1.1 (weekly reference-"
                "mission grids, counts-weighted monthly means)"),
            "coverage_note": np.str_(
                f"observations end at ~71.2S; {n_empty} mascons south of "
                "the limit carry zero anomalies (DUACS-style local-time-"
                "mean fill elsewhere)"),
        },
    )


# ---------------------------------------------------------------------------
# ERA5 10-m zonal wind (alternative wind source; stage 14 USE_ERA5_WINDS)
# ---------------------------------------------------------------------------
def prep_era5(geometry):
    import netCDF4

    print("=== ERA5 wind ===")
    with netCDF4.Dataset(require_file(ERA5_WIND_NC, "ERA5 monthly wind")) as nc:
        tname = "valid_time" if "valid_time" in nc.variables else "time"
        tvar = nc[tname]
        stamps = netCDF4.num2date(tvar[:], tvar.units,
                                  only_use_cftime_datetimes=False)
        months_all = np.array(
            [np.datetime64(f"{d.year:04d}-{d.month:02d}", "M") for d in stamps])
        u10 = np.ma.filled(nc["u10"][:], np.nan)  # [t, lat, lon], monthly means
        lon = np.asarray(nc["longitude"][:])
        lat = np.asarray(nc["latitude"][:])
        # recent months come from the preliminary ERA5T stream (expver 0005)
        # and are replaced by final ERA5 in later downloads
        expver = (np.asarray(nc["expver"][:]).astype(str)
                  if "expver" in nc.variables else
                  np.full(months_all.size, "0001"))

    lon = lon.copy()
    lon[lon > 180] -= 360
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    averager = MasconAverager(lon_grid, lat_grid, geometry,
                              weights=cell_area_weights(lon_grid, lat_grid))
    u10_mascon = averager(u10)

    keep = months_all >= np.datetime64("2002-04", "M")
    u10_mascon, expver = u10_mascon[keep], expver[keep]
    time_month = normalize_month_axis(months_all[keep], u10_mascon.shape[0],
                                      "ERA5")
    wind_baseline_sel = baseline_mask(time_month, "ERA5")
    wind_baseline = np.nanmean(u10_mascon[wind_baseline_sel], axis=0)
    u10_mascon = u10_mascon - wind_baseline
    era5t = np.datetime_as_string(time_month[expver != "0001"], unit="M")
    print(f"  {u10_mascon.shape[0]} monthly anomaly samples from "
          f"{time_month[0]}"
          + (f" | preliminary ERA5T: {', '.join(era5t)}" if era5t.size else ""))

    atlso_lpf_save(
        "uas_ERA5", u10_mascon, time_month, geometry,
        OUT_DIR / "uas_ERA5.npz", demean=False, nan_to_zero=True,
        static_fields={"wind_2004_2009_mean": wind_baseline},
        metadata={
            "wind_source": np.str_("ERA5 monthly-mean zonal 10-m wind"),
            "wind_anomaly": np.bool_(True),
            "wind_convention": np.str_("anomaly_2004_2009"),
            "baseline_period": np.asarray(BASELINE_YEARS, dtype=np.int16),
            "baseline_spec": np.str_(
                "ERA5 own per-mascon mean, 2004-01..2009-12"),
            "source_product": np.str_(ERA5_WIND_NC.name),
            "preliminary_months": np.asarray(era5t, dtype="U"),
            "missing_value_fill": np.str_(
                "ERA5 is gap-free; nan-to-zero is a dormant guard"),
        },
    )


# ---------------------------------------------------------------------------
# Bulk-formula wind stress and stress curl
# ---------------------------------------------------------------------------
def _stress_mascon_rows(u10, v10, lat_native, lon_native, averager):
    """Compute stress and curl on the native grid before mascon averaging.

    Use monotonic native longitude for derivatives to avoid a seam jump.
    """
    tau_u, tau_v = bulk_wind_stress(u10, v10)
    curl = wind_stress_curl(tau_u, tau_v, lat_native, lon_native)
    return averager(tau_u), averager(curl)


def _save_stress_products(tag, tauu_mascon, curl_mascon, time_month, geometry,
                          source_product, wind_note):
    """Anomalize, filter and save the tauu/curltau pair for one wind source."""
    baseline_sel = baseline_mask(time_month, tag)
    products = (
        ("tauu", tauu_mascon, "Pa", "eastward bulk wind stress"),
        ("curltau", curl_mascon, "N m^-3", "bulk wind-stress curl"),
    )
    for slot, values, unit, what in products:
        baseline = np.nanmean(values[baseline_sel], axis=0)
        anomaly = values - baseline
        atlso_lpf_save(
            f"{slot}_{tag}", anomaly, time_month, geometry,
            OUT_DIR / f"{slot}_{tag}.npz", demean=False, nan_to_zero=True,
            static_fields={f"{slot}_2004_2009_mean": baseline},
            metadata={
                "wind_source": np.str_(f"{wind_note} -> {what} [{unit}]"),
                "wind_anomaly": np.bool_(True),
                "wind_convention": np.str_("anomaly_2004_2009"),
                "baseline_period": np.asarray(BASELINE_YEARS, dtype=np.int16),
                "baseline_spec": np.str_(
                    f"{tag} own per-mascon mean, 2004-01..2009-12"),
                "source_product": np.str_(source_product),
                # Record the stress formulation from the active constants.
                "stress_formulation": np.str_(
                    "bulk: tau = rho_a Cd(|U10|) |U10| U10, "
                    f"rho_a={BULK_AIR_DENSITY} kg m^-3, "
                    f"Cd = {BULK_DRAG_LAW} neutral 10-m; "
                    "|U10| is the full vector speed hypot(u10, v10)"),
                "stress_drag_law": np.str_(BULK_DRAG_LAW),
                "stress_is_derived_from_wind": np.bool_(True),
                "missing_value_fill": np.str_(
                    "NaN anomalies filled with 0 before low-pass filtering"),
            },
        )


def prep_ccmp_stress(geometry):
    import netCDF4

    print("=== CCMP bulk wind stress ===")
    files = sorted(require_dir(CCMP_RAW_DIR, "CCMP data").glob(
        "CCMP_Wind_Analysis_*_monthly_mean_V03.1_L4.nc"))
    if not files:
        raise FileNotFoundError(f"No CCMP monthly files in {CCMP_RAW_DIR}")

    # One month in memory at a time: the full u and v stacks together are
    # several GB before the mascon reduction throws almost all of it away.
    averager = lat_native = lon_native = None
    tauu_rows, curl_rows, hours = [], [], []
    for path in files:
        with netCDF4.Dataset(path) as nc:
            u10 = np.ma.filled(nc["u"][:], np.nan)   # [1, lat, lon]
            v10 = np.ma.filled(nc["v"][:], np.nan)
            hours.append(np.asarray(nc["time"][:]).ravel())
            if averager is None:
                lon_native = np.asarray(nc["longitude"][:])
                lat_native = np.asarray(nc["latitude"][:])
                lon_shifted = lon_native.copy()
                lon_shifted[lon_shifted > 180] -= 360
                lon_grid, lat_grid = np.meshgrid(lon_shifted, lat_native)
                averager = MasconAverager(
                    lon_grid, lat_grid, geometry,
                    weights=cell_area_weights(lon_grid, lat_grid))
        tauu_row, curl_row = _stress_mascon_rows(
            u10, v10, lat_native, lon_native, averager)
        tauu_rows.append(tauu_row)
        curl_rows.append(curl_row)

    tauu_mascon = np.concatenate(tauu_rows, axis=0)
    curl_mascon = np.concatenate(curl_rows, axis=0)
    hours = np.concatenate(hours)
    order = np.argsort(hours)
    tauu_mascon, curl_mascon = tauu_mascon[order], curl_mascon[order]
    dates = np.datetime64("1987-01-01") + hours[order].astype("timedelta64[h]")

    time_month = normalize_month_axis(dates, tauu_mascon.shape[0], "CCMP")
    keep = time_month >= np.datetime64("2002-04", "M")
    tauu_mascon, curl_mascon = tauu_mascon[keep], curl_mascon[keep]
    time_month = time_month[keep]
    print(f"  {tauu_mascon.shape[0]} monthly anomaly samples from "
          f"{time_month[0]}")

    _save_stress_products(
        "CCMP", tauu_mascon, curl_mascon, time_month, geometry,
        "CCMP V03.1 L4 monthly mean",
        "CCMP V03.1 monthly 10-m wind, bulk formula")


def prep_era5_stress(geometry):
    import netCDF4

    print("=== ERA5 bulk wind stress ===")
    with netCDF4.Dataset(require_file(ERA5_WIND_NC, "ERA5 monthly wind")) as nc:
        tname = "valid_time" if "valid_time" in nc.variables else "time"
        tvar = nc[tname]
        stamps = netCDF4.num2date(tvar[:], tvar.units,
                                  only_use_cftime_datetimes=False)
        months_all = np.array(
            [np.datetime64(f"{d.year:04d}-{d.month:02d}", "M") for d in stamps])
        lon_native = np.asarray(nc["longitude"][:])
        lat_native = np.asarray(nc["latitude"][:])
        expver = (np.asarray(nc["expver"][:]).astype(str)
                  if "expver" in nc.variables else
                  np.full(months_all.size, "0001"))

        lon_shifted = lon_native.copy()
        lon_shifted[lon_shifted > 180] -= 360
        lon_grid, lat_grid = np.meshgrid(lon_shifted, lat_native)
        averager = MasconAverager(
            lon_grid, lat_grid, geometry,
            weights=cell_area_weights(lon_grid, lat_grid))

        tauu_rows, curl_rows = [], []
        for t in range(months_all.size):
            u10 = np.ma.filled(nc["u10"][t:t + 1], np.nan)
            v10 = np.ma.filled(nc["v10"][t:t + 1], np.nan)
            tauu_row, curl_row = _stress_mascon_rows(
                u10, v10, lat_native, lon_native, averager)
            tauu_rows.append(tauu_row)
            curl_rows.append(curl_row)

    tauu_mascon = np.concatenate(tauu_rows, axis=0)
    curl_mascon = np.concatenate(curl_rows, axis=0)

    keep = months_all >= np.datetime64("2002-04", "M")
    tauu_mascon, curl_mascon = tauu_mascon[keep], curl_mascon[keep]
    expver = expver[keep]
    time_month = normalize_month_axis(months_all[keep], tauu_mascon.shape[0],
                                      "ERA5")
    era5t = np.datetime_as_string(time_month[expver != "0001"], unit="M")
    print(f"  {tauu_mascon.shape[0]} monthly anomaly samples from "
          f"{time_month[0]}"
          + (f" | preliminary ERA5T: {', '.join(era5t)}" if era5t.size else ""))

    _save_stress_products(
        "ERA5", tauu_mascon, curl_mascon, time_month, geometry,
        ERA5_WIND_NC.name, "ERA5 monthly-mean 10-m wind, bulk formula")


def qc_figure(name: str, unit: str, note: str, series_note: str,
              shade_grace_gap: bool) -> None:
    """Key properties of one satellite input, from its saved .npz:
    (a) time-mean map, (b) std of the 2-year-filtered series (the
    variability the network sees), (c) raw vs filtered series at the
    most variable mascon, on a calendar axis."""
    with np.load(OUT_DIR / f"{name}.npz") as fh:
        raw = fh[name]
        lpf = fh[f"{name}_LPF_ALL"]
        lon = fh[f"{name}_lon"]
        lat = fh[f"{name}_lat"]
        if "time_month" in fh:
            months = normalize_month_axis(fh["time_month"], raw.shape[0], name)
            labels = np.datetime_as_string(months, unit="M")
            t = np.asarray([int(v[:4]) + int(v[5:7]) / 12 for v in labels])
        else:
            t = T0_YEARS + np.arange(raw.shape[0]) / 12.0
    land_lon, land_lat, land = load_grace_land(GRACE_MASCON_NC)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8.2), constrained_layout=True)
    mean = raw.mean(axis=0)
    vmax_mean = np.nanpercentile(np.abs(mean), 98)
    std = lpf.std(axis=0)
    for ax, values, cmap, vmin, vmax, title in [
        (axes[0], mean, CMAP_DIVERGING, -vmax_mean, vmax_mean,
         f"{name}: time mean, {raw.shape[0]} months "
         f"({t[0]:.2f}-{t[-1]:.2f}){note}"),
        (axes[1], std, CMAP_AMPLITUDE, 0, np.nanpercentile(std, 98),
         f"{name}: std of the 2-year low-passed series"),
    ]:
        ax.pcolormesh(land_lon, land_lat, land,
                      cmap=plt.matplotlib.colors.ListedColormap(["#d9d9d9"]),
                      shading="nearest", rasterized=True)
        sc = ax.scatter(lon, lat, c=values, s=5, cmap=cmap, vmin=vmin,
                        vmax=vmax, linewidths=0)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-78, 66)
        ax.set_ylabel("Latitude")
        ax.set_title(title, loc="left")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.set_title(unit, fontsize=plt.rcParams["font.size"], pad=4)

    ax = axes[2]
    j = int(np.nanargmax(std))
    if shade_grace_gap:
        shade_gap(ax)
    ax.plot(t, raw[:, j], color="0.75", lw=0.5, label="monthly")
    ax.plot(t, lpf[:, j], color="#0072B2", lw=1.2, label="2-year low-pass")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Year")
    ax.set_ylabel(unit)
    ax.set_title(f"{name} @ ({lon[j]:.0f}\N{DEGREE SIGN}, {lat[j]:.0f}\N{DEGREE SIGN}) - {series_note}",
                 loc="left")
    ax.legend(ncol=2, loc="upper right")
    ax.tick_params(top=False, right=False)
    save_figure(fig, OUT_DIR / name, formats=("png",))


def main() -> None:
    ensure_dir(OUT_DIR)
    geometry, mascon_id_grid, _, _ = load_geometry()
    grace_variants = ["GRACE_CSR", "GRACE_GSFC"] if RUN_GRACE_VARIANTS else []
    if RUN_GRACE:
        prep_grace(geometry, mascon_id_grid)
    if RUN_DUACS:
        prep_duacs(geometry)
    if RUN_CCMP:
        prep_ccmp(geometry)
    if RUN_ERA5:
        prep_era5(geometry)
    if RUN_CCMP_STRESS:
        prep_ccmp_stress(geometry)
    if RUN_ERA5_STRESS:
        prep_era5_stress(geometry)
    if RUN_NASASSH:
        prep_nasassh(geometry)
    if grace_variants:
        prep_grace_variants(geometry, mascon_id_grid, grace_variants)

    if QC_PLOTS:
        apply_style()
        # Annotate quality-control plots with the relevant processing caveat.
        for name, unit, note, series_note, gap in [
            ("obp_GRACE", "Pa", " (anomaly vs 2004-2009, basin-demeaned)",
             "shaded: GRACE/GRACE-FO gap, linearly bridged; residual land "
             "leakage near ice sheets/major rivers despite CRI and the "
             "land-free mascon selection", True),
            ("ssh_DUACS", "m", " (anomaly vs 2004-2009, basin-demeaned)",
             "caveat: ice-covered months at high-latitude mascons are filled "
             "with the local time mean", False),
            ("ssh_NASASSH", "m", " (anomaly vs 2004-2009, basin-demeaned)",
             "caveat: no observations south of ~71.2S (mascons there carry "
             "zero anomalies); ice months filled with the local time mean",
             False),
            ("uas_CCMP", "m s$^{-1}$", " (anomaly vs 2004-2009)",
             "note: CCMP V03.1 NaNs (polar ice caps only) never cover a "
             "whole mascon - no filling occurs; nan-to-zero is a dormant "
             "guard", False),
            ("uas_ERA5", "m s$^{-1}$", " (anomaly vs 2004-2009)",
             "note: ERA5 is gap-free; the newest months are the "
             "preliminary ERA5T stream (see preliminary_months)", False),
        ]:
            if (OUT_DIR / f"{name}.npz").is_file():
                qc_figure(name, unit, note, series_note, gap)


if __name__ == "__main__":
    main()
