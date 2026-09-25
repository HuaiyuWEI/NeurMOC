"""CMIP6/ACCESS file discovery and loaders for intermediate data products.

Loaders accept .npz and MATLAB .mat (v7.3) products and return time-leading
NumPy arrays.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BASELINE_YEARS
from .io_utils import require_file

# Wind inputs: 10-m wind, pressure-level wind, zonal stress, and stress curl.
# Wind-stress curl is computed from tauu and tauv on the native atmospheric
# grid before mascon averaging. Stress and curl use separate output slots.
WIND_VARIABLES = {
    "uas": {"table": "Amon", "units": "m s$^{-1}$", "slot": "uas",
            "long_name": "eastward 10-m wind", "components": ("uas",)},
    "ua": {"table": "Amon", "units": "m s$^{-1}$", "slot": "uas",
           "long_name": "eastward wind at a pressure level",
           "components": ("ua",)},
    "tauu": {"table": "Amon", "units": "Pa", "slot": "tauu",
             "long_name": "eastward surface wind stress",
             "components": ("tauu",)},
    "curltau": {"table": "Amon", "units": "N m$^{-3}$", "slot": "curltau",
                "long_name": "vertical curl of the surface wind stress",
                "components": ("tauu", "tauv")},
}
# Variables requiring a pressure level.
WIND_LEVEL_REQUIRED = ("ua",)
# Pressure-level wind may be rescaled to 10-m-wind-equivalent anomalies.
WIND_CALIBRATABLE = ("ua",)


def parse_wind_var(spec: str) -> tuple[str, float | None]:
    """Validate and normalize a wind specification -> (name, level_pa)."""
    name, separator, level_text = spec.strip().lower().partition("@")
    if name in WIND_LEVEL_REQUIRED:
        if not (separator and level_text):
            raise ValueError(
                f"{name} needs a pressure level, e.g. {name}@100000")
        level = float(level_text)
        if not np.isfinite(level) or level <= 0:
            raise ValueError(
                f"Wind pressure must be positive and finite, got {level_text!r}")
        return name, level
    if name in WIND_VARIABLES and not separator:
        return name, None
    raise ValueError(
        "Wind must be one of "
        + ", ".join(f"{v}@<pressure_pa>" if v in WIND_LEVEL_REQUIRED else v
                    for v in WIND_VARIABLES)
        + f" (got {spec!r})")


def canonical_wind_var(spec: str) -> str:
    """Standard spelling of a wind specification, e.g. 'uas' or 'ua@85000'."""
    name, level = parse_wind_var(spec)
    return name if level is None else f"{name}@{level:g}"


def wind_output_slot(spec: str) -> str:
    """File tag for a wind input; uas and pressure-level ua share a slot."""
    name, _ = parse_wind_var(spec)
    return WIND_VARIABLES[name]["slot"]


# The default wind baseline has no filename suffix.
DEFAULT_WIND_SLOT = "uas"
# Suffixes excluded when locating the untagged baseline.
BASELINE_WIND_SUFFIXES = tuple(
    sorted({f"_{spec['slot']}" for spec in WIND_VARIABLES.values()
            if spec["slot"] != DEFAULT_WIND_SLOT})
)


def baseline_wind_suffix(spec: str) -> str:
    """Baseline filename suffix, empty for the default wind slot."""
    slot = wind_output_slot(spec)
    return "" if slot == DEFAULT_WIND_SLOT else f"_{slot}"


def wind_components(spec: str) -> tuple[str, ...]:
    """Raw CMIP variables required for a wind input."""
    name, _ = parse_wind_var(spec)
    return WIND_VARIABLES[name]["components"]


def wind_is_derived(spec: str) -> bool:
    """True when the input is computed from more than one raw variable."""
    return len(wind_components(spec)) > 1


EARTH_RADIUS_M = 6.371e6


#: Air density used by the bulk stress formula, kg m^-3.
BULK_AIR_DENSITY = 1.22


# RAPID uses the Smith (1980) drag law for its Ekman transport estimate.
BULK_DRAG_LAW = "smith1980"


def bulk_drag_coefficient(speed, law=None):
    """Neutral 10-m drag coefficient, as an array matching `speed`.

    smith1980 (Smith 1980, the RAPID convention):
        1000*Cd = 1.0                below 7.5 m/s
        1000*Cd = 0.61 + 0.063*|U|   at and above it
    largepond1981 (Large and Pond 1981):
        1000*Cd = 1.2                below 11 m/s
        1000*Cd = 0.49 + 0.065*|U|   at and above it

    The Smith (1980) formula has a discontinuity at 7.5 m/s; it is retained.
    """
    law = (law or BULK_DRAG_LAW).lower()
    speed = np.asarray(speed, dtype=float)
    if law == "smith1980":
        return np.where(speed < 7.5, 1.0e-3, (0.61 + 0.063 * speed) * 1e-3)
    if law == "largepond1981":
        return np.where(speed < 11.0, 1.2e-3, (0.49 + 0.065 * speed) * 1e-3)
    raise ValueError(
        f"unknown drag law {law!r}; use 'smith1980' or 'largepond1981'")


def bulk_wind_stress(u10, v10, air_density=BULK_AIR_DENSITY, law=None):
    """Surface wind stress from 10-m winds: tau = rho_a Cd(|U|) |U| U.

    Returns `(tau_u, tau_v)` in Pa. The speed `|U|` includes both horizontal
    components. Stress computed from monthly-mean winds omits submonthly
    variability and is not an independent stress estimate.
    """
    u10 = np.asarray(u10, dtype=float)
    v10 = np.asarray(v10, dtype=float)
    if u10.shape != v10.shape:
        raise ValueError(
            f"u10 {u10.shape} and v10 {v10.shape} must share a grid")
    speed = np.hypot(u10, v10)
    factor = air_density * bulk_drag_coefficient(speed, law) * speed
    return factor * u10, factor * v10


def wind_stress_curl(tauu, tauv, lat, lon):
    """Vertical wind-stress curl on a regular lat-lon grid, in N m^-3.

    curl_z(tau) = 1/(a cos(phi)) * [ d(tau_v)/d(lambda)
                                     - d(tau_u cos(phi))/d(phi) ]

    `tauu`/`tauv` are `[time, lat, lon]`; `lat`/`lon` are 1-D degrees.
    Longitude is periodic; rows poleward of 89 degrees are returned as NaN.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if tauu.shape != tauv.shape:
        raise ValueError(
            f"tauu {tauu.shape} and tauv {tauv.shape} must share a grid")
    if tauu.shape[-2:] != (lat.size, lon.size):
        raise ValueError(
            f"stress field {tauu.shape[-2:]} does not match the "
            f"({lat.size}, {lon.size}) lat/lon grid")

    phi = np.deg2rad(lat)[:, None]
    cos_phi = np.cos(phi)
    lam = np.deg2rad(lon)

    # Wrap longitude before differencing at the seam.
    d_lam = np.gradient(np.unwrap(lam))
    padded = np.concatenate([tauv[..., -1:], tauv, tauv[..., :1]], axis=-1)
    dtauv_dlam = (padded[..., 2:] - padded[..., :-2]) / (2.0 * d_lam)

    dtauucos_dphi = np.gradient(tauu * cos_phi, np.deg2rad(lat), axis=-2)

    with np.errstate(invalid="ignore", divide="ignore"):
        curl = (dtauv_dlam - dtauucos_dphi) / (EARTH_RADIUS_M * cos_phi)
    curl[..., np.abs(lat) > 89.0, :] = np.nan
    return curl


def curl_partner_path(tauu_path) -> Path:
    """Locate the tauv file with the same model, member, grid, and dates."""
    path = Path(tauu_path)
    if not path.name.startswith("tauu_"):
        raise ValueError(f"{path.name} is not a tauu file")
    partner = path.with_name("tauv_" + path.name[len("tauu_"):])
    if not partner.is_file():
        raise FileNotFoundError(
            f"wind-stress curl needs {partner.name} beside {path.name}; "
            "download the matching tauv for this member")
    return partner


def _interp_axis(field, src, dst, axis, period=None):
    """Linear interpolation of `field` from `src` to `dst` along `axis`.

    `period` wraps the source coordinate (longitude), so the seam is
    interpolated like any other point instead of being clamped. Targets
    outside a non-periodic source range clamp to the edge value.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if period is not None:
        # Extend both ends to interpolate across the periodic seam.
        src = np.concatenate([src[-1:] - period, src, src[:1] + period])
        field = np.concatenate(
            [np.take(field, [-1], axis=axis), field,
             np.take(field, [0], axis=axis)], axis=axis)
    hi = np.clip(np.searchsorted(src, dst), 1, src.size - 1)
    lo = hi - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = (dst - src[lo]) / (src[hi] - src[lo])
    weight = np.clip(weight, 0.0, 1.0)
    shape = [1] * field.ndim
    shape[axis] = dst.size
    f_lo = np.take(field, lo, axis=axis)
    f_hi = np.take(field, hi, axis=axis)
    return f_lo + (f_hi - f_lo) * weight.reshape(shape)


def colocate_stress(tauv, lat_v, lon_v, lat_u, lon_u):
    """Put tauv on the tauu grid when a model staggers the two.

    Uses bilinear interpolation with periodic longitude when grids differ.
    """
    lat_v, lon_v = np.asarray(lat_v, float), np.asarray(lon_v, float)
    lat_u, lon_u = np.asarray(lat_u, float), np.asarray(lon_u, float)
    if lat_v.size == lat_u.size and lon_v.size == lon_u.size \
            and np.allclose(lat_v, lat_u) and np.allclose(lon_v, lon_u):
        return tauv
    # Reject grid mismatches larger than a one-cell stagger.
    if abs(lat_v.size - lat_u.size) > 1 or lon_v.size != lon_u.size:
        raise RuntimeError(
            f"tauu grid ({lat_u.size}x{lon_u.size}) and tauv grid "
            f"({lat_v.size}x{lon_v.size}) are not a half-cell stagger of "
            "one another; these files do not belong together")
    out = _interp_axis(tauv, lon_v, lon_u, axis=-1, period=360.0)
    return _interp_axis(out, lat_v, lat_u, axis=-2)


def read_stress_curl(tauu_path, tauv_path, time_selection=slice(None)):
    """Native-grid wind-stress curl `[time, lat, lon]` for a time slice.

    Validates their time axes and computes curl before mascon averaging.
    """
    import netCDF4

    with netCDF4.Dataset(tauu_path) as ncu, netCDF4.Dataset(tauv_path) as ncv:
        for nc, var, path in ((ncu, "tauu", tauu_path), (ncv, "tauv", tauv_path)):
            if nc[var].dimensions[0] != "time":
                raise ValueError(
                    f"Expected time first in {var} of {Path(path).name}, "
                    f"got {nc[var].dimensions}")
        tu, tv = np.asarray(ncu["time"][:]), np.asarray(ncv["time"][:])
        if tu.shape != tv.shape or not np.allclose(tu, tv):
            raise RuntimeError(
                f"{Path(tauu_path).name} and {Path(tauv_path).name} do not "
                "share a time axis")
        for name in ("lat", "lon"):
            for nc, path in ((ncu, tauu_path), (ncv, tauv_path)):
                if name not in nc.variables:
                    raise KeyError(
                        f"{name} missing from {Path(path).name}; the curl "
                        "needs a regular 1-D atmosphere grid")
        lat = np.asarray(ncu["lat"][:], dtype=float)
        lon = np.asarray(ncu["lon"][:], dtype=float)
        lat_v = np.asarray(ncv["lat"][:], dtype=float)
        lon_v = np.asarray(ncv["lon"][:], dtype=float)
        tauu = np.ma.filled(ncu["tauu"][time_selection], np.nan)
        tauv = np.ma.filled(ncv["tauv"][time_selection], np.nan)
    # Co-locate staggered components before differencing.
    tauv = colocate_stress(tauv, lat_v, lon_v, lat, lon)
    return wind_stress_curl(tauu, tauv, lat, lon)


def wind_units(spec: str) -> str:
    """Matplotlib-ready unit label, for QC plots and axis labels."""
    name, _ = parse_wind_var(spec)
    return WIND_VARIABLES[name]["units"]


def wind_table(spec: str) -> str:
    """CMIP table the variable is published in."""
    name, _ = parse_wind_var(spec)
    return WIND_VARIABLES[name]["table"]


def parse_realizations(spec: str) -> list[int]:
    """'1-35' -> [1..35]; '7' -> [7]; '1,6-10' -> [1, 6, 7, 8, 9, 10]."""
    numbers: set[int] = set()
    for part in spec.split(","):
        low, _, high = part.strip().partition("-")
        numbers.update(range(int(low), int(high or low) + 1))
    return sorted(numbers)


def detect_model(data_dir: Path | str) -> str:
    """Infer the CMIP source_id from the .nc filenames in `data_dir`.

    CMIP filenames follow <var>_<table>_<source_id>_<experiment>_<variant>_...;
    the folder must contain files of exactly one source_id.
    """
    models = set()
    for path in Path(data_dir).glob("*.nc"):
        parts = path.name.split("_")
        if len(parts) >= 6 and re.fullmatch(r"r\d+i\d+p\d+f\d+", parts[4]):
            models.add(parts[2])
    if len(models) == 1:
        return models.pop()
    if not models:
        raise FileNotFoundError(
            f"No CMIP .nc files in {data_dir} to detect the model from; "
            "set MODEL / pass -m explicitly")
    raise RuntimeError(
        f"Several models in {data_dir} ({sorted(models)}); "
        "set MODEL / pass -m explicitly")


def load_areacello(source_id: str) -> np.ndarray:
    """True ocean cell areas (m^2) for `source_id`, from raw/cmip6/areacello.

    Files are the ESGF `areacello_Ofx_*` products on the native `gn` grid
    (masked over land). Matched by source_id anywhere in the filename, so
    nonstandard orderings (areacello_Ofx_historical_NorESM2-LM_...) work.
    """
    import netCDF4

    from .config import CMIP_RAW_ROOT

    area_dir = CMIP_RAW_ROOT / "areacello"
    matches = sorted(p for p in area_dir.glob("areacello_*.nc")
                     if source_id in p.name)
    if not matches:
        raise FileNotFoundError(
            f"no areacello file for {source_id!r} in {area_dir} - download "
            "the Ofx areacello for this model (weighted mascon averaging "
            "requires it for ocean variables)")
    with netCDF4.Dataset(matches[0]) as nc:
        return np.ma.filled(nc["areacello"][:], np.nan)


def make_mascon_averager(sample_file, geometry, areacello_model: str | None = None):
    """Build a mascon averager for regular or curvilinear NetCDF grids.

    `areacello_model` selects true ocean-cell areas; otherwise analytic
    latitude-based areas are used.
    """
    import netCDF4

    from .grids import MasconAverager, cell_area_weights

    with netCDF4.Dataset(sample_file) as nc:
        for lat_name, lon_name in [("latitude", "longitude"), ("lat", "lon")]:
            if lat_name in nc.variables and lon_name in nc.variables:
                lat = np.asarray(nc[lat_name][:])
                lon = np.asarray(nc[lon_name][:]).astype(float)
                break
        else:
            raise KeyError(f"No lat/lon coordinates found in {sample_file}")

    lon[lon > 180] -= 360
    if lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    if areacello_model is not None:
        weights = load_areacello(areacello_model)
        if weights.shape != lat.shape:
            raise ValueError(
                f"{sample_file}: grid {lat.shape} does not match the "
                f"{areacello_model} areacello grid {weights.shape}; the "
                "variable is not on the native ocean grid")
    else:
        weights = cell_area_weights(lon, lat)
    return MasconAverager(lon, lat, geometry, weights=weights)


def find_realization_files(data_dir: Path | str, pattern: str) -> dict[int, list[Path]]:
    """Map realization number -> sorted list of files matching `pattern`.

    The realization is parsed from the CMIP `_r<N>i...` filename tag.
    """
    files = sorted(Path(data_dir).glob(pattern))
    by_realization: dict[int, list[Path]] = {}
    for path in files:
        match = re.search(r"_r(\d+)i", path.name)
        if match:
            by_realization.setdefault(int(match.group(1)), []).append(path)
    return by_realization


def realization_files(data_dir, pattern, realization: int, allow_multiple=True) -> list[Path]:
    """Files of one realization; raises if missing (or ambiguous when not allowed)."""
    found = find_realization_files(data_dir, pattern).get(realization, [])
    if not found:
        raise FileNotFoundError(
            f"No file matching {pattern} for realization r{realization} in {data_dir}"
        )
    if not allow_multiple and len(found) > 1:
        raise RuntimeError(f"Multiple files for realization r{realization}: {found}")
    return found


# ---------------------------------------------------------------------------
# Mascon-averaged predictor files
# ---------------------------------------------------------------------------
def load_mascon_realization(data_dir: Path | str, var_tag: str, realization: int) -> dict:
    """Load one `Mascon_<version>_<VAR>_r<N>.npz` file.

    Returns a dict with `data` `[time, n_mascon]`, `basin_id`, `lat`, `lon`
    (mascon centers).
    """
    data_dir = Path(data_dir)
    from .config import MASCON_VERSION

    stem = data_dir / f"Mascon_{MASCON_VERSION}_{var_tag}_r{realization}"

    with np.load(stem.with_suffix(".npz")) as fh:
        metadata_keys = (
            "baseline_spec", "wind_source", "wind_convention",
            "wind_anomaly", "wind_calibration", "wind_selected_pressure_pa",
            "baseline_period", "baseline_years",
            "source_model", "source_experiment", "time_units", "time_calendar",
        )
        return {
            "data": fh["Input_vars_mascon"],
            "basin_id": fh["Basin_id"].squeeze(),
            "lat": fh["lat_mascon_center"].squeeze(),
            "lon": fh["lon_mascon_center"].squeeze(),
            "time_month": fh["time_month"] if "time_month" in fh.files else None,
            "provenance": {key: str(fh[key])
                           for key in metadata_keys
                           if key in fh.files},
        }


# ---------------------------------------------------------------------------
# Interpolated MOC target files
# ---------------------------------------------------------------------------
def load_moc_realization(data_dir: Path | str, realization: int) -> dict:
    """Load one `FullDepth_ASMOC_interp_gr_r<N>.npz` file.

    Returns `psi` `[time, n_lev, n_lat]` (m^3/s), `rho2` `[n_lev]`,
    `lat` `[n_lat]`.
    """
    data_dir = Path(data_dir)
    stem = data_dir / f"FullDepth_ASMOC_interp_gr_r{realization}"

    with np.load(stem.with_suffix(".npz")) as fh:
        metadata_keys = ("source_model", "source_experiment",
                         "time_units", "time_calendar")
        return {
            "psi": fh["psi"], "rho2": fh["rho2"], "lat": fh["lat"],
            "time_month": fh["time_month"] if "time_month" in fh.files else None,
            "provenance": {key: str(fh[key])
                           for key in metadata_keys
                           if key in fh.files},
        }


# ---------------------------------------------------------------------------
# Preprocessed evaluation data
# ---------------------------------------------------------------------------
@dataclass
class EvaluationData:
    """Flattened predictors and targets of one preprocessed scenario."""

    x: np.ndarray            # [time, n_features]
    block_sizes: np.ndarray  # features per covariate, in order
    y: np.ndarray            # [time, n_valid_outputs]
    psi_mask: np.ndarray     # validity over the flattened (lev, lat) plane
    lat: np.ndarray
    rho2: np.ndarray
    n_lev: int
    n_lat: int
    #: [lev, lat] reference mean for anomaly targets, if available.
    moc_baseline: "np.ndarray | None" = None
    time_month: "np.ndarray | None" = None
    realization_index: "np.ndarray | None" = None
    mascon_lon: "np.ndarray | None" = None
    mascon_lat: "np.ndarray | None" = None
    moc_convention: str = "absolute"
    moc_file: str = ""
    input_files: tuple[str, ...] = ()
    input_source_names: tuple[str, ...] = ()
    input_baseline_specs: tuple[str, ...] = ()
    input_wind_sources: tuple[str, ...] = ()


def load_evaluation_data(
    data_dir: Path | str,
    covariate_names: list[str],
    tag: str = "_r36_r40",
    lpf_key: str = "_LPF_ALL",
    expected_wind_convention: str | None = None,
    expected_wind_source: str | None = None,
    expected_moc_convention: str | None = None,
) -> EvaluationData:
    """Load aligned evaluation inputs and check wind and MOC conventions."""
    data_dir = Path(data_dir)
    moc_path = require_file(data_dir / f"MOC{tag}.npz", "MOC data")
    with np.load(moc_path) as data:
        rho2 = data["rho2_full"]
        lat = data["lat_psi"]
        psi = np.transpose(data[f"MOC{lpf_key}"], (0, 2, 1))  # [time, lev, lat]
        target_time = data["time_month"] if "time_month" in data.files else None
        target_members = (
            data["realization_index"]
            if "realization_index" in data.files
            else None
        )
        moc_convention = (
            str(np.asarray(data["moc_convention"]).item())
            if "moc_convention" in data.files else "absolute"
        )
        # stored [lat, lev] -> [lev, lat] to match psi's (lev, lat) plane
        moc_baseline = (np.asarray(data["MOC_baseline_mean"]).T
                        if "MOC_baseline_mean" in data.files else None)
        moc_baseline_period = (
            np.asarray(data["moc_baseline_period"]).astype(int).reshape(-1)
            if "moc_baseline_period" in data.files else None
        )
    if (expected_moc_convention is not None
            and moc_convention != expected_moc_convention):
        raise ValueError(
            f"{data_dir / f'MOC{tag}.npz'}: moc_convention is "
            f"{moc_convention!r}, but the trained network expects "
            f"{expected_moc_convention!r}")

    n_t, n_lev, n_lat = psi.shape
    if moc_convention == "anomaly_2004_2009":
        if moc_baseline is None:
            raise ValueError(
                f"{data_dir / f'MOC{tag}.npz'}: anomaly target is missing "
                "MOC_baseline_mean"
            )
        if moc_baseline.shape != (n_lev, n_lat):
            raise ValueError(
                f"{data_dir / f'MOC{tag}.npz'}: MOC baseline has "
                f"model-facing shape {moc_baseline.shape}; expected [lev, lat] "
                f"{(n_lev, n_lat)}"
            )
        expected_period = np.asarray(BASELINE_YEARS, dtype=int)
        if (moc_baseline_period is None
                or not np.array_equal(moc_baseline_period, expected_period)):
            actual = (None if moc_baseline_period is None
                      else moc_baseline_period.tolist())
            raise ValueError(
                f"{data_dir / f'MOC{tag}.npz'}: MOC baseline period is "
                f"{actual}, expected {expected_period.tolist()}"
            )
    flat = psi.reshape(n_t, -1)
    psi_mask = ~np.isnan(flat).any(axis=0)

    blocks, sizes = [], []
    input_paths: list[str] = []
    source_names: list[str] = []
    baseline_specs: list[str] = []
    wind_sources: list[str] = []
    reference_lon = reference_lat = None
    for name in covariate_names:
        path = require_file(data_dir / f"{name}{tag}.npz", name)
        with np.load(path) as data:
            block = data[f"{name}{lpf_key}"]
            if block.shape[0] != n_t:
                raise ValueError(
                    f"{path}: {block.shape[0]} rows do not match the MOC ({n_t})"
                )
            if target_time is not None:
                if "time_month" not in data.files or not np.array_equal(
                    data["time_month"], target_time
                ):
                    raise ValueError(f"{path}: time_month is not aligned with the MOC")
            if target_members is not None:
                if "realization_index" not in data.files or not np.array_equal(
                    data["realization_index"], target_members
                ):
                    raise ValueError(
                        f"{path}: realization_index is not aligned with the MOC"
                    )

            lon = np.asarray(data["mascon_lon"])
            lat_mascon = np.asarray(data["mascon_lat"])
            if reference_lon is None:
                reference_lon, reference_lat = lon, lat_mascon
            elif not (
                np.array_equal(lon, reference_lon)
                and np.array_equal(lat_mascon, reference_lat)
            ):
                raise ValueError(f"{path}: mascon coordinates/order differ by covariate")

            if name.startswith("uas_mascon") and expected_wind_convention is not None:
                convention = (
                    str(np.asarray(data["wind_convention"]).item())
                    if "wind_convention" in data.files
                    else None
                )
                if convention != expected_wind_convention:
                    raise ValueError(
                        f"{path}: wind_convention={convention!r}, but the trained "
                        f"network expects {expected_wind_convention!r}"
                    )
            if name.startswith("uas_mascon") and expected_wind_source is not None:
                source = (
                    str(np.asarray(data["wind_source"]).item())
                    if "wind_source" in data.files
                    else None
                )
                if source != expected_wind_source:
                    raise ValueError(
                        f"{path}: wind_source={source!r}, but this evaluation "
                        f"expects {expected_wind_source!r}. Rebuild the test bed "
                        "with the matching wind (or change the expectation) so "
                        "the results are not mislabeled."
                    )
            for key in ("baseline_spec", "wind_source"):
                if key in data.files:  # provenance visibility (stage 06 output)
                    print(f"  {name}{tag}: {key} = {data[key]}")
            source_names.append(
                str(np.asarray(data["source_model"]).item())
                if "source_model" in data.files else "unknown"
            )
            baseline_specs.append(
                str(np.asarray(data["baseline_spec"]).item())
                if "baseline_spec" in data.files else "unknown"
            )
            wind_sources.append(
                str(np.asarray(data["wind_source"]).item())
                if "wind_source" in data.files else ""
            )
        blocks.append(block)
        sizes.append(block.shape[1])
        input_paths.append(str(path.resolve()))

    return EvaluationData(
        x=np.concatenate(blocks, axis=1),
        block_sizes=np.asarray(sizes, dtype=int),
        y=flat[:, psi_mask], psi_mask=psi_mask,
        lat=lat, rho2=rho2, n_lev=n_lev, n_lat=n_lat,
        moc_baseline=moc_baseline,
        time_month=(None if target_time is None else np.asarray(target_time)),
        realization_index=(
            None if target_members is None else np.asarray(target_members)
        ),
        mascon_lon=(None if reference_lon is None else np.asarray(reference_lon)),
        mascon_lat=(None if reference_lat is None else np.asarray(reference_lat)),
        moc_convention=moc_convention,
        moc_file=str(moc_path.resolve()),
        input_files=tuple(input_paths),
        input_source_names=tuple(source_names),
        input_baseline_specs=tuple(baseline_specs),
        input_wind_sources=tuple(wind_sources),
    )
