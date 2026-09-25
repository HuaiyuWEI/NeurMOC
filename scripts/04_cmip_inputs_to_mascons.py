"""Stage 04: average CMIP OBP, SSH, and wind onto mascons.

Save one anomaly array per variable and realization relative to the matching
stage-03 historical baseline. Source-grid coordinates are detected automatically.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.cmip_io import (
    BASELINE_WIND_SUFFIXES,
    WIND_CALIBRATABLE,
    baseline_wind_suffix,
    canonical_wind_var,
    curl_partner_path,
    detect_model,
    find_realization_files,
    make_mascon_averager,
    parse_realizations,
    parse_wind_var,
    read_stress_curl,
    wind_components,
    wind_is_derived,
    wind_output_slot,
    wind_table,
    wind_units,
)
from neurmoc.config import (
    BASELINE_YEARS,
    BASINMASK_DIR,
    GRACE_MASCON_NC,
    MASCON_VERSION,
    WIND_CALIBRATION_DIR,
    cmip_interim_dir,
    cmip_raw_dir,
)
from neurmoc.grids import MasconGeometry, load_grace_land
from neurmoc.io_utils import ensure_dir, load_npz_or_mat, require_dir
from neurmoc.plotting.style import apply_style, save_figure

# ========== User settings (overridable from the command line) ==========
#: CMIP experiment folder under the data root: "ACCESS_historical" |
#: "ACCESS_SSP126/245/370/585" | "MRI_SSP245" | ...
EXPERIMENT = "ACCESS_historical"
REALIZATIONS = "1-35"   # "lo-hi": "1-35" (train), "36-40" (test), "1-1" (single)
#: CMIP source_id used in the file names: "" (auto-detect from the folder's
#: .nc files) | "ACCESS-ESM1-5" | "MRI-ESM2-0" | ...
MODEL = ""
EXPERIMENT_ID = ""       # filename experiment filter ("" = any); only needed
                         # if one folder mixes files from several experiments
#: Wind input: uas, tauu, curltau, or ua@<Pa>. Optional :calibrated or :raw
#: selects the uas-equivalent calibration of pressure-level winds.
WIND_VAR = "uas"
#: Subtract the historical 2004-2009 wind mean, matching OBP/SSH and the
#: satellite wind convention. Set False only for an explicit sensitivity run.
WIND_ANOMALY = True
OUTPUT_EXPERIMENT = ""  # "" -> EXPERIMENT, or EXPERIMENT_fullwind for full wind
#: Stage-03 historical baseline stem; empty selects the matching model baseline.
BASELINE = ""
TIME_CHUNK = 240  # months processed per read (memory control)
#: Save a before/after-mascon-averaging QC figure next to each variable's
#: output (time-mean full field; first realization of the run only).
QC_PLOTS = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-e", "--experiment", default=EXPERIMENT,
                        help="CMIP experiment folder under the data root")
    parser.add_argument("-r", "--realizations", default=REALIZATIONS,
                        help="realization range, e.g. 1-35 or 36-40")
    parser.add_argument("-m", "--model", default=MODEL,
                        help="CMIP source_id in the file names "
                             "('' = auto-detect from the folder)")
    parser.add_argument("-b", "--baseline", default=BASELINE,
                        help="stage-03 baseline file stem; '' = auto-detect "
                             "in the experiment's *_historical folder")
    parser.add_argument("-x", "--experiment-id", default=EXPERIMENT_ID,
                        help="filename experiment filter, e.g. ssp245")
    parser.add_argument("-w", "--wind-var", default=WIND_VAR,
                        help="wind input: uas, or ua@<Pa> (e.g. ua@100000)")
    parser.add_argument(
        "-o",
        "--output-experiment",
        default=OUTPUT_EXPERIMENT,
        help="interim output label; default is the source experiment for "
        "anomalies and <experiment>_fullwind for full wind",
    )
    parser.add_argument(
        "--wind-only", action="store_true",
        help="process ONLY the wind variable, reusing the existing OBP/SSH "
             "products. For a wind-variant sensitivity run in a tree whose "
             "ocean products are already built: they do not depend on the "
             "wind, and rewriting them would give byte-identical files new "
             "mtimes, which the provenance staleness gate reads as a change",
    )
    wind_mode = parser.add_mutually_exclusive_group()
    wind_mode.add_argument("--wind-anomaly", dest="wind_anomaly",
                           action="store_true",
                           help="subtract the historical 2004-2009 wind mean (default)")
    wind_mode.add_argument("--full-wind", dest="wind_anomaly",
                           action="store_false",
                           help="retain full wind for an explicit sensitivity run")
    parser.set_defaults(wind_anomaly=WIND_ANOMALY)
    return parser.parse_args()


def resolved_output_experiment(
    experiment: str, output_experiment: str, wind_anomaly: bool
) -> str:
    """Choose a non-colliding label for the selected wind convention."""
    output = output_experiment.strip() or (
        experiment if wind_anomaly else f"{experiment}_fullwind"
    )
    if not wind_anomaly and output == experiment:
        raise ValueError(
            "A full-wind sensitivity cannot write to the standard anomaly "
            "experiment. Use -o/--output-experiment with a distinct label."
        )
    return output


def file_pattern(var: str, table: str, model: str, experiment_id: str) -> str:
    return f"{var}_{table}_{model}_{experiment_id}*.nc" if experiment_id \
        else f"{var}_{table}_{model}*.nc"


def split_wind_spec(spec: str) -> tuple[str, str]:
    """Split -w into (standard wind variable, calibration mode).

    "uas" | "ua@<Pa>"                     -> mode "auto" (calibrated for
                                             NorESM models, raw otherwise)
    "ua@<Pa>:calibrated" | "ua@<Pa>:raw"  -> forced mode
    """
    base, sep, mode = spec.strip().partition(":")
    mode = mode.strip().lower() if sep else "auto"
    if mode not in {"auto", "calibrated", "raw"}:
        raise ValueError(
            f"Unknown wind-calibration mode {mode!r}; use ':calibrated' or ':raw'")
    name, _ = parse_wind_var(base)
    if name not in WIND_CALIBRATABLE and mode != "auto":
        raise ValueError(
            f"{name} IS the measured quantity; the ':calibrated'/':raw' "
            "suffix rescales ua anomalies to uas-equivalent and applies "
            "only to ua@<Pa>")
    return canonical_wind_var(base), mode


def resolve_wind_calibration(mode: str, wind_var: str, model: str,
                             wind_anomaly: bool) -> Path | None:
    """Slope file to apply to the ua anomalies, or None otherwise.

    Only ua is calibratable: uas and tauu ARE the quantity being modelled,
    so there is nothing to rescale them towards.
    """
    name, level = parse_wind_var(wind_var)
    if name not in WIND_CALIBRATABLE:
        return None
    if not wind_anomaly:
        if mode == "calibrated":
            raise ValueError(
                "':calibrated' rescales the wind ANOMALIES and cannot be "
                "combined with --full-wind")
        return None
    if mode == "raw" or (mode == "auto" and not model.startswith("NorESM")):
        return None

    candidates = [WIND_CALIBRATION_DIR / f"uas_calibration_ua{level:g}_{model}.npz"]
    if model.startswith("NorESM2") and model != "NorESM2-LM":
        # Reuse the NorESM2-LM mascon calibration for related NorESM2 models.
        candidates.append(
            WIND_CALIBRATION_DIR / f"uas_calibration_ua{level:g}_NorESM2-LM.npz")
    for path in candidates:
        if path.is_file():
            if path != candidates[0]:
                print(f"wind calibration: no {candidates[0].name}; using "
                      f"{path.name} (same NorESM2 atmosphere)")
            return path
    raise FileNotFoundError(
        f"No wind calibration for {model} ({candidates[0]}). Fit one with "
        "make_wind_calibration.py from a member that has both ua and uas, "
        f"or run with -w {wind_var}:raw to skip the rescaling.")


def load_wind_calibration(path: Path, geometry, wind_var: str) -> np.ndarray:
    """Per-mascon uas-equivalent slopes, validated against grid and level."""
    with np.load(path) as data:
        recorded = canonical_wind_var(str(np.asarray(data["wind_var"]).item()))
        if recorded != wind_var:
            raise RuntimeError(f"{path.name} calibrates {recorded}, not {wind_var}")
        if not (np.allclose(data["lon_mascon"], geometry.lon_center)
                and np.allclose(data["lat_mascon"], geometry.lat_center)):
            raise RuntimeError(
                f"{path.name}: mascon coordinates/order differ from the "
                "active Mascon_AtlSO geometry; rerun make_wind_calibration.py")
        slope = np.asarray(data["slope"], dtype=float)
        print(f"wind calibration: {path.name} | median slope "
              f"{np.nanmedian(slope):.3f}, median r "
              f"{np.nanmedian(data['r']):.3f} (fitted on "
              f"{np.asarray(data['source_model']).item()} "
              f"{np.asarray(data['member']).item()})")
    return slope


def pressure_level_index(nc, var: str, target_pa: float) -> tuple[int, int, float]:
    """Return `(axis, index, selected_pa)` for a pressure-level variable."""
    dimensions = nc[var].dimensions
    for level_name in ("plev", "lev", "level"):
        if level_name not in nc.variables or level_name not in dimensions:
            continue
        coordinate = nc[level_name]
        levels = np.asarray(coordinate[:], dtype=float)
        units = str(getattr(coordinate, "units", "Pa")).strip().lower()
        if units in {"hpa", "mbar", "millibar", "millibars"}:
            levels_pa = levels * 100.0
        elif units in {"pa", "pascal", "pascals", ""}:
            levels_pa = levels
        else:
            raise ValueError(
                f"Unsupported pressure units {units!r} in {coordinate.name}; expected Pa or hPa")
        index = int(np.abs(levels_pa - target_pa).argmin())
        return dimensions.index(level_name), index, float(levels_pa[index])
    raise KeyError(f"No pressure-level coordinate found for {var} in {nc.filepath()}")


def read_variable_chunks(path, varname, averager, chunk=TIME_CHUNK,
                         level_target=None, collect_sum=False):
    """Read `[time, ...grid]` variable in chunks; return mascon means, time,
    years, and (when `collect_sum`) the native-grid time sum for QC figures.

    `level_target` (Pa) selects the nearest vertical level of a 3-D variable
    (e.g. zonal wind `ua` on pressure levels) before averaging.
    """
    import netCDF4

    parts = []
    native_sum = None
    with netCDF4.Dataset(path) as nc:
        time_var = nc["time"]
        time = np.asarray(time_var[:])
        time_units = str(time_var.units)
        time_calendar = str(getattr(time_var, "calendar", "standard"))
        dates = netCDF4.num2date(time, time_units, time_calendar)
        years = np.array([d.year for d in dates])
        time_month = np.asarray([f"{d.year:04d}-{d.month:02d}" for d in dates])

        if nc[varname].dimensions[0] != "time":
            raise ValueError(
                f"Expected time to be the first dimension of {varname} in {path.name}, "
                f"got {nc[varname].dimensions}")
        level_axis = None
        level_idx = None
        selected_pressure_pa = None
        if level_target is not None:
            level_axis, level_idx, selected_pressure_pa = pressure_level_index(
                nc, varname, level_target)
            print(f"    pressure[{level_idx}] = {selected_pressure_pa:g} Pa "
                  f"(nearest {level_target:g} Pa)")

        for start in range(0, time.size, chunk):
            selection = [slice(None)] * nc[varname].ndim
            selection[0] = slice(start, start + chunk)
            if level_idx is not None:
                selection[level_axis] = level_idx
            field = np.ma.filled(nc[varname][tuple(selection)], np.nan)
            if collect_sum:
                block_sum = field.sum(axis=0)  # NaN cells stay NaN
                native_sum = block_sum if native_sum is None \
                    else native_sum + block_sum
            parts.append(averager(field))
    time_metadata = {
        "time_units": time_units,
        "time_calendar": time_calendar,
        "time_month": time_month,
    }
    if selected_pressure_pa is not None:
        time_metadata["selected_pressure_pa"] = selected_pressure_pa
    return np.concatenate(parts, axis=0), time, years, native_sum, time_metadata


def read_curl_chunks(tauu_path, tauv_path, averager, chunk=TIME_CHUNK,
                     collect_sum=False):
    """Compute wind-stress curl on the native grid before mascon averaging."""
    import netCDF4

    parts = []
    native_sum = None
    with netCDF4.Dataset(tauu_path) as nc:
        time_var = nc["time"]
        time = np.asarray(time_var[:])
        time_units = str(time_var.units)
        time_calendar = str(getattr(time_var, "calendar", "standard"))
        dates = netCDF4.num2date(time, time_units, time_calendar)
        time_month = np.asarray([f"{d.year:04d}-{d.month:02d}" for d in dates])
        years = np.array([d.year for d in dates])

    for start in range(0, time.size, chunk):
        field = read_stress_curl(tauu_path, tauv_path,
                                 slice(start, start + chunk))
        if collect_sum:
            block_sum = np.nansum(field, axis=0)
            native_sum = block_sum if native_sum is None \
                else native_sum + block_sum
        parts.append(averager(field))
    time_metadata = {
        "time_units": time_units,
        "time_calendar": time_calendar,
        "time_month": time_month,
    }
    return np.concatenate(parts, axis=0), time, years, native_sum, time_metadata


def concatenate_time_blocks(times, metadata, label):
    """Concatenate file time axes and validate their units and monthly order."""
    units = {part["time_units"] for part in metadata}
    calendars = {part["time_calendar"] for part in metadata}
    if len(units) != 1 or len(calendars) != 1:
        raise RuntimeError(
            f"{label}: split files use inconsistent time metadata: "
            f"units={sorted(units)}, calendars={sorted(calendars)}")
    time = np.concatenate(times)
    time_month = np.concatenate([part["time_month"] for part in metadata])
    month_numbers = time_month.astype("datetime64[M]").astype(np.int64)
    if np.any(np.diff(month_numbers) != 1):
        raise RuntimeError(f"{label}: time_month is duplicated, gapped, or non-monotonic")
    if np.any(np.diff(time) <= 0):
        raise RuntimeError(f"{label}: numeric NetCDF time is not strictly increasing")
    merged = {
        "time_units": next(iter(units)),
        "time_calendar": next(iter(calendars)),
        "time_month": time_month,
    }
    selected_pressures = [part.get("selected_pressure_pa") for part in metadata]
    if any(value is not None for value in selected_pressures):
        pressures = {value for value in selected_pressures if value is not None}
        if any(value is None for value in selected_pressures) or len(pressures) != 1:
            raise RuntimeError(
                f"{label}: split files selected inconsistent pressure levels: {pressures}")
        merged["selected_pressure_pa"] = next(iter(pressures))
    return time, merged


def grid_coords(sample_file):
    """Native lon/lat of a NetCDF file (2-D curvilinear or 1-D regular)."""
    import netCDF4

    with netCDF4.Dataset(sample_file) as nc:
        for lat_name, lon_name in [("latitude", "longitude"), ("lat", "lon")]:
            if lat_name in nc.variables and lon_name in nc.variables:
                return (np.asarray(nc[lon_name][:]).astype(float),
                        np.asarray(nc[lat_name][:]))
    raise KeyError(f"No lat/lon coordinates found in {sample_file}")


def qc_plot(tag, sample_file, native_mean, geometry, mascon_mean, unit, out_stem):
    """Time-mean field on the model's native grid vs the mascon averages."""
    apply_style()
    lon, lat = grid_coords(sample_file)
    land_lon, land_lat, land = load_grace_land(GRACE_MASCON_NC)
    vmin, vmax = np.nanpercentile(mascon_mean, [2, 98])

    fig, axes = plt.subplots(2, 1, figsize=(7, 6.4), constrained_layout=True)
    ax = axes[0]
    ax.set_facecolor("#d9d9d9")
    if lon.ndim == 1:  # regular grid: recenter to -180..180 and mesh
        lon_pm = (lon + 180.0) % 360.0 - 180.0
        order = np.argsort(lon_pm)
        ax.pcolormesh(lon_pm[order], lat, native_mean[:, order],
                      cmap="viridis", vmin=vmin, vmax=vmax,
                      shading="nearest", rasterized=True)
    else:  # curvilinear grid: robust cell scatter
        lon_pm = (lon + 180.0) % 360.0 - 180.0
        ax.scatter(lon_pm.ravel(), lat.ravel(), c=native_mean.ravel(), s=0.5,
                   cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0,
                   rasterized=True)
    ax.set_title(f"{tag}: before - native model grid "
                 f"{'x'.join(str(s) for s in native_mean.shape)}", loc="left")

    ax = axes[1]
    ax.pcolormesh(land_lon, land_lat, land,
                  cmap=plt.matplotlib.colors.ListedColormap(["#d9d9d9"]),
                  shading="nearest", rasterized=True)
    sc = ax.scatter(geometry.lon_center, geometry.lat_center, c=mascon_mean,
                    s=4, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(f"{tag}: after - JPL mascon averages "
                 f"({mascon_mean.size} mascons)", loc="left")
    ax.set_xlabel("Longitude")

    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 80)
        ax.set_ylabel("Latitude")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.set_title(unit, pad=6)
    save_figure(fig, out_stem, formats=("png",))


def default_baseline(experiment: str, wind_var: str = WIND_VAR) -> str:
    """Stage-03 historical baseline matching the model and wind input."""
    historical_experiment = experiment if experiment.endswith("_historical") \
        else f"{experiment.rsplit('_', 1)[0]}_historical"
    hist_dir = cmip_interim_dir(historical_experiment)
    stems = sorted({p.with_suffix("")
                    for p in hist_dir.glob("Mascon_TimeMean_2004_2009*")
                    if p.suffix in (".npz", ".mat")})
    suffix = baseline_wind_suffix(wind_var)
    if suffix:
        matching = [s for s in stems if s.name.endswith(suffix)]
    else:
        matching = [s for s in stems
                    if not s.name.endswith(BASELINE_WIND_SUFFIXES)]
    if len(matching) == 1:
        return str(matching[0])
    found = [s.name for s in stems] if stems else "nothing"
    wanted = f"*{suffix}" if suffix else "an untagged"
    raise FileNotFoundError(
        f"Baseline auto-detection in {hist_dir} found {found}; expected "
        f"exactly one {wanted} Mascon_TimeMean_2004_2009* file for "
        f"--wind-var {canonical_wind_var(wind_var)}. Run stage 03 there "
        "with the same -w, or set BASELINE / -b explicitly.")


def resolve_baseline_file(baseline_spec: str | Path) -> Path:
    """Resolve an extensionless baseline stem, preferring NPZ over MAT."""
    stem = Path(baseline_spec)
    for suffix in (".npz", ".mat"):
        candidate = stem.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Neither {stem.with_suffix('.npz')} nor {stem.with_suffix('.mat')} exists"
    )


def _scalar_text(value, field: str) -> str:
    """Read one required string scalar from NPZ/MAT provenance."""
    array = np.asarray(value).squeeze()
    if array.size != 1:
        raise ValueError(f"Baseline provenance {field!r} must be one scalar, got {array.shape}")
    item = array.item()
    return item.decode() if isinstance(item, bytes) else str(item)


def _text_list(value) -> list[str]:
    """Convert a NumPy string vector to ordinary Python strings."""
    items = np.asarray(value).reshape(-1)
    return [item.decode() if isinstance(item, bytes) else str(item) for item in items]


def load_baseline(baseline_spec: str, model: str, wind_var: str,
                  require_wind: bool = True) -> dict[str, np.ndarray | float]:
    """Load stage-03 reference means for the selected model and wind input."""
    try:
        data = load_npz_or_mat(baseline_spec)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No stage-03 baseline at '{baseline_spec}'. Download the model's "
            "historical pbo/zos/wind, run 03_cmip_baseline_mean.py on its "
            "*_historical folder, and point BASELINE / -b at that output "
            "stem.") from None
    source = data.get("source_model")
    if source is None:
        raise RuntimeError(
            f"Baseline '{baseline_spec}' has no source_model provenance. "
            "Rerun Stage 03 before rebuilding the model-ready data.")
    source = _scalar_text(source, "source_model")
    if source != model:
        raise RuntimeError(
            f"Baseline '{baseline_spec}' was computed from "
            f"{source}, but this experiment's model is {model}.")

    period = data.get("baseline_period")
    if period is not None and not np.array_equal(
            np.asarray(period, dtype=int).reshape(-1), np.asarray(BASELINE_YEARS)):
        raise RuntimeError(
            f"Baseline '{baseline_spec}' uses period {np.asarray(period).reshape(-1)}, "
            f"expected {BASELINE_YEARS}.")

    values = np.asarray(data["Input_vars_mascon"])
    if values.ndim != 2:
        raise ValueError(
            f"Baseline Input_vars_mascon must be 2-D, got shape {values.shape}")
    if "baseline_variables" in data:
        names = _text_list(data["baseline_variables"])
    elif values.shape[1] == 2:
        names = ["pbo", "zos"]  # baseline without a wind column
    else:
        raise RuntimeError(
            f"Baseline '{baseline_spec}' does not identify its {values.shape[1]} columns. "
            "Rerun stage 03 to create baseline_variables provenance.")
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError(
            f"Invalid baseline_variables {names} for baseline shape {values.shape}")

    missing_ocean = [name for name in ("pbo", "zos") if name not in names]
    if missing_ocean:
        raise RuntimeError(f"Baseline '{baseline_spec}' is missing {missing_ocean}")
    references = {name: values[:, names.index(name)] for name in ("pbo", "zos")}

    if require_wind:
        wind_var = canonical_wind_var(wind_var)
        recorded_wind = data.get("wind_source")
        if recorded_wind is None:
            raise RuntimeError(
                f"Baseline '{baseline_spec}' has no wind baseline. Rerun stage 03 with "
                f"--wind-var {wind_var}, or explicitly request --full-wind.")
        recorded_wind = canonical_wind_var(_scalar_text(recorded_wind, "wind_source"))
        if recorded_wind != wind_var:
            raise RuntimeError(
                f"Baseline '{baseline_spec}' used {recorded_wind}, but stage 04 uses "
                f"{wind_var}. Rerun stage 03 with the same --wind-var setting.")
        if wind_var not in names:
            raise RuntimeError(
                f"Baseline '{baseline_spec}' records {wind_var} but has columns {names}")
        references["wind"] = values[:, names.index(wind_var)]
        wind_name, wind_level = parse_wind_var(wind_var)
        if wind_name == "ua" and wind_level is not None:
            selected = data.get("wind_selected_pressure_pa")
            if selected is None:
                raise RuntimeError(
                    f"Baseline '{baseline_spec}' does not record the pressure actually "
                    "selected for ua. Rerun stage 03.")
            references["wind_selected_pressure_pa"] = float(np.asarray(selected).item())

    print(f"baseline: {baseline_spec}")
    print(f"  source_model={source}; variables={names}")
    return references


def subtract_baseline(data: np.ndarray, reference: np.ndarray, label: str) -> np.ndarray:
    """Subtract one per-mascon reference after an explicit shape check."""
    if data.ndim != 2:
        raise ValueError(f"{label}: predictor data must be 2-D, got {data.shape}")
    reference = np.asarray(reference)
    expected = (data.shape[1],)
    if reference.shape != expected:
        raise ValueError(
            f"{label}: data/reference shape mismatch: {data.shape} vs {reference.shape}; "
            f"expected reference {expected}")
    return data - reference[None, :]


def save_realization(out_dir, tag, realization, data, time, time_metadata,
                     geometry, extra=None):
    payload = geometry.to_dict()
    payload.update({"Input_vars_mascon": data, "Input_time": time,
                    "time_units": time_metadata["time_units"],
                    "time_calendar": time_metadata["time_calendar"],
                    "time_month": time_metadata["time_month"]})
    if extra:
        payload.update(extra)
    out_file = out_dir / f"Mascon_{MASCON_VERSION}_{tag}_r{realization}.npz"
    np.savez(out_file, **payload)
    print("  saved", out_file.name, data.shape)


def main(experiment: str = EXPERIMENT, realizations_spec: str = REALIZATIONS,
         model: str = MODEL, baseline_spec: str = BASELINE,
         experiment_id: str = EXPERIMENT_ID, wind_var: str = WIND_VAR,
         wind_anomaly: bool = WIND_ANOMALY,
         output_experiment: str = OUTPUT_EXPERIMENT,
         wind_only: bool = False) -> None:
    data_dir = require_dir(cmip_raw_dir(experiment), "raw CMIP experiment data")
    output_experiment = resolved_output_experiment(
        experiment, output_experiment, wind_anomaly
    )
    out_dir = ensure_dir(cmip_interim_dir(output_experiment))
    if not model:
        model = detect_model(data_dir)
        print(f"model auto-detected from {experiment}: {model}")
    # Resolve the baseline from the standard wind variable.
    wind_var, wind_calibration_mode = split_wind_spec(wind_var)
    if not baseline_spec:
        baseline_spec = default_baseline(experiment, wind_var)
    realizations = parse_realizations(realizations_spec)
    geometry = MasconGeometry.from_dict(load_npz_or_mat(BASINMASK_DIR / "Mascon_AtlSO"))

    # wind availability pre-check, also before the slow ocean processing
    wind_name, wind_level = parse_wind_var(wind_var)
    # Keep wind-variant outputs separate.
    wind_slot = wind_output_slot(wind_var)
    baseline = load_baseline(
        baseline_spec, model, wind_var, require_wind=wind_anomaly)
    calibration_file = resolve_wind_calibration(
        wind_calibration_mode, wind_var, model, wind_anomaly)
    wind_slope = (load_wind_calibration(calibration_file, geometry, wind_var)
                  if calibration_file else None)
    if wind_name == "ua" and wind_slope is None:
        print("wind calibration: none (raw ua anomalies)")
    print("wind convention:", "2004-2009 anomaly" if wind_anomaly else "full wind")
    print("output experiment:", output_experiment)
    # Discover derived curl through its tauu source files.
    wind_read_var = wind_components(wind_var)[0]
    wind_by_r = find_realization_files(
        data_dir,
        file_pattern(wind_read_var, wind_table(wind_var), model, experiment_id))
    if not wind_by_r:
        hint = ""
        if wind_name == "uas" and find_realization_files(
                data_dir, file_pattern("ua", "Amon", model, experiment_id)):
            hint = (" - 3-D ua files exist though: this model likely "
                    "publishes no 10-m wind (e.g. NorESM); rerun with "
                    "-w ua@100000")
        raise FileNotFoundError(f"No {wind_read_var} files in {data_dir}{hint}")

    # ---- ocean variables: pbo (OBP) and zos (SSH) --------------------------
    ocean_averagers = {}
    time_month_by_realization = {}
    if wind_only:
        # Reuse wind-independent OBP and SSH products without changing their
        # timestamps; read only their month axes for wind alignment.
        print("wind-only: reusing the existing OBP/SSH products "
              "(not recomputed, so their mtimes are untouched)")
        for r in realizations:
            for tag in ("OBP", "SSH"):
                path = out_dir / f"Mascon_{MASCON_VERSION}_{tag}_r{r}.npz"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"--wind-only needs the existing {tag} product "
                        f"{path}. Run stage 04 once without --wind-only "
                        "for this experiment first.")
                with np.load(path) as fh:
                    if "time_month" not in fh.files:
                        raise RuntimeError(
                            f"{path} predates the time_month axis; rerun "
                            "stage 04 for this experiment without "
                            "--wind-only.")
                    months = np.asarray(fh["time_month"]).astype("U7")
                previous = time_month_by_realization.get(r)
                if previous is not None and not np.array_equal(previous, months):
                    raise RuntimeError(
                        f"r{r}: existing OBP and SSH products disagree on "
                        "their monthly axis")
                time_month_by_realization[r] = months
    else:
        ocean_files = {
            var: find_realization_files(
                data_dir, file_pattern(var, "Omon", model, experiment_id))
            for var in ("pbo", "zos")
        }
        for r in realizations:
            want_qc = QC_PLOTS and r == realizations[0]
            for var, tag in [("pbo", "OBP"), ("zos", "SSH")]:
                files = ocean_files[var].get(r)
                if not files:
                    raise FileNotFoundError(
                        f"No {var} file for realization r{r} in {data_dir}")
                if var not in ocean_averagers:
                    ocean_averagers[var] = make_mascon_averager(
                        files[0], geometry, areacello_model=model)
                blocks, times = [], []
                time_metadata_blocks = []
                native_sum = None
                for var_file in files:
                    print(f"r{r}: {var_file.name}")
                    mascon_data, time, _, nsum, time_metadata = read_variable_chunks(
                        var_file, var, ocean_averagers[var], collect_sum=want_qc)
                    blocks.append(mascon_data)
                    times.append(time)
                    time_metadata_blocks.append(time_metadata)
                    if nsum is not None:
                        native_sum = nsum if native_sum is None else native_sum + nsum
                data = np.concatenate(blocks, axis=0)
                time, time_metadata = concatenate_time_blocks(
                    times, time_metadata_blocks, f"{var} r{r}")
                previous_months = time_month_by_realization.get(r)
                if previous_months is not None and not np.array_equal(
                        previous_months, time_metadata["time_month"]):
                    raise RuntimeError(
                        f"{var} r{r}: monthly time axis does not match the other predictors")
                time_month_by_realization[r] = time_metadata["time_month"]

                if want_qc:
                    qc_plot(tag, files[0], native_sum / data.shape[0], geometry,
                            data.mean(axis=0), "Pa" if var == "pbo" else "m",
                            out_dir / f"Mascon_{MASCON_VERSION}_{tag}_r{r}")

                ref = baseline[var]
                data = subtract_baseline(data, ref, f"{var} r{r}")
                extra = {"OBP_2004_2009_mean" if tag == "OBP" else "SSH_2004_2009_mean": ref,
                         "baseline_spec": baseline_spec,
                         "baseline_period": np.asarray(BASELINE_YEARS, dtype=int),
                         "source_model": model}
                save_realization(
                    out_dir, tag, r, data, time, time_metadata, geometry, extra)

    # ---- selected atmospheric predictor ----
    atm_averager = None
    for r in realizations:
        files = wind_by_r.get(r)
        if not files:
            raise FileNotFoundError(f"No {wind_name} file for realization r{r}")
        if atm_averager is None:
            atm_averager = make_mascon_averager(files[0], geometry)

        want_qc = QC_PLOTS and r == realizations[0]
        blocks, times = [], []
        time_metadata_blocks = []
        native_sum = None
        for wind_file in files:
            if wind_is_derived(wind_var):
                partner = curl_partner_path(wind_file)
                print(f"r{r}: {wind_file.name} + {partner.name}")
                mascon_data, time, _, nsum, time_metadata = read_curl_chunks(
                    wind_file, partner, atm_averager, collect_sum=want_qc)
            else:
                print(f"r{r}: {wind_file.name}")
                mascon_data, time, _, nsum, time_metadata = read_variable_chunks(
                    wind_file, wind_name, atm_averager,
                    level_target=wind_level, collect_sum=want_qc)
            blocks.append(mascon_data)
            times.append(time)
            time_metadata_blocks.append(time_metadata)
            if nsum is not None:
                native_sum = nsum if native_sum is None else native_sum + nsum
        data = np.concatenate(blocks, axis=0)
        time, time_metadata = concatenate_time_blocks(
            times, time_metadata_blocks, f"{wind_name} r{r}")
        if not np.array_equal(
                time_month_by_realization[r], time_metadata["time_month"]):
            raise RuntimeError(
                f"{wind_name} r{r}: monthly time axis does not match OBP/SSH")
        if wind_level is not None and wind_anomaly:
            historical_pressure = float(baseline["wind_selected_pressure_pa"])
            experiment_pressure = float(time_metadata["selected_pressure_pa"])
            if not np.isclose(historical_pressure, experiment_pressure):
                raise RuntimeError(
                    f"{wind_name} r{r}: scenario selected {experiment_pressure:g} Pa, "
                    f"but the historical baseline used {historical_pressure:g} Pa")

        if want_qc:
            qc_plot(wind_var, files[0], native_sum / data.shape[0], geometry,
                    data.mean(axis=0), wind_units(wind_var),
                    out_dir / f"Mascon_{MASCON_VERSION}_{wind_slot}_r{r}")

        extra = {
            "wind_source": wind_var,
            "wind_anomaly": np.bool_(wind_anomaly),
            "wind_convention": np.str_(
                "anomaly_2004_2009" if wind_anomaly else "full_wind"),
            "wind_calibration": np.str_(
                calibration_file.stem if calibration_file else "none"),
            "baseline_spec": baseline_spec,
            "baseline_period": np.asarray(BASELINE_YEARS, dtype=int),
            "source_model": model,
        }
        if wind_anomaly:
            wind_reference = np.asarray(baseline["wind"])
            data = subtract_baseline(data, wind_reference, f"{wind_name} r{r}")
            extra["wind_2004_2009_mean"] = wind_reference
            if wind_slope is not None:  # uas-equivalent anomaly rescaling
                data = data * wind_slope[None, :]
        if wind_level is not None:
            extra["wind_selected_pressure_pa"] = time_metadata["selected_pressure_pa"]
        save_realization(
            out_dir, wind_slot, r, data, time, time_metadata, geometry, extra)


if __name__ == "__main__":
    args = parse_args()
    main(args.experiment, args.realizations, args.model, args.baseline,
         args.experiment_id, args.wind_var, args.wind_anomaly,
         args.output_experiment, args.wind_only)
