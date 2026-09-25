"""Stage 03: compute historical 2004-2009 mascon means for OBP, SSH, and wind.

Stage 04 subtracts these means to match the satellite anomaly reference.
Use ``-r common`` when the variables have different realization counts.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.cmip_io import (
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
    wind_table,
)
from neurmoc.config import (
    BASELINE_YEARS,
    BASINMASK_DIR,
    GRACE_MASCON_NC,
    cmip_interim_dir,
    cmip_raw_dir,
)
from neurmoc.grids import MasconGeometry, load_grace_land
from neurmoc.io_utils import ensure_dir, load_npz_or_mat, require_dir
from neurmoc.plotting.style import apply_style, save_figure

# ========== User settings (overridable from the command line) ==========
#: Data folder under the CMIP root holding the historical run:
#: "ACCESS_historical" | "MRI_historical" | ...
EXPERIMENT = "ACCESS_historical"

#: CMIP source_id in the file names: "" (auto-detect from the folder's .nc
#: files) | "ACCESS-ESM1-5" | "MRI-ESM2-0" | ...
MODEL = ""

#: Filename experiment filter; needed if a folder mixes experiments.
EXPERIMENT_ID = ""
#: "1-35" (ACCESS) | "1-1" | "1,6-10" (comma list) | "common" = only the
#: realizations available for all variables (unequal ensembles, e.g. MRI)
REALIZATIONS = "1-35"
#: Wind baseline. This must match stage 04's WIND_VAR setting.
WIND_VAR = "uas"
OUTPUT_STEM = ""                   # "" -> Mascon_TimeMean_2004_2009_r<lo>-r<hi>
QC_PLOTS = True                    # map the saved baselines next to the .npz

#: One-line description of every variable saved to the output file.
SAVED_VARS = {
    "mascon_ID_uniq": "JPL mascon ID of each ocean mascon",
    "lon_mascon_bound1": "mascon bounding box, western edge (deg)",
    "lon_mascon_bound2": "mascon bounding box, eastern edge (deg)",
    "lat_mascon_bound1": "mascon bounding box, southern edge (deg)",
    "lat_mascon_bound2": "mascon bounding box, northern edge (deg)",
    "flag_across_180": "1 = the mascon box crosses the 180-deg meridian",
    "lon_mascon_center": "mascon center longitude (deg)",
    "lat_mascon_center": "mascon center latitude (deg)",
    "Basin_id": "1 = Atlantic + Southern Ocean mascon, NaN elsewhere",
    "Input_vars_mascon": "[mascon, variable] baseline mean; columns = baseline_variables",
    "baseline_variables": "column names/specifications for Input_vars_mascon",
    "baseline_period": "inclusive reference-period years [start, end]",
    "baseline_realizations": "historical realization numbers averaged",
    "source_model": "CMIP source_id the baseline was computed from (stage-04 check)",
    "source_experiment": "historical data folder used to compute the baseline",
    "source_experiment_id": "CMIP experiment_id filename filter",
    "wind_source": "zonal-wind variable and optional pressure used for the baseline",
    "wind_selected_pressure_pa": "actual pressure level selected for a 3-D ua baseline",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-e", "--experiment", default=EXPERIMENT,
                        help="data folder under the CMIP root")
    parser.add_argument("-m", "--model", default=MODEL,
                        help="CMIP source_id in the file names "
                             "('' = auto-detect from the folder)")
    parser.add_argument("-x", "--experiment-id", default=EXPERIMENT_ID,
                        help="filename experiment filter, e.g. historical "
                             "(needed when one folder mixes experiments)")
    parser.add_argument("-r", "--realizations", default=REALIZATIONS,
                        help="realization range, e.g. 1-35 or 1-1")
    parser.add_argument("-o", "--output", default=OUTPUT_STEM,
                        help="output file stem (default derived from realizations)")
    parser.add_argument("-w", "--wind-var", default=WIND_VAR,
                        help="wind baseline: uas, tauu, or ua@<Pa>")
    return parser.parse_args()


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


def baseline_time_slice(nc, label: str) -> slice:
    """Time slice covering Jan of BASELINE_YEARS[0] to Dec of BASELINE_YEARS[1]."""
    import netCDF4 as nc4

    time_var = nc["time"]
    dates = nc4.num2date(time_var[:], time_var.units, time_var.calendar)
    years = np.array([d.year for d in dates])
    idx = np.flatnonzero((years >= BASELINE_YEARS[0]) & (years <= BASELINE_YEARS[1]))
    expected = 12 * (BASELINE_YEARS[1] - BASELINE_YEARS[0] + 1)
    if idx.size != expected:
        raise RuntimeError(f"Expected {expected} baseline months in {label}, "
                           f"found {idx.size}")
    return slice(idx[0], idx[-1] + 1)


def variable_pattern(var, table, model, experiment_id):
    return (f"{var}_{table}_{model}_{experiment_id}*.nc" if experiment_id
            else f"{var}_{table}_{model}*.nc")


def common_realizations(data_dir, variable_specs, model, experiment_id):
    """Realizations shared by all baseline variables."""
    available = {}
    for var, table, _, _ in variable_specs:
        pattern = variable_pattern(var, table, model, experiment_id)
        found = set(find_realization_files(data_dir, pattern))
        if not found:
            raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")
        available[var] = found
        print(f"  {var:12s} {len(found):2d} realizations: {sorted(found)}")
    common = sorted(set.intersection(*available.values()))
    if not common:
        raise RuntimeError(
            f"No realization has all of {list(available)} in {data_dir}")
    print(f"  common      {len(common):2d} realizations: {common}")
    return common


def variable_baseline(data_dir, var, table, model, experiment_id, realizations,
                      geometry, level_target=None, derived_curl=False):
    """Per-mascon 2004-2009 ensemble mean of one variable.

    Wind-stress curl is computed from tauu/tauv on the native grid.
    """
    import netCDF4


    pattern = variable_pattern(var, table, model, experiment_id)
    by_realization = find_realization_files(data_dir, pattern)
    if not by_realization:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")

    averager = None
    mean_field = None
    selected_pressure_pa = None
    count = 0
    for r in realizations:
        files = by_realization.get(r)
        if not files:
            raise FileNotFoundError(f"No {var} file for realization r{r} "
                                    f"(pattern {pattern})")
        # baseline years live in a single file of each realization
        for path in files:
            with netCDF4.Dataset(path) as nc:
                try:
                    tsl = baseline_time_slice(nc, path.name)
                except RuntimeError:
                    if len(files) > 1:
                        continue  # multi-file realization: try the next chunk
                    raise
                print(f"  {path.name} "
                      f"[{BASELINE_YEARS[0]}-01..{BASELINE_YEARS[1]}-12 = "
                      f"months {tsl.start}:{tsl.stop} of the file]")
                selection = [slice(None)] * nc[var].ndim
                selection[0] = tsl
                if nc[var].dimensions[0] != "time":
                    raise ValueError(
                        f"Expected time to be the first dimension of {var} in {path.name}, "
                        f"got {nc[var].dimensions}")
                if level_target is not None:
                    axis, level_idx, selected_pa = pressure_level_index(
                        nc, var, level_target)
                    if selected_pressure_pa is not None and not np.isclose(
                            selected_pressure_pa, selected_pa):
                        raise RuntimeError(
                            f"{var} pressure levels differ across realizations: "
                            f"{selected_pressure_pa:g} vs {selected_pa:g} Pa")
                    selected_pressure_pa = selected_pa
                    selection[axis] = level_idx
                    print(f"    pressure[{level_idx}] = {selected_pa:g} Pa "
                          f"(nearest {level_target:g} Pa)")
                if derived_curl:
                    # Compute curl from both native-grid stress components.
                    field = read_stress_curl(
                        path, curl_partner_path(path), tsl)
                else:
                    field = np.ma.filled(nc[var][tuple(selection)], np.nan)
            if averager is None:
                # Use model cell areas for ocean fields and cosine-latitude
                # areas for atmospheric fields.
                averager = make_mascon_averager(
                    path, geometry,
                    areacello_model=model if table == "Omon" else None)
            with warnings.catch_warnings():
                # All-NaN land cells are handled by mascon averaging.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                block = np.nanmean(field, axis=0)
            mean_field = block if mean_field is None else mean_field + block
            count += 1
            break
        else:
            raise RuntimeError(f"No file of realization r{r} covers the "
                               f"baseline years for {var}")

    return averager(mean_field / count), selected_pressure_pa  # [n_mascons], Pa or None


def qc_figure(geometry, input_vars_mascon, variable_labels, out_stem):
    """One map per saved baseline column, at the mascon centers, over land."""
    lon, lat, land = load_grace_land(GRACE_MASCON_NC)

    fig, axes = plt.subplots(len(variable_labels), 1, squeeze=False,
                             figsize=(7, 3.4 * len(variable_labels)),
                             constrained_layout=True)
    for v, (ax, var) in enumerate(zip(axes.flat, variable_labels)):
        ax.pcolormesh(lon, lat, land, cmap=plt.matplotlib.colors.ListedColormap(
            ["#d9d9d9"]), shading="nearest", rasterized=True)
        values = input_vars_mascon[:, v]
        vmin, vmax = np.nanpercentile(values, [2, 98])
        sc = ax.scatter(geometry.lon_center, geometry.lat_center, c=values,
                        s=4, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 80)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Input_vars_mascon[:, {v}] - {var} "
                     f"{BASELINE_YEARS[0]}-{BASELINE_YEARS[1]} mean", loc="left")
        fig.colorbar(sc, ax=ax, fraction=0.03)
    save_figure(fig, out_stem, formats=("png",))


def main(experiment: str = EXPERIMENT, model: str = MODEL,
         experiment_id: str = EXPERIMENT_ID,
         realizations_spec: str = REALIZATIONS, output_stem: str = OUTPUT_STEM,
         wind_var: str = WIND_VAR) -> None:
    data_dir = require_dir(cmip_raw_dir(experiment), "raw CMIP experiment data")
    out_dir = ensure_dir(cmip_interim_dir(experiment))
    if not model:
        model = detect_model(data_dir)
        print(f"model auto-detected from {experiment}: {model}")
    geometry = MasconGeometry.from_dict(load_npz_or_mat(BASINMASK_DIR / "Mascon_AtlSO"))

    wind_var = canonical_wind_var(wind_var)
    wind_name, wind_level = parse_wind_var(wind_var)
    variable_specs = [("pbo", "Omon", None, "pbo"),
                      ("zos", "Omon", None, "zos"),
                      # Discover derived curl through its tauu source files.
                      (wind_components(wind_var)[0], wind_table(wind_var),
                       wind_level, wind_var)]

    if realizations_spec.strip().lower() == "common":
        realizations = common_realizations(data_dir, variable_specs, model,
                                           experiment_id)
    else:
        realizations = parse_realizations(realizations_spec)

    columns = []
    variable_labels = []
    wind_selected_pressure_pa = None
    for var, table, level_target, label in variable_specs:
        print(f"=== {var} ===")
        column, selected_pressure_pa = variable_baseline(
            data_dir, var, table, model, experiment_id, realizations, geometry,
            level_target=level_target,
            derived_curl=(label == wind_var and wind_is_derived(wind_var)))
        columns.append(column)
        variable_labels.append(label)
        if level_target is not None:
            wind_selected_pressure_pa = selected_pressure_pa
    input_vars_mascon = np.column_stack(columns)  # [n_mascons, n_vars]

    if not output_stem:
        r0, r1 = realizations[0], realizations[-1]
        output_stem = f"Mascon_TimeMean_2004_2009_r{r0}-r{r1}"
        if len(realizations) != r1 - r0 + 1:  # non-contiguous (e.g. common mode)
            output_stem += f"_n{len(realizations)}"
    # Separate baselines for alternative wind predictors.
    wind_suffix = baseline_wind_suffix(wind_var)
    if wind_suffix and not output_stem.endswith(wind_suffix):
        output_stem += wind_suffix
    out_file = out_dir / f"{output_stem}.npz"
    payload = geometry.to_dict()
    payload["Input_vars_mascon"] = input_vars_mascon
    payload["baseline_variables"] = np.asarray(variable_labels)
    payload["baseline_period"] = np.asarray(BASELINE_YEARS, dtype=int)
    payload["baseline_realizations"] = np.asarray(realizations, dtype=int)
    payload["source_model"] = model
    payload["source_experiment"] = experiment
    payload["source_experiment_id"] = experiment_id
    payload["wind_source"] = wind_var
    if wind_selected_pressure_pa is not None:
        payload["wind_selected_pressure_pa"] = wind_selected_pressure_pa
    np.savez(out_file, **payload)

    print("saved", out_file)
    print("variables in the file:")
    for key, value in payload.items():
        arr = np.asarray(value)
        print(f"  {key:18s} {str(arr.shape):12s} {SAVED_VARS.get(key, '')}")
    for v, var in enumerate(variable_labels):
        print(f"  column {v} ({var}): mean {np.nanmean(input_vars_mascon[:, v]):.4g}, "
              f"std {np.nanstd(input_vars_mascon[:, v]):.4g}")

    if QC_PLOTS:
        apply_style()
        qc_figure(geometry, input_vars_mascon, variable_labels,
                  out_dir / output_stem)


if __name__ == "__main__":
    args = parse_args()
    main(args.experiment, args.model, args.experiment_id,
         args.realizations, args.output, args.wind_var)
