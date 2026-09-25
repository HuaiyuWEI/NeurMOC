"""Stage 05: map density-space CMIP MOC onto the common latitude-density grid.

Convert transport to Sv, interpolate each sector by nearest neighbor, and
stitch them at the training-model sector boundary.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.cmip_io import detect_model, find_realization_files, parse_realizations
from neurmoc.config import (
    BASINMASK_DIR,
    CM4_GR_DIR,
    RHO_CONST,
    cmip_interim_dir,
    cmip_raw_dir,
)
from neurmoc.io_utils import ensure_dir, load_npz_or_mat, require_dir
from neurmoc.plotting.style import apply_style, save_figure

# ========== User settings (overridable from the command line) ==========
#: Training model that defines the target density grid.
REFERENCE_MODEL = "ACCESS-ESM1-5"


#: CMIP experiment folder under the data root: "ACCESS_historical" |
#: "ACCESS_SSP126/245/370/585" | "MRI_SSP245" | ...
EXPERIMENT = "ACCESS_historical"
REALIZATIONS = "1-35"       # "lo-hi": "1-40", "36-40" (test), "1-1" (single)
#: CMIP source_id used in the file names: "" (auto-detect from the folder's
#: .nc files, as in stages 03/04) | "ACCESS-ESM1-5" | "MRI-ESM2-0" | ...
MODEL = ""
EXPERIMENT_ID = ""           # filename experiment filter: "" (any) | "ssp245" | ...
OUTPUT_EXPERIMENT = ""       # "" -> same label as EXPERIMENT
PSI_NAME = "msftmrho"        # residual MOC in density coordinates
AMOC_BASIN_INDEX = 0         # atlantic_arctic_ocean sector
GLOBAL_BASIN_INDEX = 2       # global_ocean sector (ACCESS)
#: Use only the Atlantic-Arctic sector if global overturning is unavailable.
AMOC_ONLY = False
RHO_RANGE = (1035.0, 1037.2)
#: Target density-grid file; empty retains native levels for the training model.
TARGET_GRID = str(BASINMASK_DIR / "ACCESS_target_MOC_grid.npz")
TARGET_LAT_SLICE = slice(15, 155)   # gr-grid rows 16..155 (75S..65N)
#: Save a before/after-interpolation QC figure next to the output file
#: (time-mean MOC sections; first realization of the run only).
QC_PLOTS = True


def load_target_latitudes():
    import netCDF4

    files = sorted(require_dir(CM4_GR_DIR, "CM4 gr grid").glob("*deptho*gr.nc"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one deptho gr file in {CM4_GR_DIR}")
    with netCDF4.Dataset(files[0]) as nc:
        return np.asarray(nc["lat"][:])


def nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(source[None, :] - target[:, None]).argmin(axis=1)


def find_separation_index(psi_amoc: np.ndarray) -> int:
    """Latitude index of the AMOC sector's southern data edge."""
    valid = np.isfinite(psi_amoc[0]).any(axis=0)
    sep = int(np.argmax(valid))
    if sep == 0 or not valid.any():
        raise RuntimeError("Could not determine the AMOC/SOMOC separation latitude")
    return sep


def qc_plot(psi_before, rho2_src, lat_src, psi_after, rho_target, lat_target,
            stitch_lat, out_stem):
    """Time-mean MOC on the native grid vs after the nearest-neighbour
    interpolation, over the evaluated density window, stitch latitude dashed."""
    vlim = 25.0
    sig_lo, sig_hi = rho_target.min() - 1000 - 0.1, rho_target.max() - 1000 + 0.1

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), constrained_layout=True)
    panels = [
        (psi_before, rho2_src - 1000, lat_src,
         f"before: native grid ({rho2_src.size} densities x {lat_src.size} lats)"),
        (psi_after, rho_target - 1000, lat_target,
         f"after: nearest-neighbour on target grid "
         f"({rho_target.size} x {lat_target.size})"),
    ]
    for ax, (field, sigma2, lat, title) in zip(axes, panels):
        ax.set_facecolor("#d9d9d9")
        mesh = ax.pcolormesh(lat, sigma2, np.ma.masked_invalid(field) / 1e6,
                             cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                             shading="nearest", rasterized=True)
        ax.axvline(stitch_lat, color="k", ls="--", lw=0.7)
        ax.set_xlim(lat_target.min(), lat_target.max())
        ax.set_ylim(sig_lo, sig_hi)
        ax.invert_yaxis()
        ax.set_ylabel(r"$\sigma_2$ (kg m$^{-3}$)")
        ax.set_title(title, loc="left")
    axes[1].set_xlabel("Latitude")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.set_title("Sv", pad=6)
    save_figure(fig, out_stem, formats=("png",))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-e", "--experiment", default=EXPERIMENT,
                        help="CMIP experiment folder under the data root")
    parser.add_argument("-r", "--realizations", default=REALIZATIONS,
                        help="realization range, e.g. 1-40 or 36-40")
    parser.add_argument("-m", "--model", default=MODEL,
                        help="CMIP source_id in the file names "
                             "('' = auto-detect from the folder)")
    parser.add_argument("-t", "--target-grid", default=TARGET_GRID,
                        help="file stem providing rho2_full target levels "
                             "(for cross-model tests)")
    parser.add_argument("-x", "--experiment-id", default=EXPERIMENT_ID,
                        help="filename experiment filter, e.g. ssp245")
    parser.add_argument("-o", "--output-experiment", default=OUTPUT_EXPERIMENT,
                        help="interim output label; default = source experiment")
    parser.add_argument("--amoc-only", action="store_true", default=AMOC_ONLY,
                        help="use only the Atlantic-Arctic sector (Southern "
                             "Ocean left NaN); for models with an empty global sector")
    parser.add_argument("--amoc-basin", type=int, default=AMOC_BASIN_INDEX,
                        help="Atlantic-Arctic sector index in msftmrho")
    return parser.parse_args()


def main(experiment: str = EXPERIMENT, realizations_spec: str = REALIZATIONS,
         target_grid: str = TARGET_GRID,
         experiment_id: str = EXPERIMENT_ID, amoc_only: bool = AMOC_ONLY,
         amoc_basin: int = AMOC_BASIN_INDEX,
         output_experiment: str = OUTPUT_EXPERIMENT,
         model: str = MODEL) -> None:
    import netCDF4

    data_dir = require_dir(cmip_raw_dir(experiment), "Raw CMIP experiment data")
    output_experiment = output_experiment.strip() or experiment
    out_dir = ensure_dir(cmip_interim_dir(output_experiment))

    # Native density levels are valid only for the training model.
    if not model:
        model = detect_model(data_dir)
        print(f"model auto-detected from {experiment}: {model}")
    if model != REFERENCE_MODEL and not target_grid:
        raise RuntimeError(
            f"{experiment} holds {model}, not the training model "
            f"{REFERENCE_MODEL}; cross-model runs require --target-grid "
            "<ACCESS MOC stem> (the density levels the trained network "
            "predicts on)")

    lat_target = load_target_latitudes()[TARGET_LAT_SLICE]

    rho_target_fixed = None
    seam_lat_fixed = None  # training model's AMOC south edge (target-grid seam)
    target_path = Path(target_grid) if target_grid else None
    if target_path and target_path.exists():
        target = load_npz_or_mat(target_path)
        rho_target_fixed = np.asarray(target["rho2_full"]).squeeze()
        target_lat = np.asarray(target["lat_psi"]).squeeze()
        if not np.array_equal(target_lat, lat_target):
            raise RuntimeError(
                f"Target-grid latitudes in {target_path} do not match the configured gr grid")
        if "amoc_south_edge_lat" in target:
            seam_lat_fixed = float(np.asarray(target["amoc_south_edge_lat"]).squeeze())
        print(f"target density levels from {target_path}: "
              f"{rho_target_fixed.size} levels, "
              f"{rho_target_fixed.min():.2f}..{rho_target_fixed.max():.2f}")
    elif target_path and model != REFERENCE_MODEL:
        raise FileNotFoundError(
            f"Target MOC grid {target_path} does not exist. Run stage 05 for "
            f"{REFERENCE_MODEL} first to create the immutable reference grid.")
    elif target_path:
        print(f"target grid {target_path} will be created from {REFERENCE_MODEL}")

    # Match msftmrho files for the selected model only.
    pattern = (f"{PSI_NAME}_Omon_{model}_{experiment_id}*.nc" if experiment_id
               else f"{PSI_NAME}_Omon_{model}*.nc")
    files_by_r = find_realization_files(data_dir, pattern)
    if not files_by_r:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")

    realizations = parse_realizations(realizations_spec)
    outputs = []
    for r in realizations:
        files = files_by_r.get(r)
        if not files:
            raise FileNotFoundError(f"No {PSI_NAME} file for realization r{r}")

        psi_parts, time_parts, month_parts = [], [], []
        time_units_seen, time_calendars_seen = set(), set()
        for path in files:
            print(f"r{r}: {path.name}")
            with netCDF4.Dataset(path) as nc:
                psi = np.ma.filled(nc[PSI_NAME][:], np.nan)  # [time, basin, rho, lat]
                time_values = np.asarray(nc["time"][:])
                time_parts.append(time_values)
                time_units = getattr(nc["time"], "units", "")
                time_calendar = getattr(nc["time"], "calendar", "standard")
                if not time_units:
                    raise RuntimeError(f"{path}: time coordinate has no units")
                time_units_seen.add(time_units)
                time_calendars_seen.add(time_calendar)
                dates = netCDF4.num2date(
                    time_values, units=time_units, calendar=time_calendar,
                    only_use_cftime_datetimes=True)
                month_parts.append(np.asarray(
                    [f"{d.year:04d}-{d.month:02d}" for d in dates], dtype="U7"))
                rho2 = np.asarray(nc["rho"][:])
                lat_src = np.asarray(nc["lat"][:])
            psi_parts.append(psi)
        psi = np.concatenate(psi_parts, axis=0) / RHO_CONST  # kg/s -> m^3/s
        time = np.concatenate(time_parts)
        time_month = np.concatenate(month_parts)
        if len(time_units_seen) != 1 or len(time_calendars_seen) != 1:
            raise RuntimeError(
                f"r{r}: split MOC files use inconsistent time metadata: "
                f"units={sorted(time_units_seen)}, "
                f"calendars={sorted(time_calendars_seen)}"
            )
        time_units = next(iter(time_units_seen))
        time_calendar = next(iter(time_calendars_seen))
        month_number = np.asarray([
            int(value[:4]) * 12 + int(value[5:7]) - 1 for value in time_month])
        if np.any(np.diff(month_number) != 1):
            raise RuntimeError(
                f"Non-contiguous or duplicate monthly time for realization r{r}")

        if rho2.max() < 1000:  # coordinate stored as sigma2, not absolute density
            rho2 = rho2 + 1000.0
            print("  density coordinate offset by +1000 (sigma2 -> absolute)")

        psi_amoc = psi[:, amoc_basin]     # [time, rho, lat]
        sep = find_separation_index(psi_amoc)
        own_edge = float(lat_src[sep])    # this model's native AMOC south edge

        psi_somoc = None
        if not amoc_only:
            psi_somoc = psi[:, GLOBAL_BASIN_INDEX]
            if not np.any(np.nan_to_num(psi_somoc) != 0):
                raise RuntimeError(
                    "The global_ocean msftmrho sector is empty or identically "
                    "zero (as published by NorESM2); rerun with --amoc-only "
                    "instead of stitching zeros in as Southern Ocean truth.")
        del psi

        # Use the training-model seam for cross-model comparisons.
        if seam_lat_fixed is not None:
            seam_lat = seam_lat_fixed
            seam_source = "target grid (training-model edge)"
        elif model == REFERENCE_MODEL or not target_grid:
            seam_lat = own_edge
            seam_source = "own sector edge"
        else:
            raise RuntimeError(
                f"Target grid {target_path} records no amoc_south_edge_lat; "
                f"regenerate it by rerunning stage 05 for {REFERENCE_MODEL} "
                "(one realization suffices) before cross-model runs.")
        amoc_rows = lat_target >= seam_lat   # AMOC rows of the evaluation grid
        print(f"  native AMOC south edge {own_edge:.2f} | seam on target grid: "
              f"AMOC for lat >= {seam_lat:.2f} [{seam_source}]")

        # Interpolate each sector before stitching on the target grid.
        if rho_target_fixed is not None:
            rho_target = rho_target_fixed
        else:
            rho_target = rho2[(rho2 > RHO_RANGE[0]) & (rho2 < RHO_RANGE[1])]
        rho_idx = nearest_indices(rho2, rho_target)
        lat_idx = nearest_indices(lat_src, lat_target)

        psi_interp = psi_amoc[:, rho_idx][:, :, lat_idx]  # [time, lev, lat]
        if amoc_only:
            # Exclude the Southern Ocean sector without global MOC truth.
            psi_interp[:, :, ~amoc_rows] = np.nan
        else:
            somoc_interp = psi_somoc[:, rho_idx][:, :, lat_idx]
            psi_interp = np.where(amoc_rows[None, None, :],
                                  psi_interp, somoc_interp)

        if target_path and not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target_path, rho2_full=rho_target, lat_psi=lat_target,
                     amoc_south_edge_lat=own_edge,
                     source_model=model, source_experiment=experiment)
            print("  saved immutable target grid", target_path)
        elif (target_path and seam_lat_fixed is None
              and model == REFERENCE_MODEL):
            # migrate a pre-existing target grid: record the training seam
            existing = dict(np.load(target_path))
            existing["amoc_south_edge_lat"] = own_edge
            np.savez(target_path, **existing)
            seam_lat_fixed = own_edge
            print(f"  added amoc_south_edge_lat={own_edge:.2f} to {target_path.name}")

        out_file = out_dir / f"FullDepth_ASMOC_interp_gr_r{r}.npz"
        np.savez(out_file, psi=psi_interp, rho2=rho_target, lat=lat_target,
                 time_days=time, time_units=time_units, time_calendar=time_calendar,
                 time_month=time_month,
                 source_model=model, source_experiment=experiment)
        print("  saved", out_file.name, psi_interp.shape)
        outputs.append(out_file)

        if QC_PLOTS and r == parse_realizations(realizations_spec)[0]:
            apply_style()
            if psi_somoc is None:
                psi_before = psi_amoc
            else:  # native-edge stitch, for the before-interpolation panel only
                psi_before = np.concatenate(
                    [psi_somoc[:, :, :sep], psi_amoc[:, :, sep:]], axis=2)
            qc_plot(psi_before.mean(axis=0), rho2, lat_src,
                    psi_interp.mean(axis=0), rho_target, lat_target,
                    seam_lat, out_file.with_suffix(""))
            del psi_before



if __name__ == "__main__":
    args = parse_args()
    main(args.experiment, args.realizations, args.target_grid,
         args.experiment_id, args.amoc_only, args.amoc_basin,
         args.output_experiment, args.model)
