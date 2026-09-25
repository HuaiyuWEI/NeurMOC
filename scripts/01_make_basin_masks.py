"""Stage 01: basin masks on the CMIP6 1-degree "gr" grid.

Reads the GFDL-CM4 depth and sea-area-fraction files, closes off marginal
seas and inter-basin passages, flood-fills the Atlantic and Indo-Pacific,
and saves the masks (1 = Atlantic, 2 = Indo-Pacific, 3 = Southern Ocean)
used by all later stages.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import BASINMASK_DIR, CM4_GR_DIR
from neurmoc.grids import block_line, flood_fill, nearest_cell
from neurmoc.io_utils import ensure_dir, require_dir
from neurmoc.plotting.style import apply_style, save_figure

# ========== User settings ==========
SEA_FRACTION_MIN = 50.0  # % ocean required to count a cell as water
QC_PLOTS = True


LAND_COLOR = "#d9d9d9"


def qc_panel(ax, LON, LAT, coded, title, colors, labels):
    """One categorical QC map: `coded` holds values 1..len(colors), NaN = land."""
    ax.set_facecolor(LAND_COLOR)  # NaN cells are transparent -> land shows in gray
    ax.pcolormesh(LON, LAT, coded, cmap=ListedColormap(colors),
                  vmin=0.5, vmax=len(colors) + 0.5, shading="nearest",
                  rasterized=True)
    ax.set_ylim(-80, 80)
    ax.set_title(title, loc="left")
    handles = [Patch(facecolor=c, label=lb) for c, lb in zip(colors, labels)]
    handles.append(Patch(facecolor=LAND_COLOR, label="land"))
    ax.legend(handles=handles, loc="lower left", framealpha=0.9, handlelength=1.2)


def read_gr_field(data_dir: Path, pattern: str, varname: str):
    """Read a 2-D field from the single file matching `pattern`; recenter Atlantic."""
    import netCDF4

    files = sorted(data_dir.glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern} in {data_dir}, found {len(files)}")
    with netCDF4.Dataset(files[0]) as nc:
        lat = np.asarray(nc["lat"][:])
        lon = np.asarray(nc["lon"][:])
        field = np.ma.filled(nc[varname][:], np.nan)  # [lat, lon]

    # convert 0..360 to -180..180 with the Atlantic in the center
    lon = np.concatenate([lon[180:] - 360, lon[:180]])
    field = np.concatenate([field[:, 180:], field[:, :180]], axis=1)
    return lon, lat, field.T  # [lon, lat]


def main() -> None:
    apply_style()
    data_dir = require_dir(CM4_GR_DIR, "GFDL CM4 gr-grid data")
    out_dir = ensure_dir(BASINMASK_DIR)

    lon, lat, depth = read_gr_field(data_dir, "*deptho*gr.nc", "deptho")
    _, _, sea_frac = read_gr_field(data_dir, "*sftof*gr.nc", "sftof")
    LON, LAT = np.meshgrid(lon, lat, indexing="ij")

    # cells that are mostly land are treated as land
    wet = sea_frac.copy()
    wet[wet <= SEA_FRACTION_MIN] = np.nan
    wet_open = wet.copy()  # QC snapshot: before any passage is closed

    # ---- close off passages so basins do not leak into each other ----
    # west side of the Labrador Sea
    i, j = nearest_cell(LON, LAT, -77, 64)
    wet[i, j : j + 2] = np.nan
    # east side of the North Sea
    i, j = nearest_cell(LON, LAT, 10, 58)
    wet[i, j : j + 2] = np.nan
    # gap between North and South America
    block_line(wet, LON, LAT, np.arange(-82.5, -75.5, 0.01), 8.5)
    i, j = nearest_cell(LON, LAT, -83.5, 9.5)
    wet[i, j] = np.nan
    # Southern Ocean cuts: south of Africa (21.5E) and west of Drake Passage
    block_line(wet, LON, LAT, 21.5, np.arange(-33, -71, -0.1))
    block_line(wet, LON, LAT, -66.5, np.arange(-55, -67, -0.1))
    # Arctic Ocean north of 67.5N
    block_line(wet, LON, LAT, np.arange(-180, 180, 0.01), 67.5)

    blocked = np.isnan(wet)
    cut_cells = blocked & np.isfinite(wet_open)  # QC: cells the cuts removed

    # ---- flood fills ----
    mask_atl = flood_fill(blocked, nearest_cell(LON, LAT, -20, 0))
    mask_pac = flood_fill(blocked, nearest_cell(LON, LAT, -150, 0))
    mask_pac = np.maximum(mask_pac, flood_fill(blocked, nearest_cell(LON, LAT, 150, 0)))

    mask_all = mask_pac + mask_atl
    mask_all[mask_all != 1] = np.nan  # 1 where exactly one basin claims the cell
    mask_single = mask_all.copy()  # QC snapshot: before the basin labelling

    # ---- label: 1 Atlantic, 2 Indo-Pacific, 3 Southern Ocean ----
    so_boundary = np.flatnonzero(lat < -35)[-1]
    south = slice(0, so_boundary + 2)
    cols = mask_all[:, south]
    cols[~np.isnan(cols)] = 3
    mask_all[:, south] = cols

    # keep the Southern-Ocean boundary lines attached to the SO
    # Include both endpoints of each sector boundary.
    for lon0, lat_range in [(21.5, np.arange(-35, -69.05, -0.1)),
                            (-66.5, np.arange(-55, -65.05, -0.1))]:
        for lat0 in np.atleast_1d(lat_range):
            i, j = nearest_cell(LON, LAT, lon0, lat0)
            mask_all[i, j] = 3

    north = np.arange(so_boundary + 2, lat.size)
    for j in north:
        col = mask_all[:, j]
        col[mask_pac[:, j] == 1] = 2
        mask_all[:, j] = col

    mask_atl_so = mask_all.copy()
    mask_atl_so[mask_atl_so == 2] = np.nan
    mask_atl_so[mask_atl_so == 3] = 1

    mask_ocean = mask_all.copy()
    mask_ocean[np.isfinite(mask_ocean)] = 1

    out_file = out_dir / "BasinMasks_gr_V2026.npz"
    np.savez(out_file, X=LON, Y=LAT, MaskAtl=mask_atl, MaskPac=mask_pac,
             MaskAtlSO=mask_atl_so, MaskOcean=mask_ocean, MaskAll=mask_all)
    print("saved", out_file)

    if QC_PLOTS:
        # one panel per construction step, so every cut and fill can be checked
        wet_now = np.where(np.isfinite(wet), 1.0, np.nan)
        blue, red, atl_c, pac_c, so_c = (
            "#c6dbef", "#d62728", "#1f78b4", "#33a02c", "#ff7f00")
        steps = [
            (f"1. wet cells (sftof > {SEA_FRACTION_MIN:.0f}%)",
             np.where(np.isfinite(wet_open), 1.0, np.nan),
             [blue], ["wet cell"]),
            ("2. passages closed (Labrador, North Sea, Panama,\n"
             "    Africa 21.5\N{DEGREE SIGN}E, Drake 66.5\N{DEGREE SIGN}W, Arctic 67.5\N{DEGREE SIGN}N)",
             np.where(cut_cells, 2.0, wet_now),
             [blue, red], ["wet cell", "closed passage"]),
            ("3. Atlantic flood fill (seed 20\N{DEGREE SIGN}W, 0\N{DEGREE SIGN})",
             np.where(mask_atl == 1, 2.0, wet_now),
             [blue, atl_c], ["not reached", "Atlantic"]),
            ("4. Indo-Pacific flood fill (seeds 150\N{DEGREE SIGN}W / 150\N{DEGREE SIGN}E, 0\N{DEGREE SIGN})",
             np.where(mask_pac == 1, 2.0, wet_now),
             [blue, pac_c], ["not reached", "Indo-Pacific"]),
            ("5. keep cells claimed by exactly one basin",
             np.where(mask_single == 1, 2.0, wet_now),
             [blue, "#6a3d9a"], ["excluded", "kept"]),
            ("6. basin labels (Southern Ocean south of 35\N{DEGREE SIGN}S)",
             np.where(np.isfinite(mask_all), mask_all + 1, wet_now),
             [blue, atl_c, pac_c, so_c],
             ["excluded", "1 Atlantic", "2 Indo-Pacific", "3 Southern Ocean"]),
            ("7. Atlantic + Southern Ocean mask (NN domain)",
             np.where(mask_atl_so == 1, 2.0, wet_now),
             [blue, atl_c], ["excluded", "Atlantic + SO"]),
        ]
        nrows = (len(steps) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(11, 2.8 * nrows),
                                 constrained_layout=True)
        for ax, (title, coded, colors, labels) in zip(axes.flat, steps):
            qc_panel(ax, LON, LAT, coded, title, colors, labels)
        for ax in axes.flat[len(steps):]:
            ax.set_visible(False)
        save_figure(fig, out_dir / "BasinMasks_gr", formats=("png",))


if __name__ == "__main__":
    main()
