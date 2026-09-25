"""Stage 02: define the JPL GRACE mascon geometry used for all NN inputs.

Identifies land-free mascons between 75S and 64.5N in the JPL RL06.3M
solution, computes each mascon's bounding box, and tags each mascon with a
basin id (1 = Atlantic + Southern Ocean) from the stage-01 masks.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import BASINMASK_DIR, GRACE_MASCON_NC
from neurmoc.grids import assign_basin_ids, build_mascon_geometry
from neurmoc.io_utils import ensure_dir, load_npz_or_mat, require_file
from neurmoc.plotting.style import apply_style, save_figure

# ========== User settings ==========
LAT_MIN, LAT_MAX = -75.0, 64.5
QC_PLOTS = True


def load_grace_grid(path):
    """Mascon-ID and land-mask grids, longitude recentered to -180..180."""
    import netCDF4

    with netCDF4.Dataset(require_file(path, "GRACE mascon NetCDF")) as nc:
        lon = np.asarray(nc["lon"][:])
        lat = np.asarray(nc["lat"][:])
        mascon_id = np.asarray(nc["mascon_ID"][:])   # [lat, lon]
        land_mask = np.asarray(nc["land_mask"][:])

    n_half = lon.size // 2
    lon = np.concatenate([lon[n_half:], lon[:n_half]])
    lon[lon > 180] -= 360
    mascon_id = np.concatenate([mascon_id[:, n_half:], mascon_id[:, :n_half]], axis=1)
    land_mask = np.concatenate([land_mask[:, n_half:], land_mask[:, :n_half]], axis=1)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lon_grid, lat_grid, mascon_id, land_mask


def main() -> None:
    apply_style()
    out_dir = ensure_dir(BASINMASK_DIR)

    lon_grid, lat_grid, mascon_id, land_mask = load_grace_grid(GRACE_MASCON_NC)
    geometry = build_mascon_geometry(
        mascon_id, lon_grid, lat_grid, land_mask, lat_min=LAT_MIN, lat_max=LAT_MAX
    )
    print(f"{geometry.n_mascons} ocean mascons between {LAT_MIN} and {LAT_MAX} deg")

    masks = load_npz_or_mat(out_dir / "BasinMasks_gr_V2026", ["X", "Y", "MaskAtlSO"])
    geometry.basin_id = assign_basin_ids(
        geometry, np.asarray(masks["MaskAtlSO"]), np.asarray(masks["X"]), np.asarray(masks["Y"])
    )
    n_atl = int(np.nansum(geometry.basin_id == 1))
    print(f"{n_atl} mascons in the Atlantic + Southern Ocean")

    out_file = out_dir / "Mascon_AtlSO.npz"
    payload = geometry.to_dict()
    # Exclude polar mascons outside the analysis domain.
    mascon_id_banded = np.where(
        (lat_grid < LAT_MIN) | (lat_grid > LAT_MAX), np.nan, mascon_id)
    payload.update({
        "lon_mascon": lon_grid, "lat_mascon": lat_grid,
        "mascon_ID": mascon_id_banded, "Nmascon": geometry.n_mascons,
    })
    np.savez(out_file, **payload)
    print("saved", out_file)

    if QC_PLOTS:
        fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
        land = np.where(land_mask > 0.5, 1.0, np.nan)
        ax.pcolormesh(lon_grid, lat_grid, land, cmap=ListedColormap(["#d9d9d9"]),
                      shading="nearest", rasterized=True)
        atlso = geometry.basin_id == 1
        ax.scatter(geometry.lon_center[~atlso], geometry.lat_center[~atlso],
                   s=3, color="#c6dbef", linewidths=0, label="other ocean mascon")
        ax.scatter(geometry.lon_center[atlso], geometry.lat_center[atlso],
                   s=3, color="#1f78b4", linewidths=0,
                   label="Atlantic + Southern Ocean mascon")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 80)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Ocean mascon centers ({LAT_MIN}\N{DEGREE SIGN} to {LAT_MAX}\N{DEGREE SIGN})", loc="left")
        ax.legend(loc="lower left", framealpha=0.9, markerscale=3)
        save_figure(fig, out_dir / "Mascon_basin_ids", formats=("png",))


if __name__ == "__main__":
    main()
