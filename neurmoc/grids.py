"""Mascon-grid construction, basin masks, and grid-to-mascon averaging.

Basin masks, mascon geometry, and NaN-aware sparse-matrix averaging of
`[time, ...]` fields onto mascons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import label as ndi_label
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Basin masks (stage A0)
# ---------------------------------------------------------------------------
def flood_fill(blocked: np.ndarray, seed_index: tuple[int, int]) -> np.ndarray:
    """Return a mask that is 1 over the region connected to `seed_index`.

    `blocked` is True/nonzero on land or artificially closed cells.
    Connectivity is 4-neighbour without longitude wrap-around; Pacific
    regions on either side of the seam require separate seeds.
    """
    open_cells = ~np.asarray(blocked, dtype=bool)
    labels, _ = ndi_label(open_cells)
    seed_label = labels[seed_index]
    if seed_label == 0:
        raise ValueError(f"Flood-fill seed {seed_index} sits on a blocked cell")
    return (labels == seed_label).astype(float)


def nearest_cell(lon_grid: np.ndarray, lat_grid: np.ndarray, lon0: float, lat0: float):
    """Index of the grid cell whose center is closest to (lon0, lat0)."""
    dist2 = (lon_grid - lon0) ** 2 + (lat_grid - lat0) ** 2
    return np.unravel_index(np.argmin(dist2), lon_grid.shape)


def block_line(mask: np.ndarray, lon_grid, lat_grid, lons, lats) -> None:
    """Mark the cells nearest to a sequence of (lon, lat) points as blocked.

    Used to close straits/passages (Drake Passage, south of Africa, ...) so
    that the flood fill cannot leak between basins. Modifies `mask` in place
    by setting the selected cells to NaN.
    """
    lons = np.atleast_1d(lons)
    lats = np.atleast_1d(lats)
    if lons.size == 1:
        lons = np.full_like(lats, lons[0], dtype=float)
    if lats.size == 1:
        lats = np.full_like(lons, lats[0], dtype=float)
    for lon0, lat0 in zip(lons, lats):
        i, j = nearest_cell(lon_grid, lat_grid, lon0, lat0)
        mask[i, j] = np.nan


# ---------------------------------------------------------------------------
# Mascon geometry (stage A1)
# ---------------------------------------------------------------------------
@dataclass
class MasconGeometry:
    """Bounding boxes and metadata of the ocean mascons used as NN inputs."""

    mascon_ids: np.ndarray          # [n] unique mascon IDs (ocean only)
    lon_bound1: np.ndarray          # [n] western bound (deg E)
    lon_bound2: np.ndarray          # [n] eastern bound (deg E)
    lat_bound1: np.ndarray          # [n] southern bound (deg N)
    lat_bound2: np.ndarray          # [n] northern bound (deg N)
    across_180: np.ndarray          # [n] bool, box crosses the date line
    lon_center: np.ndarray          # [n]
    lat_center: np.ndarray          # [n]
    basin_id: np.ndarray | None = None  # [n] 1 = Atlantic+SO, else other/NaN

    @property
    def n_mascons(self) -> int:
        return self.mascon_ids.size

    def to_dict(self) -> dict:
        out = {
            "mascon_ID_uniq": self.mascon_ids,
            "lon_mascon_bound1": self.lon_bound1,
            "lon_mascon_bound2": self.lon_bound2,
            "lat_mascon_bound1": self.lat_bound1,
            "lat_mascon_bound2": self.lat_bound2,
            "flag_across_180": self.across_180.astype(float),
            "lon_mascon_center": self.lon_center,
            "lat_mascon_center": self.lat_center,
        }
        if self.basin_id is not None:
            out["Basin_id"] = self.basin_id
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "MasconGeometry":
        def get(name):
            return np.asarray(data[name]).squeeze()

        basin = get("Basin_id") if "Basin_id" in data else None
        return cls(
            mascon_ids=get("mascon_ID_uniq"),
            lon_bound1=get("lon_mascon_bound1"),
            lon_bound2=get("lon_mascon_bound2"),
            lat_bound1=get("lat_mascon_bound1"),
            lat_bound2=get("lat_mascon_bound2"),
            across_180=get("flag_across_180").astype(bool),
            lon_center=get("lon_mascon_center"),
            lat_center=get("lat_mascon_center"),
            basin_id=basin,
        )


def build_mascon_geometry(
    mascon_id_grid: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    land_mask: np.ndarray,
    lat_min: float = -75.0,
    lat_max: float = 64.5,
    cell_half_width: float = 0.25,
) -> MasconGeometry:
    """Identify ocean-only mascons within a latitude band and box them.

    Mirrors A1_Mascon_grid_prep.m: mascons containing *any* land cell are
    discarded, polar mascons are dropped, and each mascon is described by
    its lon/lat bounding box (with a flag for boxes crossing the date line).
    """
    mid = mascon_id_grid.astype(float).copy()
    mid[(lat_grid < lat_min) | (lat_grid > lat_max)] = np.nan

    # The land check and bounding boxes consider
    # only the mascon's cells INSIDE the latitude band, so a coastal mascon
    # straddling the band edge is judged (and boxed) by its in-band part
    ids = np.unique(mid[np.isfinite(mid)])
    keep = []
    for mascon_id in ids:
        cells = mid == mascon_id
        if np.any(land_mask[cells] != 0):
            continue
        keep.append(mascon_id)
    ids = np.asarray(keep)

    n = ids.size
    lon_b1 = np.zeros(n)
    lon_b2 = np.zeros(n)
    lat_b1 = np.zeros(n)
    lat_b2 = np.zeros(n)
    across = np.zeros(n, dtype=bool)

    for k, mascon_id in enumerate(ids):
        cells = mid == mascon_id
        lons = lon_grid[cells]
        lats = lat_grid[cells]
        if np.any(lons > 0) and np.any(lons < 0):
            # crossing either the prime meridian (contiguous) or the date line
            if lons.max() - lons.min() > 180:  # date line
                lon_b1[k] = lons[lons > 0].min() - cell_half_width
                lon_b2[k] = lons[lons < 0].max() + cell_half_width
                across[k] = True
            else:
                lon_b1[k] = lons.min() - cell_half_width
                lon_b2[k] = lons.max() + cell_half_width
        else:
            lon_b1[k] = lons.min() - cell_half_width
            lon_b2[k] = lons.max() + cell_half_width
        lat_b1[k] = lats.min() - cell_half_width
        lat_b2[k] = lats.max() + cell_half_width

    lon_c = 0.5 * (lon_b1 + lon_b2)
    lat_c = 0.5 * (lat_b1 + lat_b2)
    lon_c[across] = lon_c[across] + 180.0

    return MasconGeometry(
        mascon_ids=ids,
        lon_bound1=lon_b1,
        lon_bound2=lon_b2,
        lat_bound1=lat_b1,
        lat_bound2=lat_b2,
        across_180=across,
        lon_center=lon_c,
        lat_center=lat_c,
    )


def load_grace_land(mascon_nc_path):
    """(lon, lat, land) from the JPL mascon file for gray QC backdrops.

    Longitude recentered to -180..180 (Atlantic centered); land is 1 over
    land and NaN over ocean, ready for a single-color pcolormesh.
    """
    import netCDF4

    with netCDF4.Dataset(mascon_nc_path) as nc:
        lon = np.asarray(nc["lon"][:])
        lat = np.asarray(nc["lat"][:])
        land = np.asarray(nc["land_mask"][:])
    n_half = lon.size // 2
    lon = np.concatenate([lon[n_half:] - 360, lon[:n_half]])
    land = np.concatenate([land[:, n_half:], land[:, :n_half]], axis=1)
    return lon, lat, np.where(land > 0.5, 1.0, np.nan)


def assign_basin_ids(
    geometry: MasconGeometry,
    basin_mask: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
) -> np.ndarray:
    """Basin id of each mascon = mask value at the nearest basin-grid cell."""
    tree = cKDTree(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))
    _, idx = tree.query(np.column_stack([geometry.lon_center, geometry.lat_center]))
    return basin_mask.ravel()[idx]


def cell_area_weights(lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    """Approximate cell areas (arbitrary units) from cell-center coordinates.

    area ~ |J| * cos(lat), with the Jacobian J of (lon, lat) with respect to
    the grid indices estimated by central differences. Exact for rectilinear
    grids up to a constant factor (which cancels in weighted means); a good
    approximation on curvilinear grids. Used where no areacello exists
    (atmosphere grids, observational products, ECCO tiles).
    """
    lon = np.asarray(lon_grid, dtype=float)
    lat = np.asarray(lat_grid, dtype=float)
    if lon.ndim != 2 or lon.shape != lat.shape:
        raise ValueError("cell_area_weights needs matching 2-D center grids")

    # Unwrap both axes before differentiating across longitude seams.
    dlon_0 = np.gradient(np.unwrap(lon, period=360.0, axis=0), axis=0)
    dlon_1 = np.gradient(np.unwrap(lon, period=360.0, axis=1), axis=1)
    dlat_0 = np.gradient(lat, axis=0)
    dlat_1 = np.gradient(lat, axis=1)
    jacobian = np.abs(dlon_0 * dlat_1 - dlon_1 * dlat_0)
    return jacobian * np.cos(np.deg2rad(lat))


# ---------------------------------------------------------------------------
# Grid-to-mascon averaging (averageFieldToMascons.m)
# ---------------------------------------------------------------------------
class MasconAverager:
    """NaN-aware grid-to-mascon averaging with optional cell-area weights.

    The trailing dimensions of each input must match the source grid.
    Leading dimensions, such as time, are preserved.
    """

    def __init__(self, lon_grid: np.ndarray, lat_grid: np.ndarray,
                 geometry: MasconGeometry,
                 weights: np.ndarray | None = None):
        if lon_grid.shape != lat_grid.shape:
            raise ValueError("lon_grid and lat_grid must have the same shape")
        self.grid_shape = lon_grid.shape
        self.weighted = weights is not None
        w = self._check_weights(weights, lon_grid.shape)

        lon = lon_grid.ravel()
        lat = lat_grid.ravel()
        rows, cols = [], []
        for k in range(geometry.n_mascons):
            in_lat = (lat >= geometry.lat_bound1[k]) & (lat < geometry.lat_bound2[k])
            if geometry.across_180[k]:
                in_lon = (lon >= geometry.lon_bound1[k]) | (lon < geometry.lon_bound2[k])
            else:
                in_lon = (lon >= geometry.lon_bound1[k]) & (lon < geometry.lon_bound2[k])
            idx = np.flatnonzero(in_lon & in_lat)
            rows.append(np.full(idx.size, k))
            cols.append(idx)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        self.membership = sparse.csr_matrix(
            (w[cols], (rows, cols)),
            shape=(geometry.n_mascons, lon.size),
        )
        self.n_mascons = geometry.n_mascons

    @staticmethod
    def _check_weights(weights: np.ndarray | None, grid_shape) -> np.ndarray:
        """Validated flat weight vector (ones when unweighted)."""
        if weights is None:
            return np.ones(int(np.prod(grid_shape)))
        w = np.asarray(weights, dtype=float)
        if w.shape != tuple(grid_shape):
            raise ValueError(
                f"weights shape {w.shape} does not match the grid {grid_shape}")
        w = w.ravel()
        # areacello masks land with NaN/fill; those cells carry no weight
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if not w.any():
            raise ValueError("weights are all zero/invalid")
        return w

    @classmethod
    def from_mascon_ids(cls, mascon_id_grid: np.ndarray,
                        mascon_ids: np.ndarray,
                        weights: np.ndarray | None = None) -> "MasconAverager":
        """Average over the exact cells of each mascon ID (native 0.5-deg grid)."""
        ids = np.asarray(mascon_ids).ravel()
        flat_ids = mascon_id_grid.ravel()
        w = cls._check_weights(weights, mascon_id_grid.shape)
        rows, cols = [], []
        for k, mascon_id in enumerate(ids):
            idx = np.flatnonzero(flat_ids == mascon_id)
            rows.append(np.full(idx.size, k))
            cols.append(idx)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)

        averager = cls.__new__(cls)
        averager.grid_shape = mascon_id_grid.shape
        averager.weighted = weights is not None
        averager.membership = sparse.csr_matrix(
            (w[cols], (rows, cols)), shape=(ids.size, flat_ids.size)
        )
        averager.n_mascons = ids.size
        return averager

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """Weighted average `field[..., *grid_shape]` -> `[..., n_mascons]`.

        NaN cells drop out per time step; the remaining weights renormalize
        (with unit weights this is the NaN-aware cell mean).
        """
        ndim_grid = len(self.grid_shape)
        if field.shape[-ndim_grid:] != self.grid_shape:
            raise ValueError(
                f"Trailing dims {field.shape[-ndim_grid:]} do not match grid {self.grid_shape}"
            )
        lead_shape = field.shape[:-ndim_grid]
        flat = field.reshape(-1, int(np.prod(self.grid_shape)))

        valid = np.isfinite(flat)
        sums = self.membership @ np.where(valid, flat, 0.0).T          # [n_mascons, n_lead]
        weight_sums = self.membership @ valid.T.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / weight_sums
        means[weight_sums == 0] = np.nan
        return means.T.reshape(*lead_shape, self.n_mascons)


def mascon_values_to_grid(
    mascon_id_grid: np.ndarray, mascon_ids: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Paint per-mascon values back onto the 0.5-degree mascon-ID grid."""
    ids = np.asarray(mascon_ids).ravel()
    values = np.asarray(values).ravel()
    order = np.argsort(ids)
    ids_sorted = ids[order]
    values_sorted = values[order]

    flat_ids = np.nan_to_num(mascon_id_grid.ravel(), nan=-1.0)
    pos = np.searchsorted(ids_sorted, flat_ids)
    pos = np.clip(pos, 0, ids_sorted.size - 1)
    hit = ids_sorted[pos] == flat_ids

    out = np.full(flat_ids.shape, np.nan)
    out[hit] = values_sorted[pos[hit]]
    return out.reshape(mascon_id_grid.shape)
