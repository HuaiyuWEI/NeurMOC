"""Stage 16: propagate JPL GRACE mascon uncertainty to the reconstruction.

Monte Carlo perturbations use the reported marginal mascon uncertainties
and pass through the observation processing and trained network. The draws
assume independent native errors; they do not represent the full spatial
or temporal GRACE error covariance. A CSR run uses JPL uncertainty as an
explicit proxy. Run with --help for product and simulation options.
"""

# %% [1] User settings and imports --------------------------------------------
import argparse
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import (
    ACTIVE_RUN_ID,
    BASELINE_YEARS,
    BASINMASK_DIR,
    CSR_MASCON_NC,
    GRACE_MASCON_NC,
    LPF_OBS,
    OBS_MASCON_ROOT,
    SATELLITE_DATASET_ID,
    mascon_var,
)
from neurmoc.evaluation import (
    BASELINE_EXPERIMENT,
    COVARIATES_ALL,
    PERF_ROOT,
    REALWORLD_ROOT,
    model_dir,
)
from neurmoc.export import save_realworld_trend_npz
from neurmoc.filtering import lowpass
from neurmoc.grids import MasconGeometry
from neurmoc.inference import TrainedEnsemble
from neurmoc.io_utils import load_npz_or_mat, require_file
from neurmoc.moc_utils import find_nearest_index, robust_trend, unflatten
from neurmoc.naming import lpf_tag_from_name, prepare_covariate_config
from neurmoc.plotting import (
    CMAP_AMPLITUDE,
    apply_style,
    section_row,
    shade_gap,
)
from neurmoc.plotting.style import COLORS, TEXT_BBOX, save_figure
from neurmoc.satellite_products import (
    DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC,
    uncertainty_product_registry,
)
from neurmoc.results import exact_common_month_indices, load_real_world
from neurmoc.timeaxis import decimal_year

# ========== User settings ==========
# Monte Carlo draws per mode; each draw runs the network ensemble.
DRAWS = 500
#: spatial error structure: "all" | "independent" | "correlated" | "coherent"
MODE = "independent"
# Independent perturbations define the default budget term.
PRODUCTION_MODE = "independent"
#: e-folding length of the "correlated" mode, km (the JPL caps are ~330 km)
CORR_KM = 500.0
SEED = 0
#: trained network to price. None -> the profile baseline / env override.
EXPERIMENT = None
COVARIATES = None                 # None -> all three inputs
# For CSR reconstructions, use JPL uncertainty magnitudes as a proxy.
OBP_SOURCE = "GRACE"              # "GRACE" | "GRACE_CSR"
#: Reuse cached intermediate results during interactive reruns.
REUSE_CACHED = True
#: draw-progress print interval (1 = every draw)
PROGRESS_EVERY = 10
# Optional output suffix keeps exploratory runs separate from default products.
OUT_TAG = ""
# Tagged Stage-15 budgets require a tagged output.
BUDGET_TAG = ""
# ===================================

# Convert JPL equivalent-water-height uncertainty using 1000 kg m^-3,
# including when that uncertainty is used as a proxy for CSR.
GRAVITY_TO_PA = 9.806 * 1000.0      # water-equivalent m -> Pa (JPL, rho=1000)
POLAR_BAND = (-75.0, 64.5)          # mascon band of the input domain
EDGE_MONTHS = 12                    # low-pass-filter edge trim
EARTH_R_KM = 6371.0

_cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_cli.add_argument("-n", "--draws", type=int, default=None,
                  help=f"realizations per mode (User setting: {DRAWS})")
_cli.add_argument("--mode", default=None,
                  choices=["all", "independent", "correlated", "coherent"],
                  help=f"spatial error structure (User setting: {MODE})")
_cli.add_argument("--corr-km", type=float, default=None,
                  help="e-folding length of the 'correlated' mode (km)")
_cli.add_argument("--seed", type=int, default=None)
_cli.add_argument("-E", "--experiment", default=None,
                  help="trained experiment folder (default: profile baseline)")
_cli.add_argument("-x", "--covariates", default=None,
                  help="comma/plus-separated input set (default: all three)")
_cli.add_argument(
    "--obp-source", default=None, choices=["GRACE", "GRACE_CSR"],
    help=("OBP point estimate to perturb: GRACE=JPL (default) or "
          "GRACE_CSR. CSR uses the JPL uncertainty field as an explicit "
          "measurement-noise proxy and writes a separate *_obpCSR file"),
)
_cli.add_argument("--out-tag", default=None,
                  help="suffix for the output npz - USE for exploratory "
                       "runs so the production budget is not overwritten")
_cli.add_argument("--budget-tag", default=None,
                  help="Stage-15 trend_error_budget suffix to combine; "
                       "independent of --out-tag")
_args = _cli.parse_known_args()[0]          # tolerate interactive environments
#: the CLI wins over the User settings block, which wins over the defaults
DRAWS = _args.draws if _args.draws is not None else DRAWS
MODE = _args.mode or MODE
CORR_KM = _args.corr_km if _args.corr_km is not None else CORR_KM
SEED = _args.seed if _args.seed is not None else SEED
if DRAWS < 2:
    raise SystemExit("DRAWS must be at least 2 (the spread across draws IS "
                     "the estimate)")
MODES = (["independent", "correlated", "coherent"] if MODE == "all"
         else [MODE])
COVARIATES = (_args.covariates or COVARIATES
              or os.environ.get("NEURMOC_FIG_COVARIATES")
              or COVARIATES_ALL).replace(",", "+")
EXPERIMENT = (_args.experiment or EXPERIMENT
              or os.environ.get("NEURMOC_FIG_EXPERIMENT")
              or BASELINE_EXPERIMENT)
OBP_SOURCE = _args.obp_source or OBP_SOURCE
RECONSTRUCTION_PRODUCT_TAG = "" if OBP_SOURCE == "GRACE" else "_obpCSR"
RECONSTRUCTION_OBP_LABEL = "JPL" if OBP_SOURCE == "GRACE" else "CSR"
_COVARIATE_SET = set(COVARIATES.split("+"))
# These fields describe the inputs that actually enter the selected network.
# Unused axes are empty, matching Stage 15's exact product registry.
RECONSTRUCTION_OBP_SOURCE = (
    OBP_SOURCE if mascon_var("obp") in _COVARIATE_SET else ""
)
RECONSTRUCTION_SSH_SOURCE = (
    "DUACS" if mascon_var("ssh") in _COVARIATE_SET else ""
)
RECONSTRUCTION_WIND_SOURCE = (
    "CCMP" if mascon_var("uas") in _COVARIATE_SET else ""
)
MEASUREMENT_NOISE_IS_PROXY = OBP_SOURCE == "GRACE_CSR"
MEASUREMENT_NOISE_SOURCE = "JPL RL06.3 mascon `uncertainty`"
MEASUREMENT_NOISE_DEFINITION = (
    "JPL's empirically calibrated ocean-OBP uncertainty, interpolated to "
    "the CSR native solution epochs and propagated around the exact "
    "selected CSR point estimate; the CSR RL0603 granule contains no "
    "uncertainty variable, so this is a proxy rather than a CSR formal error"
    if MEASUREMENT_NOISE_IS_PROXY else
    "JPL's empirically calibrated ocean-OBP uncertainty propagated around "
    "the exact selected JPL point estimate"
)
NN_DIR = model_dir(root=REALWORLD_ROOT, experiment=EXPERIMENT,
                   covariates=COVARIATES)
SOURCE_NN_DIR = model_dir(root=PERF_ROOT, experiment=EXPERIMENT,
                          covariates=COVARIATES)
OUT_DIR = NN_DIR / "RealWorld"
OUT_TAG = _args.out_tag if _args.out_tag is not None else OUT_TAG
BUDGET_TAG = (_args.budget_tag if _args.budget_tag is not None else
              os.environ.get("NEURMOC_TREND_BUDGET_TAG", BUDGET_TAG)).strip()
for _name, _value in (("OUT_TAG", OUT_TAG), ("BUDGET_TAG", BUDGET_TAG)):
    if _value.startswith("_") or "/" in _value or "\\" in _value:
        raise SystemExit(
            f"{_name} must be a filename-safe tag without a leading "
            "underscore or path separators"
        )
if BUDGET_TAG and not OUT_TAG:
    raise SystemExit(
        "A tagged Stage-15 budget requires --out-tag so its combined trend "
        "product cannot overwrite production"
    )
# Non-default Monte Carlo settings require a separate output tag.
if not OUT_TAG:
    if PRODUCTION_MODE not in MODES:
        raise SystemExit(
            f"untagged production requires mode {PRODUCTION_MODE!r}; "
            "use --out-tag for a mode sensitivity"
        )
    if DRAWS < 500 or SEED != 0:
        raise SystemExit(
            "untagged production requires at least 500 draws and seed 0; "
            "use --out-tag for exploratory settings"
        )
# Product and optional run tags are included in output filenames.
_EXTRA_SUFFIX = f"_{OUT_TAG}" if OUT_TAG else ""
_SUFFIX = f"{RECONSTRUCTION_PRODUCT_TAG}{_EXTRA_SUFFIX}"
_BUDGET_SUFFIX = f"_{BUDGET_TAG}" if BUDGET_TAG else ""
OUT_FILE = OUT_DIR / f"grace_noise_budget{_SUFFIX}.npz"
budget_file = OUT_DIR / f"trend_error_budget{_BUDGET_SUFFIX}.npz"
if budget_file.is_file():
    try:
        with np.load(budget_file, allow_pickle=False) as _policy_handle:
            if "product_spread_include_gsfc" not in _policy_handle.files:
                raise KeyError("product_spread_include_gsfc")
            _early_include_gsfc = bool(
                np.asarray(
                    _policy_handle["product_spread_include_gsfc"]
                ).squeeze().item()
            )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"{budget_file.name} predates the explicit product-spread "
            "policy; rerun Stage 15"
        ) from exc
    if (
        not BUDGET_TAG
        and _early_include_gsfc != DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC
    ):
        raise SystemExit(
            f"untagged budget {budget_file.name} has "
            f"product_spread_include_gsfc={_early_include_gsfc}, but the "
            "scientific settings require "
            f"{DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC}; use the profile "
            "default or select tagged sensitivity budget/output files"
        )
else:
    # A combined trend product requires a matching Stage-15 budget.
    _early_include_gsfc = None
apply_style()

#: Cache intermediate results across interactive reruns when inputs match.
_CACHE = globals().get("_CACHE", {})


def cached(name: str, key, build, quiet: bool = False):
    """Return a cached value, rebuilding when ``key`` changes."""
    if REUSE_CACHED and _CACHE.get(name, (None,))[0] == key:
        if not quiet:
            print(f"  (reusing {name})")
        return _CACHE[name][1]
    value = build()
    _CACHE[name] = (key, value)
    return value


print("=== stage 16: GRACE measurement-noise Monte Carlo")
print(f"  draws {DRAWS} | modes {', '.join(MODES)}"
      + (f" | corr {CORR_KM:.0f} km" if "correlated" in MODES else "")
      + f" | seed {SEED}")
print(f"  network {EXPERIMENT}/{COVARIATES}")
print(f"  reconstruction OBP {RECONSTRUCTION_OBP_LABEL} ({OBP_SOURCE})"
      + (" | JPL uncertainty used as proxy" if OBP_SOURCE == "GRACE_CSR"
         else ""))
print(f"  Stage-15 budget tag: {BUDGET_TAG or '(production)'}")
print(f"  -> {OUT_FILE.name}")

# Reuse Stage-14 input assembly and its ordering/provenance checks.
_s14_spec = importlib.util.spec_from_file_location(
    "stage14", Path(__file__).resolve().parent / "14_reconstruct_real_world.py")
s14 = importlib.util.module_from_spec(_s14_spec)
_s14_spec.loader.exec_module(s14)
# Match one Stage-14 reconstruction; SSH and wind remain DUACS/CCMP.
s14.OBP_SOURCE = OBP_SOURCE
s14.SSH_SOURCE = "DUACS"
s14.USE_ERA5_WINDS = False


def shift_lon_halves(arr, axis):
    """Recenter a 0..360-longitude array to -180..180 (as stage 12)."""
    n = arr.shape[axis]
    return np.concatenate([np.take(arr, range(n // 2, n), axis=axis),
                           np.take(arr, range(0, n // 2), axis=axis)],
                          axis=axis)


# %% [2] The shipped sigma, on the pipeline's grid (SLOW - cached) -------------
def _read_granule():
    print("granule:", GRACE_MASCON_NC.name)
    with netCDF4.Dataset(require_file(GRACE_MASCON_NC,
                                      "JPL mascon NetCDF")) as nc:
        sigma = np.ma.filled(nc["uncertainty"][:], np.nan).astype(np.float32)
        jpl_days = np.asarray(nc["time"][:], dtype=float)
        lat_grid = np.asarray(nc["lat"][:])
        land = np.asarray(nc["land_mask"][:])
    sigma = shift_lon_halves(sigma, axis=2) / 100.0 * GRAVITY_TO_PA
    polar = (lat_grid < POLAR_BAND[0]) | (lat_grid > POLAR_BAND[1])
    sigma[:, polar, :] = np.nan
    return sigma, jpl_days, lat_grid, shift_lon_halves(land, axis=1) == 0


sigma_grid, jpl_days, lat_grid, ocean = cached(
    "granule", (str(GRACE_MASCON_NC), GRACE_MASCON_NC.stat().st_mtime_ns),
    _read_granule)

# %% [3] Independent-estimate check and cos-lat reduction (cached) -------------
# Draw one perturbation per distinct JPL mascon uncertainty series.
geom_data = load_npz_or_mat(BASINMASK_DIR / "Mascon_AtlSO")
geometry = MasconGeometry.from_dict(geom_data)
mascon_id_grid = np.asarray(geom_data["mascon_ID"])
atlso = geometry.basin_id == 1
net_ids = geometry.mascon_ids[atlso]
m_lon = geometry.lon_center[atlso]
m_lat = geometry.lat_center[atlso]
n_mascon = int(atlso.sum())


def _reduce_sigma():
    in_domain = (np.isin(mascon_id_grid, net_ids) & ocean
                 & np.isfinite(sigma_grid).all(axis=0))
    rows, cols = np.nonzero(in_domain)
    sig_cells = sigma_grid[:, rows, cols].T                  # [cell, sample]
    _, jpl_group = np.unique(np.ascontiguousarray(sig_cells).view(
        np.dtype((np.void, sig_cells.dtype.itemsize * sig_cells.shape[1]))
    ).ravel(), return_inverse=True)
    net_of_cell = mascon_id_grid[rows, cols]
    n_jpl = np.array([np.unique(jpl_group[net_of_cell == i]).size
                      for i in net_ids])
    print(f"  {rows.size} cells -> {jpl_group.max() + 1} JPL 3-deg mascons "
          f"-> {n_mascon} network mascons (JPL per network: "
          f"{n_jpl.min()}..{n_jpl.max()})")
    if n_jpl.max() > 1:
        print("  note: a network mascon spans several JPL mascons; each "
              "draw perturbs whole JPL mascons")
    w_cells = np.cos(np.deg2rad(lat_grid))[rows]
    out = np.empty((jpl_days.size, n_mascon))
    for k, mid in enumerate(net_ids):            # cos-lat weights, as stage 12
        sel = net_of_cell == mid
        w = w_cells[sel]
        out[:, k] = (w[:, None] * sig_cells[sel]).sum(axis=0) / w.sum()
    return out


sig_net_jpl = cached(
    "sig_net_jpl", (str(GRACE_MASCON_NC), n_mascon), _reduce_sigma
)


def _native_product_days() -> np.ndarray:
    """Native solution epochs used by the selected Stage-12 OBP product."""
    if OBP_SOURCE == "GRACE":
        return jpl_days.copy()
    with netCDF4.Dataset(require_file(CSR_MASCON_NC,
                                      "CSR mascon NetCDF")) as nc:
        raw_units = str(getattr(nc["time"], "units",
                                getattr(nc["time"], "Units", "")))
        if "days since 2002-01-01" not in raw_units:
            raise SystemExit(
                f"unexpected CSR time units {raw_units!r}; cannot align the "
                "JPL uncertainty proxy to CSR solution epochs"
            )
        return np.asarray(nc["time"][:], dtype=float)


native_days = cached(
    "native_product_days",
    (OBP_SOURCE,
     str(CSR_MASCON_NC) if OBP_SOURCE == "GRACE_CSR" else
     str(GRACE_MASCON_NC)),
    _native_product_days,
)
if np.any(np.diff(native_days) <= 0):
    raise SystemExit(f"{OBP_SOURCE} native solution epochs are not increasing")

# For CSR, interpolate JPL uncertainty magnitudes to CSR solution dates.
sig_net = interp1d(
    jpl_days, sig_net_jpl, axis=0, bounds_error=False, fill_value=np.nan
)(native_days)
if not np.isfinite(sig_net).all():
    raise SystemExit(
        f"{OBP_SOURCE} solution epochs extend beyond the JPL uncertainty "
        "record; the proxy cannot be extrapolated"
    )
print(f"  network-mascon 1-sigma: median {np.median(sig_net):.0f} Pa "
      f"on {native_days.size} native {RECONSTRUCTION_OBP_LABEL} epochs")

# %% [4] Spatial covariance factor for the correlated mode (cached) ------------
# Correlated mode uses an exponential great-circle covariance kernel.
def _chol_exponential(length_km: float) -> np.ndarray:
    print(f"  building {length_km:.0f} km covariance factor "
          f"({n_mascon}x{n_mascon})...")
    lon_r, lat_r = np.deg2rad(m_lon), np.deg2rad(m_lat)
    cosd = (np.sin(lat_r)[:, None] * np.sin(lat_r)[None, :]
            + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :]
            * np.cos(lon_r[:, None] - lon_r[None, :]))
    dist = EARTH_R_KM * np.arccos(np.clip(cosd, -1.0, 1.0))
    cov = np.exp(-dist / length_km)
    try:
        return np.linalg.cholesky(cov + 1e-8 * np.eye(n_mascon))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        return vecs * np.sqrt(np.clip(vals, 0.0, None))


def chol_for(length_km: float) -> np.ndarray:
    """Return a cached covariance factor for one correlation length."""
    return cached("chol", (length_km, n_mascon),
                  lambda: _chol_exponential(length_km), quiet=True)


def draw_mascon_noise(mode: str, rng) -> np.ndarray:
    """One realization on the selected native sample axis [sample, mascon]."""
    if mode == "coherent":
        z = rng.standard_normal((native_days.size, 1)) * np.ones(n_mascon)
    elif mode == "correlated":
        z = (rng.standard_normal((native_days.size, n_mascon))
             @ chol_for(CORR_KM).T)
    else:
        z = rng.standard_normal((native_days.size, n_mascon))
    return z * sig_net


# %% [5] The stage-12 treatment of the OBP record, applied to the noise --------
# Apply the Stage-12 interpolation, baseline, spatial demeaning, and filter.
t0 = np.datetime64("2002-01-01")
t_samples = t0 + native_days.astype("timedelta64[D]")
grid_months = np.arange(np.datetime64("2002-05"),
                        t_samples[-1].astype("datetime64[M]") + 1,
                        np.timedelta64(1, "M"))
t_monthly = np.concatenate([[t_samples[0]],
                            [np.datetime64(f"{m}-16") for m in grid_months]])
x_src = (t_samples - t0) / np.timedelta64(1, "D")
x_dst = (t_monthly - t0) / np.timedelta64(1, "D")
obp_months = t_monthly.astype("datetime64[M]")
_baseline_start = np.datetime64(f"{BASELINE_YEARS[0]}-01", "M")
_baseline_stop = np.datetime64(f"{BASELINE_YEARS[1]}-12", "M")
obp_baseline = ((obp_months >= _baseline_start)
                & (obp_months <= _baseline_stop))
_expected_baseline_months = 12 * (
    BASELINE_YEARS[1] - BASELINE_YEARS[0] + 1
)
if int(obp_baseline.sum()) != _expected_baseline_months:
    raise SystemExit(
        "JPL uncertainty axis does not contain the complete anomaly "
        f"baseline {_baseline_start}..{_baseline_stop}"
    )

# Require the saved product's month axis to match the reconstructed axis.
_obp_product_name = f"obp_{OBP_SOURCE}"
_obp_product_file = OBS_MASCON_ROOT / f"{_obp_product_name}.npz"
with np.load(require_file(_obp_product_file,
                          f"stage-12 {OBP_SOURCE} OBP product"),
             allow_pickle=False) as _p:
    _prod_months = np.asarray(_p["time_month"]).astype("datetime64[M]")
if not np.array_equal(_prod_months, obp_months):
    raise SystemExit(
        "the monthly axis rebuilt from the selected native epochs does not "
        f"match {_obp_product_file.name} "
        f"({obp_months[0]}..{obp_months[-1]} vs "
        f"{_prod_months[0]}..{_prod_months[-1]}). Rerun stage 12 against "
        "this source granule before pricing its noise.")


def pipeline_noise(mode: str, rng) -> np.ndarray:
    """Noise as the network sees it: [month, mascon] in the LPF frame."""
    native_noise = draw_mascon_noise(mode, rng)
    # Match each Stage-12 product's treatment of its final monthly node.
    fill_value = (
        np.nan if OBP_SOURCE == "GRACE"
        else (native_noise[0], native_noise[-1])
    )
    noise = interp1d(x_src, native_noise, axis=0, bounds_error=False,
                     fill_value=fill_value)(x_dst)
    if not np.isfinite(noise).all():
        # Avoid NaNs at the final node before temporal filtering.
        raise SystemExit(
            "interpolated noise is not finite - the monthly axis extends "
            f"past the last granule sample ({t_samples[-1]}). Stage 12 has "
            "the same exposure; fix both before pricing this granule.")
    # Re-reference each noise draw to its 2004-2009 per-mascon mean.
    noise = noise - noise[obp_baseline].mean(axis=0, keepdims=True)
    noise = noise - noise.mean(axis=1, keepdims=True)     # basin demean
    return lowpass(noise, LPF_OBS)                        # 2-yr Butterworth


# %% [6] Trained ensemble and the unperturbed inputs (SLOW - cached) -----------
lpf_tag = lpf_tag_from_name(EXPERIMENT)
covariates = prepare_covariate_config(COVARIATES.replace("+", ","))
_log = io.StringIO()
with redirect_stdout(_log):
    assembled = s14.assemble_inputs(covariates, lpf=bool(lpf_tag))
x0 = assembled.values
rec_months = np.asarray(assembled.time_month).astype("datetime64[M]")
names = list(assembled.covariate_names)
if mascon_var("obp") not in names:
    raise SystemExit(f"network {COVARIATES} has no OBP input - nothing to "
                     "perturb in this stage")
obp_block = names.index(mascon_var("obp")) * n_mascon
if x0.shape[1] != len(names) * n_mascon:
    raise SystemExit(f"input width {x0.shape[1]} is not {len(names)} blocks "
                     f"of {n_mascon} mascons")
#: rows of the OBP monthly axis that the reconstruction actually uses
if (rec_months[0] < obp_months[0]
        or rec_months[-1] > obp_months[-1]):
    raise SystemExit(
        "reconstruction months extend beyond the selected OBP/noise axis"
    )
take = np.searchsorted(obp_months, rec_months)
if (np.any(take >= obp_months.size)
        or not np.array_equal(obp_months[take], rec_months)):
    raise SystemExit("reconstruction months are not a subset of the OBP axis")

reference = load_npz_or_mat(s14.REFERENCE_GRID, ["rho2_full", "lat_psi"])
rho2 = np.asarray(reference["rho2_full"]).squeeze()
lat = np.asarray(reference["lat_psi"]).squeeze()
sigma2 = rho2 - 1000 if rho2[0] > 1000 else rho2
mask = np.asarray(load_npz_or_mat(SOURCE_NN_DIR / "Psi_mask",
                                  ["Psi_mask"])["Psi_mask"]).squeeze().astype(bool)
ensemble = cached("ensemble", str(SOURCE_NN_DIR),
                  lambda: TrainedEnsemble.load(SOURCE_NN_DIR))
n_members = ensemble.n_folds * ensemble.n_ensembles
print(f"  inputs {x0.shape}, {rec_months[0]}..{rec_months[-1]}, "
      f"{n_members} members")

# Use saved Stage-14 output and reference-state cell cores for validation.
with redirect_stdout(_log):                  # scenario-RMSE stub warning
    rw_full = load_real_world(
        NN_DIR, rmse_scenario=None, edge_months=EDGE_MONTHS,
        file_stem=f"Pred_RealWorld{RECONSTRUCTION_PRODUCT_TAG}",
    )
if rw_full.time_month is None:
    raise SystemExit(
        f"Pred_RealWorld{RECONSTRUCTION_PRODUCT_TAG} lacks the required "
        "calendar month axis"
    )
if rw_full.training_source_run_id != ACTIVE_RUN_ID:
    raise SystemExit(
        "Pred_RealWorld training source differs from the active profile: "
        f"{rw_full.training_source_run_id!r} != {ACTIVE_RUN_ID!r}"
    )
if tuple(rw_full.input_covariates) != tuple(assembled.covariate_names):
    raise SystemExit(
        "saved Stage-14 covariates differ from the freshly assembled inputs"
    )
if tuple(rw_full.input_sources) != tuple(assembled.source_names):
    raise SystemExit(
        "saved Stage-14 satellite sources differ from the selected "
        f"{RECONSTRUCTION_OBP_LABEL}+DUACS+CCMP inputs"
    )
if tuple(map(str, rw_full.input_source_files)) != tuple(
    map(str, assembled.source_files)
):
    raise SystemExit(
        "saved Stage-14 source files differ from the current Stage-12 "
        "products; rerun Stage 14 before Stage 16"
    )

# Align all draws and trend calculations to the Stage-15 common month interval.
trimmed_rows = np.arange(rec_months.size)[
    slice(EDGE_MONTHS, -EDGE_MONTHS) if EDGE_MONTHS else slice(None)
]
trimmed_rec_months = rec_months[trimmed_rows]
rw_full_months = np.asarray(rw_full.time_month).astype("datetime64[M]")
if not np.array_equal(trimmed_rec_months, rw_full_months):
    raise SystemExit(
        "fresh Stage-14 input months differ from saved Pred_RealWorld months"
    )

if budget_file.is_file():
    try:
        with np.load(budget_file, allow_pickle=False) as _axis_handle:
            budget_months = np.asarray(
                _axis_handle["time_month_int"], dtype=np.int64
            ).reshape(-1).astype("datetime64[M]")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"{budget_file.name} has no readable common month axis; "
            "rerun Stage 15"
        ) from exc
    analysis_months, (_analysis_take, _budget_take) = exact_common_month_indices(
        (rw_full_months, budget_months),
        labels=("Pred_RealWorld", budget_file.name),
    )
    if not np.array_equal(analysis_months, budget_months):
        raise SystemExit(
            f"{budget_file.name} months are not an exact contiguous subset "
            "of Pred_RealWorld"
        )
else:
    analysis_months = rw_full_months
    _analysis_take = np.arange(rw_full_months.size, dtype=np.int64)

analysis_rows = trimmed_rows[_analysis_take]
rw = replace(
    rw_full,
    pred=rw_full.pred[_analysis_take],
    epistemic=rw_full.epistemic[_analysis_take],
    total_uncertainty=rw_full.total_uncertainty[_analysis_take],
    t_years=rw_full.t_years[_analysis_take],
    time_month=analysis_months,
)


def reconstruct_members(x: np.ndarray) -> np.ndarray:
    """All reconstructed members [member, time, lev, lat]."""
    members = ensemble.predict_all_members(x)
    flat = members.reshape(-1, members.shape[-1])
    return unflatten(flat, mask, rho2.size, lat.size).reshape(
        members.shape[0], members.shape[1], rho2.size, lat.size)


def reconstruct(x: np.ndarray) -> np.ndarray:
    """Member-MEAN reconstruction [T, lev, lat] - the stage-14 product."""
    return reconstruct_members(x).mean(axis=0)


# %% [7] Monte Carlo (re-run this cell after changing DRAWS/MODE/CORR_KM) ------
t_years = decimal_year(analysis_months)
xc = t_years - t_years.mean()
a_trend = xc / (xc**2).sum()                 # OLS slope operator, Sv/yr
j26 = find_nearest_index(lat, 26.5)

_analysis_key = tuple(analysis_months.astype(np.int64)[[0, -1]]) + (
    analysis_months.size,
)
_input_file_key = tuple(
    (str(path), Path(path).stat().st_mtime_ns)
    for path in assembled.source_files
)


def _build_base_summary():
    members = reconstruct_members(x0)[:, analysis_rows]
    member_trends = np.einsum("t,ktij->kij", a_trend, members)
    return members.mean(axis=0), member_trends.std(axis=0, ddof=1)


base, sigma_eps_reconstruction = cached(
    "base_summary",
    (str(SOURCE_NN_DIR), OBP_SOURCE, _input_file_key, x0.shape,
     float(x0.sum()), _analysis_key),
    _build_base_summary,
)
print("  reconstruction-specific ensemble trend spread: median "
      f"{np.nanmedian(sigma_eps_reconstruction):.4f} Sv/yr")
#: Mid-depth AMOC core at 26.5N, consistent with Stage 14.
k26 = int(rw.cores.mid_index[j26])
print(f"  26.5N mid-depth core: level {k26}, sigma2 = {sigma2[k26]:.2f}")
if base.shape == rw.pred.shape:
    _dev = float(np.nanmax(np.abs(base - rw.pred)))
    print(f"  reproduces the saved Pred_RealWorld to {_dev:.2e} Sv")
    if _dev > 1e-3:
        print("  WARNING: the saved product disagrees with a fresh "
              "reconstruction from the same inputs - it is stale, or the "
              "stage-12 products changed since stage 14 last ran")
else:
    print(f"  NOTE: saved product is {rw.pred.shape}, this run is "
          f"{base.shape} - stage 14 predates the current stage-12 inputs; "
          "the cell-core levels are still valid, the comparison is skipped")

results = {}
for mode in MODES:
    rng = np.random.default_rng(SEED)
    # Accumulate deviations from the unperturbed run for numerical precision.
    s1 = np.zeros_like(base)                 # running sum of (pred - base)
    s2 = np.zeros_like(base)                 # running sum of squares
    trends = np.empty((DRAWS, rho2.size, lat.size))
    series26 = np.empty((DRAWS, base.shape[0]))
    for i in range(DRAWS):
        x = x0.copy()
        x[:, obp_block:obp_block + n_mascon] += pipeline_noise(mode, rng)[take]
        pred = reconstruct(x)[analysis_rows]
        dev = pred - base
        s1 += dev
        s2 += dev**2
        trends[i] = np.einsum("t,tij->ij", a_trend, pred)
        series26[i] = pred[:, k26, j26]
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == DRAWS:
            print(f"  {mode:<12} draw {i + 1}/{DRAWS}")
    bias = s1 / DRAWS                        # nonlinearity of the mapping
    var = np.maximum(s2 - DRAWS * bias**2, 0.0) / (DRAWS - 1)     # ddof=1
    results[mode] = dict(
        sigma_month=np.sqrt(np.mean(var, axis=0)),   # RMS over time, as stage 15
        sigma_trend=trends.std(axis=0, ddof=1),
        bias=bias,
        trends=trends, series26=series26,
        sigma_month_t=np.sqrt(var),                  # [T, lev, lat]
    )
    r = results[mode]
    print(f"  {mode:<12} monthly median {np.nanmedian(r['sigma_month']):.3f} Sv"
          f" | trend median {np.nanmedian(r['sigma_trend']):.4f} Sv/yr"
          f" | 26.5N monthly {r['sigma_month'][k26, j26]:.3f} Sv,"
          f" trend {r['sigma_trend'][k26, j26]:.4f} Sv/yr")
    print(f"  {'':<12} noise-induced bias (map nonlinearity): median "
          f"{np.nanmedian(np.abs(r['bias'])):.4f} Sv")

# %% [8] Against the stage-15 budget, and the effect on significance -----------
CENTRAL = PRODUCTION_MODE if PRODUCTION_MODE in results else MODES[0]
sigma_grace = results[CENTRAL]["sigma_trend"]
sigma_grace_month = results[CENTRAL]["sigma_month"]
sig_frac = {}
with_grace = None
if budget_file.is_file():
    with np.load(budget_file, allow_pickle=False) as _handle:
        b = {name: np.asarray(_handle[name]) for name in _handle.files}
    _required = {
        "sigma_map", "sigma_sate", "sigma_eps", "sigma_sate_month",
        "sigma_map_monthly", "sate_month_axis", "edge_months", "lat",
        "sigma2", "t_years", "time_month_int", "mapping_error_estimator",
        "sigma_map_estimator", "sigma_map_monthly_estimator", "run_id",
        "training_source_run_id",
        "sigma_map_estimator_code",
        "cmip_dataset_id", "satellite_dataset_id", "training_experiment",
        "covariates", "n_combos", "combo_tags", "combo_obp_sources",
        "combo_ssh_sources", "combo_wind_sources",
        "product_spread_include_gsfc", "product_spread_definition",
    }
    _missing = sorted(_required - set(b))
    if _missing:
        raise SystemExit(
            f"{budget_file.name} is missing {_missing}; rerun Stage 15"
        )

    def _text(name):
        value = np.asarray(b[name]).squeeze().item()
        return value.decode() if isinstance(value, bytes) else str(value)

    _plane = (rw.sigma2.size, rw.lat.size)
    s_map = np.asarray(b["sigma_map"], dtype=float)
    s_sate = np.asarray(b["sigma_sate"], dtype=float)
    s_eps = np.asarray(b["sigma_eps"], dtype=float)
    s_sate_month = np.asarray(b["sigma_sate_month"], dtype=float)
    s_map_month = np.asarray(b["sigma_map_monthly"], dtype=float)
    for _name, _field in (
        ("sigma_map", s_map), ("sigma_sate", s_sate),
        ("sigma_eps", s_eps), ("sigma_map_monthly", s_map_month),
    ):
        if _field.shape != _plane or not np.isfinite(_field).all() \
                or np.any(_field < 0):
            raise SystemExit(
                f"{budget_file.name} {_name} must be a finite, "
                f"nonnegative {_plane} field"
            )
    if s_sate_month.shape != (rw.pred.shape[0], *_plane) \
            or not np.isfinite(s_sate_month).all() \
            or np.any(s_sate_month < 0):
        raise SystemExit(
            f"{budget_file.name} sigma_sate_month does not match the "
            "trimmed reconstruction"
        )
    _budget_lat = np.asarray(b["lat"], dtype=float).reshape(-1)
    _budget_sigma2 = np.asarray(b["sigma2"], dtype=float).reshape(-1)
    _budget_t = np.asarray(b["t_years"], dtype=float).reshape(-1)
    _budget_month = np.asarray(b["time_month_int"], dtype="int64").reshape(-1)
    _product_month = np.asarray(b["sate_month_axis"], dtype="int64").reshape(-1)
    if rw.time_month is None:
        raise SystemExit("Pred_RealWorld lacks the required calendar month axis")
    _rw_month = np.asarray(rw.time_month).astype("datetime64[M]").astype("int64")
    if not (
        np.array_equal(_budget_lat, np.asarray(rw.lat, dtype=float))
        and np.array_equal(_budget_sigma2, np.asarray(rw.sigma2, dtype=float))
        and np.allclose(_budget_t, rw.t_years, rtol=0, atol=1e-12)
        and np.array_equal(_budget_month, _rw_month)
        and np.array_equal(_product_month, _rw_month)
    ):
        raise SystemExit(
            f"{budget_file.name} coordinates/months differ from "
            "Pred_RealWorld"
        )
    if int(np.asarray(b["edge_months"]).squeeze()) != EDGE_MONTHS:
        raise SystemExit(f"{budget_file.name} uses a different edge trim")
    _estimators = {
        _text("mapping_error_estimator"), _text("sigma_map_estimator"),
        _text("sigma_map_monthly_estimator"),
    }
    if len(_estimators) != 1:
        raise SystemExit(
            f"{budget_file.name} mixes monthly/trend mapping estimators"
        )
    _estimator_code = int(np.asarray(b["sigma_map_estimator_code"]).squeeze())
    if _estimator_code not in (1, 2):
        raise SystemExit(f"{budget_file.name} has an unknown estimator code")
    _expected_text = {
        "run_id": rw.run_id or "unknown",
        "training_source_run_id": ACTIVE_RUN_ID,
        "cmip_dataset_id": rw.cmip_dataset_id or "unknown",
        "satellite_dataset_id": rw.satellite_dataset_id or "unknown",
        "training_experiment": EXPERIMENT,
        "covariates": COVARIATES,
    }
    for _name, _expected in _expected_text.items():
        if _text(_name) != _expected:
            raise SystemExit(
                f"{budget_file.name} {_name}={_text(_name)!r}, expected "
                f"{_expected!r}"
            )
    _product_spread_include_gsfc = bool(
        np.asarray(b["product_spread_include_gsfc"]).squeeze().item()
    )
    _product_spread_definition = _text("product_spread_definition")
    if (
        not BUDGET_TAG
        and _product_spread_include_gsfc
        != DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC
    ):
        raise SystemExit(
            f"untagged budget {budget_file.name} does not match the active "
            "profile's GSFC-inclusion policy; use tagged Stage-15/16 "
            "sensitivity outputs instead"
        )
    _saved_registry = list(zip(
        np.asarray(b["combo_obp_sources"]).astype(str).reshape(-1).tolist(),
        np.asarray(b["combo_ssh_sources"]).astype(str).reshape(-1).tolist(),
        np.asarray(b["combo_wind_sources"]).astype(str).reshape(-1).tolist(),
        np.asarray(b["combo_tags"]).astype(str).reshape(-1).tolist(),
    ))
    _saved_n_combos = int(np.asarray(b["n_combos"]).squeeze())
    _expected_registry = uncertainty_product_registry(
        COVARIATES, include_gsfc=_product_spread_include_gsfc
    )
    if (
        _saved_n_combos != len(_saved_registry)
        or _saved_registry != _expected_registry
    ):
        raise SystemExit(
            f"{budget_file.name} has an internally inconsistent selected "
            "product registry; rerun Stage 15"
        )
    print(
        "product-spread registry: "
        f"{_saved_n_combos} combinations; GSFC "
        f"{'included' if _product_spread_include_gsfc else 'excluded'}"
    )
    print("=" * 70)
    print(f"{'term':<34}{'monthly (Sv)':>14}{'trend (Sv/yr)':>15}")
    print(f"{'sigma_map (mapping)':<34}"
          f"{np.nanmedian(s_map_month):>14.3f}"
          f"{np.nanmedian(s_map):>15.4f}")
    print(f"{'sigma_sate (product swap)':<34}"
          f"{np.nanmedian(s_sate_month):>14.3f}{np.nanmedian(s_sate):>15.4f}")
    for mode in MODES:
        print(f"{'sigma_grace, ' + mode:<34}"
              f"{np.nanmedian(results[mode]['sigma_month']):>14.3f}"
              f"{np.nanmedian(results[mode]['sigma_trend']):>15.4f}")
    print(f"{'sigma_eps (ensemble)':<34}{'':>14}"
          f"{np.nanmedian(sigma_eps_reconstruction):>15.4f}")

    # Match Stage-15 bootstrap settings and use product-specific member spread.
    if OBP_SOURCE == "GRACE" and not np.allclose(
        sigma_eps_reconstruction, s_eps, rtol=0.0, atol=2e-7,
        equal_nan=True,
    ):
        raise SystemExit(
            "fresh JPL ensemble-trend spread does not reproduce Stage 15 "
            "sigma_eps; rerun Stage 14/15 before Stage 16"
        )
    kw = dict(
        method="mbb", block_months=48, n_boot=1000, seed=0,
        sigma_map=s_map, sigma_eps=sigma_eps_reconstruction,
    )
    without = robust_trend(rw.pred, rw.t_years,
                           sigma_sate=s_sate, **kw)
    with_grace = robust_trend(
        rw.pred, rw.t_years, sigma_sate=s_sate,
        sigma_grace=sigma_grace, **kw
    )
    # Match the Stage-15 testable domain, excluding structural-zero cells.
    valid = np.isfinite(without.slope_pval) & (rw.pred.std(axis=0) > 0)
    sig_frac = {
        "without_grace": float(without.is_significant()[valid].mean()),
        "with_grace": float(with_grace.is_significant()[valid].mean()),
    }
    print(f"significant fraction: {sig_frac['without_grace']:.1%} "
          f"(stage-15 budget) -> {sig_frac['with_grace']:.1%} "
          f"(+ sigma_grace, {CENTRAL})")
else:
    print(f"(no {budget_file.name} yet - run stage 15 to see the combined "
          "budget and significance effect)")

# %% [9] Figures ---------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 3.1 * len(MODES) + 3.1))
gs = fig.add_gridspec(len(MODES) + 1, 1, hspace=0.55,
                      left=0.08, right=0.88, bottom=0.05, top=0.94)
for row, mode in enumerate(MODES):
    field = results[mode]["sigma_month"]
    _, (ax_s, ax_a), mesh = section_row(
        field, lat, sigma2, cmap=CMAP_AMPLITUDE, vmin=0,
        vmax=float(np.nanpercentile(field, 98)),
        fig=fig, subplot_spec=gs[row], add_colorbar=False)
    ax_s.set_title(f"monthly $\\sigma_{{GRACE}}$, {mode} "
                   f"(median {np.nanmedian(field):.2f} Sv)", loc="left",
                   fontsize=plt.rcParams["font.size"])
    cbar = fig.colorbar(mesh, ax=[ax_s, ax_a], pad=0.015, fraction=0.03,
                        aspect=22)
    cbar.outline.set_visible(False)
    if row == 0:
        cbar.ax.set_title("Sv", fontsize=plt.rcParams["font.size"], pad=6)
field = results[CENTRAL]["sigma_trend"]
_, (ax_s, ax_a), mesh = section_row(
    field, lat, sigma2, cmap=CMAP_AMPLITUDE, vmin=0,
    vmax=float(np.nanpercentile(field, 98)),
    fig=fig, subplot_spec=gs[-1], add_colorbar=False)
ax_s.set_title(f"TREND $\\sigma_{{GRACE}}$, {CENTRAL} "
               f"(median {np.nanmedian(field):.4f} Sv yr$^{{-1}}$)",
               loc="left", fontsize=plt.rcParams["font.size"])
cbar = fig.colorbar(mesh, ax=[ax_s, ax_a], pad=0.015, fraction=0.03, aspect=22)
cbar.outline.set_visible(False)
cbar.ax.set_title("Sv yr$^{-1}$", fontsize=plt.rcParams["font.size"], pad=6)
save_figure(fig, OUT_DIR / f"Pred_GraceNoise_sections{_SUFFIX}",
            formats=("png",))

t_plot = decimal_year(analysis_months)
fig, ax = plt.subplots(figsize=(7.4, 3.4), constrained_layout=True)
shade_gap(ax)
for mode, color in zip(MODES, ("0.55", "crimson", "steelblue")):
    s = results[mode]["sigma_month_t"][:, k26, j26]
    ax.fill_between(t_plot, base[:, k26, j26] - s, base[:, k26, j26] + s,
                    color=color, alpha=0.25, linewidth=0,
                    label=f"$\\pm1\\sigma$ GRACE noise ({mode})")
ax.plot(t_plot, base[:, k26, j26], color=COLORS["prediction"], lw=1.3,
        label="NeurMOC (unperturbed)")
ax.set_xlim(t_plot[0], t_plot[-1])
ax.set_xlabel("Year")
ax.set_ylabel(r"$\Psi$ at 26.5" "\N{DEGREE SIGN}N (Sv)")
ax.set_title(f"GRACE measurement noise propagated to the reconstruction "
             f"({DRAWS} draws)", loc="left")
ax.text(0.02, 0.05, "shaded band at the GRACE/GRACE-FO gap: the linear "
        "bridge of stage 12\nstretches two noisy endpoints across it",
        transform=ax.transAxes, va="bottom", bbox=TEXT_BBOX)
ax.legend(loc="upper right", ncol=2, fontsize=plt.rcParams["font.size"] - 1)
save_figure(fig, OUT_DIR / f"Pred_GraceNoise_rapid{_SUFFIX}",
            formats=("png",))

# %% [10] Save ------------------------------------------------------------------
payload = dict(
    sigma_grace=sigma_grace, sigma_grace_month=sigma_grace_month,
    # Time-resolved monthly uncertainty, including the GRACE/GRACE-FO gap.
    sigma_grace_month_t=results[CENTRAL]["sigma_month_t"].astype(np.float32),
    time_month=np.datetime_as_string(analysis_months, unit="M"),
    t_years=decimal_year(analysis_months),
    core_index_26N=np.int64(k26), lat_index_26N=np.int64(j26),
    central_mode=np.str_(CENTRAL),
    central_mode_code=np.int64(
        {"independent": 1, "correlated": 2, "coherent": 3}[CENTRAL]
    ),
    modes=np.asarray(MODES, dtype="U"),
    n_draws=np.int64(DRAWS), seed=np.int64(SEED),
    corr_length_km=np.float64(CORR_KM),
    edge_months=np.int64(EDGE_MONTHS),
    n_mascon=np.int64(n_mascon),
    sigma_input_median_pa=np.float64(np.median(sig_net)),
    lat=lat, rho2=rho2,
    granule=np.str_(Path(GRACE_MASCON_NC).name),
    run_id=np.str_(ACTIVE_RUN_ID),
    training_source_run_id=np.str_(ACTIVE_RUN_ID),
    satellite_dataset_id=np.str_(SATELLITE_DATASET_ID),
    training_experiment=np.str_(EXPERIMENT),
    covariates=np.str_(COVARIATES),
    reconstruction_product_tag=np.str_(RECONSTRUCTION_PRODUCT_TAG),
    reconstruction_obp_source=np.str_(RECONSTRUCTION_OBP_SOURCE),
    reconstruction_ssh_source=np.str_(RECONSTRUCTION_SSH_SOURCE),
    reconstruction_wind_source=np.str_(RECONSTRUCTION_WIND_SOURCE),
    reconstruction_file=np.str_(
        f"Pred_RealWorld{RECONSTRUCTION_PRODUCT_TAG}.mat"
    ),
    measurement_noise_source=np.str_(MEASUREMENT_NOISE_SOURCE),
    measurement_noise_source_center=np.str_("JPL"),
    measurement_noise_is_proxy=np.bool_(MEASUREMENT_NOISE_IS_PROXY),
    measurement_noise_definition=np.str_(MEASUREMENT_NOISE_DEFINITION),
    sigma_source=np.str_(
        "JPL mascon granule `uncertainty`; over the ocean JPL scales the "
        "formal covariance to match GRACE-minus-in-situ bottom pressure "
        "(conservative). Excludes GIA/geocenter/C20-C30 correction errors."),
    complementary_to=np.str_(
        "sigma_sate (stage 15): the exact selected product-processing "
        "registry prices processing sensitivity while this term prices "
        "measurement noise propagated around the selected point estimate"),
    **{f"sigma_grace_{m}": results[m]["sigma_trend"] for m in MODES},
    **{f"sigma_grace_month_{m}": results[m]["sigma_month"] for m in MODES},
    **{f"bias_{m}": results[m]["bias"] for m in MODES},
    **{f"trends_{m}": results[m]["trends"].astype(np.float32) for m in MODES},
    **{f"series26_{m}": results[m]["series26"].astype(np.float32)
       for m in MODES},
    **({k: np.float64(v) for k, v in sig_frac.items()} if sig_frac else {}),
    **({
        "source_budget_file": np.str_(budget_file.name),
        "n_combos": np.asarray(b["n_combos"], dtype=np.int64),
        "combo_tags": np.asarray(b["combo_tags"]).astype("U"),
        "combo_obp_sources": np.asarray(
            b["combo_obp_sources"]
        ).astype("U"),
        "combo_ssh_sources": np.asarray(
            b["combo_ssh_sources"]
        ).astype("U"),
        "combo_wind_sources": np.asarray(
            b["combo_wind_sources"]
        ).astype("U"),
        "product_spread_include_gsfc": np.asarray(
            b["product_spread_include_gsfc"], dtype=np.bool_
        ),
        "product_spread_definition": np.asarray(
            b["product_spread_definition"]
        ).astype("U"),
    } if with_grace is not None else {}),
)
try:
    np.savez(OUT_FILE, **payload)
except PermissionError as exc:
    raise SystemExit(
        f"Cannot overwrite {OUT_FILE}. Close any open NPZ file handles "
        "and rerun Stage 16."
    ) from exc
print("saved:", OUT_FILE)
if with_grace is not None:
    save_realworld_trend_npz(
        rw,
        with_grace,
        OUT_DIR / f"real_world_trend_stats{_SUFFIX}.npz",
        block_months=kw["block_months"],
        n_boot=kw["n_boot"],
        seed=kw["seed"],
        edge_months=EDGE_MONTHS,
        extra_fields={
            "training_source_run_id": np.asarray(ACTIVE_RUN_ID),
            "grace_n_draws": np.asarray(DRAWS, dtype=np.int64),
            "grace_seed": np.asarray(SEED, dtype=np.int64),
            "grace_central_mode": np.asarray(CENTRAL),
            "reconstruction_product_tag": np.asarray(
                RECONSTRUCTION_PRODUCT_TAG
            ),
            "reconstruction_obp_source": np.asarray(
                RECONSTRUCTION_OBP_SOURCE
            ),
            "reconstruction_ssh_source": np.asarray(
                RECONSTRUCTION_SSH_SOURCE
            ),
            "reconstruction_wind_source": np.asarray(
                RECONSTRUCTION_WIND_SOURCE
            ),
            "measurement_noise_source": np.asarray(
                MEASUREMENT_NOISE_SOURCE
            ),
            "measurement_noise_source_center": np.asarray("JPL"),
            "measurement_noise_is_proxy": np.asarray(
                MEASUREMENT_NOISE_IS_PROXY, dtype=np.bool_
            ),
            "measurement_noise_definition": np.asarray(
                MEASUREMENT_NOISE_DEFINITION
            ),
            "mapping_error_estimator": np.asarray(next(iter(_estimators))),
            "sigma_map_estimator_code": np.asarray(
                _estimator_code, dtype=np.int64
            ),
            "source_budget_file": np.asarray(budget_file.name),
            "product_spread_include_gsfc": np.asarray(
                b["product_spread_include_gsfc"], dtype=np.bool_
            ),
            "product_spread_definition": np.asarray(
                b["product_spread_definition"]
            ).astype("U"),
            "n_combos": np.asarray(b["n_combos"], dtype=np.int64),
            "combo_tags": np.asarray(b["combo_tags"]).astype("U"),
            "combo_obp_sources": np.asarray(
                b["combo_obp_sources"]
            ).astype("U"),
            "combo_ssh_sources": np.asarray(
                b["combo_ssh_sources"]
            ).astype("U"),
            "combo_wind_sources": np.asarray(
                b["combo_wind_sources"]
            ).astype("U"),
        },
    )
print(f"figures: {OUT_DIR}")
