"""Stage 14: reconstruct MOC anomalies from satellite observations.

Apply every trained ensemble member to the Stage-12 mascon inputs. The member
mean is the reconstruction, and the member spread quantifies network-ensemble
variation. Results are written to ``RealWorld/Pred_RealWorld.mat``.

The user-settings block selects the input combination and ablation options;
``--trained-on`` selects the trained experiment.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import (
    ACTIVE_RUN_ID,
    BASELINE_YEARS,
    BASINMASK_DIR,
    CMIP_DATASET_ID,
    MOC_CONVENTION,
    OBS_MASCON_ROOT,
    OSNAP_DIR,
    SATELLITE_DATASET_ID,
    SCIENTIFIC_CONFIG,
    RUN_ROOT,
    mascon_var,
    results_dir,
)
from neurmoc.inference import TrainedEnsemble
from neurmoc.io_utils import ensure_dir, load_npz_or_mat, require_dir, require_file, save_mat
from neurmoc.moc_utils import find_nearest_index, locate_cell_cores, unflatten
from neurmoc.model import configure_tensorflow_runtime
from neurmoc.naming import (
    lpf_tag_from_name,
    prepare_covariate_config,
    training_config_from_scientific,
)
from neurmoc.plotting import (
    CMAP_AMPLITUDE,
    CMAP_DIVERGING,
    GRACE_GAP,
    apply_style,
    section_row,
    shade_gap,
)
from neurmoc.plotting.style import COLORS, TEXT_BBOX, save_figure
from neurmoc.rapid import load_rapid
from neurmoc.results import load_training_moc_baseline
from neurmoc.timeaxis import decimal_year, normalize_month_axis

# ========== User settings ==========
#: Experiment the networks were trained on: "ACCESS_hist+SSP126" |
#: "ACCESS_historical" | "ACCESS_hist+SSP585" | ...
DEFAULT_TRAINING = training_config_from_scientific(SCIENTIFIC_CONFIG)
TRAINED_ON = DEFAULT_TRAINING.cmip_name
#: Input set - selects the trained (ablation) network: any comma-separated
#: subset of the three inputs.
COVARIATE_NAMES = DEFAULT_TRAINING.covariate_names


EXPERIMENTS = [DEFAULT_TRAINING.experiment_name()]
REFERENCE_GRID = BASINMASK_DIR / "ACCESS_target_MOC_grid.npz"
# Alternative wind inputs must use the same time and anomaly convention.
USE_ERA5_WINDS = False            # True = ERA5 10-m wind instead of CCMP

#: Replace an input by its time mean to test its contribution (ablation).
PERMUTE = {"obp": False, "ssh": False, "uas": False,
           "tauu": False, "curltau": False}

#: Supported wind predictors; each follows the same input checks.
WIND_SLOTS = ("uas", "tauu", "curltau")

OBS_DIR = OBS_MASCON_ROOT


@dataclass(frozen=True)
class ObservationRecord:
    """One timestamped, spatially registered observation feature block."""

    values: np.ndarray
    time_month: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    source_name: str
    source_file: str
    provenance: dict[str, object]


@dataclass(frozen=True)
class AssembledInputs:
    """Aligned feature matrix plus the inputs used to construct it."""

    values: np.ndarray
    time_month: np.ndarray
    covariate_names: tuple[str, ...]
    source_names: tuple[str, ...]
    source_files: tuple[str, ...]
    baseline_specs: tuple[str, ...]
    lon: np.ndarray = None   # shared mascon coordinates, in feature order
    lat: np.ndarray = None   # (each covariate block was verified identical)


# Stage-12 OBP product; alternatives receive distinct output suffixes.
OBP_SOURCE = "GRACE"

# Stage-12 SSH product; alternatives receive distinct output suffixes.
SSH_SOURCE = "DUACS"

# Latitudes used for the OSNAP-section comparison.
OSNAP_LATS = (53.5, 56.5, 59.5)


def permute_suffix() -> str:
    active = [k for k, v in PERMUTE.items() if v]
    suffix = "_permute_" + "".join(active) if active else ""
    if OBP_SOURCE != "GRACE":
        suffix += f"_obp{OBP_SOURCE.removeprefix('GRACE_')}"
    if SSH_SOURCE != "DUACS":
        suffix += f"_ssh{SSH_SOURCE}"
    if USE_ERA5_WINDS:
        # Keep the alternative wind product separate from the default CCMP output.
        suffix += "_ERA5wind"
    return suffix


def _metadata_value(data, key: str):
    if key not in data:
        return None
    value = np.asarray(data[key])
    return value.item() if value.ndim == 0 else value.copy()


def load_observation(name: str, key: str, lpf: bool) -> ObservationRecord:
    """Load one observed input field with its monthly time coordinate."""
    path = require_file(OBS_DIR / f"{name}.npz", name)
    with np.load(path, allow_pickle=False) as data:
        value_key = f"{key}_LPF_ALL" if lpf else key
        if value_key not in data:
            raise RuntimeError(f"{path}: missing {value_key}")
        values = np.asarray(data[value_key])
        if values.ndim != 2:
            raise RuntimeError(
                f"{path}: {value_key} must be [time, mascon], got {values.shape}")
        if "time_month" not in data:
            raise RuntimeError(f"{path} has no time_month coordinate")
        months = normalize_month_axis(data["time_month"], values.shape[0], name)
        lon_key, lat_key = f"{key}_lon", f"{key}_lat"
        if lon_key not in data or lat_key not in data:
            raise RuntimeError(f"{path}: missing {lon_key} or {lat_key}")
        lon, lat = np.asarray(data[lon_key]), np.asarray(data[lat_key])
        if lon.shape != (values.shape[1],) or lat.shape != (values.shape[1],):
            raise RuntimeError(
                f"{path}: coordinate/feature mismatch: data {values.shape}, "
                f"lon {lon.shape}, lat {lat.shape}")
        provenance = {
            field: _metadata_value(data, field)
            for field in (
                "anomaly", "wind_anomaly", "baseline_period", "baseline_spec",
                "wind_convention", "wind_source", "source_product",
                "observation_schema_version",
            )
            if field in data
        }
    return ObservationRecord(
        values=values, time_month=months, lon=lon, lat=lat,
        source_name=name, source_file=str(path), provenance=provenance,
    )


def _wind_covariate_name(names) -> str | None:
    """The wind covariate present in `names`, or None if the set has none."""
    present = [mascon_var(slot) for slot in WIND_SLOTS
               if mascon_var(slot) in tuple(names)]
    if len(present) > 1:
        raise RuntimeError(
            f"a network takes one wind input, but {present} were configured")
    return present[0] if present else None


def _require_anomaly_provenance(record: ObservationRecord, kind: str) -> None:
    is_wind = kind in WIND_SLOTS
    flag_key = "wind_anomaly" if is_wind else "anomaly"
    if record.provenance.get(flag_key) is not True:
        raise RuntimeError(
            f"{record.source_file}: expected {flag_key}=true. Rebuild the "
            f"{kind} observation as a 2004-2009 anomaly.")
    if is_wind and record.provenance.get("wind_convention") != "anomaly_2004_2009":
        raise RuntimeError(
            f"{record.source_file}: wind_convention must be "
            "'anomaly_2004_2009'")
    period = np.asarray(record.provenance.get("baseline_period", [])).astype(int)
    expected = np.asarray(BASELINE_YEARS)
    if period.shape != expected.shape or not np.array_equal(period, expected):
        raise RuntimeError(
            f"{record.source_file}: baseline_period must be "
            f"{expected.tolist()}, got {period.tolist()}")


def _feature_spec(covariate_name: str) -> tuple[str, str, str]:
    """Map an exact trained feature name to observation file/key/type."""
    specs = {
        mascon_var("obp"): (f"obp_{OBP_SOURCE}", f"obp_{OBP_SOURCE}", "obp"),
        mascon_var("ssh"): (f"ssh_{SSH_SOURCE}", f"ssh_{SSH_SOURCE}", "ssh"),
    }
    # Match the observed wind product to the trained covariate.
    wind_product = "ERA5" if USE_ERA5_WINDS else "CCMP"
    for slot in WIND_SLOTS:
        specs[mascon_var(slot)] = (f"{slot}_{wind_product}",
                                   f"{slot}_{wind_product}", slot)
    try:
        return specs[covariate_name]
    except KeyError as exc:
        raise ValueError(
            f"Real-world reconstruction does not support covariate "
            f"{covariate_name!r}; supported names are {sorted(specs)}") from exc


def assemble_inputs(covariates, lpf: bool) -> AssembledInputs:
    """Load configured blocks in training order and align their dates exactly."""
    names = tuple(covariates.names)
    if not names:
        raise ValueError("At least one real-world covariate is required")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate covariates are not allowed: {names}")

    records: list[tuple[str, ObservationRecord]] = []
    for covariate_name in names:
        source, key, kind = _feature_spec(covariate_name)
        record = load_observation(source, key, lpf)
        _require_anomaly_provenance(record, kind)
        records.append((kind, record))

    common = records[0][1].time_month
    for _, record in records[1:]:
        common = np.intersect1d(common, record.time_month, assume_unique=True)
    if common.size == 0:
        spans = [
            f"{r.source_name}={r.time_month[0]}..{r.time_month[-1]}"
            for _, r in records
        ]
        raise RuntimeError(f"Observation records do not overlap: {spans}")
    if common.size > 1 and np.any(np.diff(common).astype(int) != 1):
        raise RuntimeError("The common observation window contains missing months")

    ref = records[0][1]
    blocks = []
    for kind, record in records:
        same_shape = record.lon.shape == ref.lon.shape and record.lat.shape == ref.lat.shape
        if not (same_shape
                and np.allclose(record.lon, ref.lon, equal_nan=True)
                and np.allclose(record.lat, ref.lat, equal_nan=True)):
            raise RuntimeError(
                f"{record.source_file}: mascon coordinates/order differ from "
                f"{ref.source_file}")
        index = np.searchsorted(record.time_month, common)
        if not np.array_equal(record.time_month[index], common):
            raise RuntimeError(f"Internal date-index error for {record.source_file}")
        block = record.values[index]
        if PERMUTE[kind]:
            block = np.broadcast_to(block.mean(axis=0), block.shape).copy()
        blocks.append(block)
        dropped = record.values.shape[0] - common.size
        print(f"  {record.source_name}: {record.values.shape[0]} months; "
              f"using {common[0]}..{common[-1]} ({common.size}, dropped {dropped})")

    return AssembledInputs(
        values=np.concatenate(blocks, axis=1), time_month=common,
        covariate_names=names,
        source_names=tuple(r.source_name for _, r in records),
        source_files=tuple(r.source_file for _, r in records),
        baseline_specs=tuple(str(r.provenance.get("baseline_spec") or "not recorded")
                             for _, r in records),
        lon=np.asarray(ref.lon).squeeze(), lat=np.asarray(ref.lat).squeeze(),
    )


def validate_training_covariates(
    nn_path: Path, configured_names: tuple[str, ...]
) -> dict:
    """Validate and return the trained model's saved configuration."""
    config_path = require_file(nn_path / "training_config.json", "training config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trained = tuple(
        part.strip() for part in config.get("covariate_names", "").split(",")
        if part.strip()
    )
    if trained != configured_names:
        raise RuntimeError(
            "Configured real-world covariates do not match the trained model "
            f"order: configured={configured_names}, trained={trained}")
    if _wind_covariate_name(configured_names) is not None:
        trained_wind = config.get("wind_convention")
        if trained_wind != "anomaly_2004_2009":
            raise RuntimeError(
                "Real-world wind is a 2004-2009 anomaly, but the trained "
                f"model records wind_convention={trained_wind!r}")
    # Require the trained model's target convention to match this profile.
    trained_moc = config.get("moc_convention", "absolute")
    if trained_moc != MOC_CONVENTION:
        raise RuntimeError(
            f"Trained model records moc_convention={trained_moc!r}, but the "
            f"active profile expects {MOC_CONVENTION!r} - the reconstruction "
            "would be interpreted in the wrong reference frame")
    return config


def overview_figure(pred_yz, pred_yz_std, lat, sigma2, t, out_stem,
                    time_month=None, baseline=None):
    """Plot mid-depth MOC strength and the AMOC comparison at 26.5N.

    `baseline` is the validated 2004-2009 mean for anomaly reconstructions;
    absolute reconstructions pass None.
    """
    if t.shape != (pred_yz.shape[0],):
        raise RuntimeError(
            f"Overview time/data mismatch: {t.shape} vs {pred_yz.shape}")
    # Locate cell cores in the reference mean, not the anomalies.
    cores = locate_cell_cores(
        baseline if baseline is not None else pred_yz.mean(axis=0),
        lat, sigma2)
    cols = np.arange(lat.size)
    strength = pred_yz[:, cores.mid_index, cols]        # [time, lat]
    spread = pred_yz_std[:, cores.mid_index, cols]
    # Add the reference mean for the strength plot; compare RAPID anomalies separately.
    strength_abs = strength
    if baseline is not None:
        strength_abs = strength + baseline[cores.mid_index, cols][None, :]

    fig, axes = plt.subplots(3, 1, figsize=(7, 7.8), constrained_layout=True)
    ax = axes[0]
    pm = ax.pcolormesh(t, lat, strength_abs.T, cmap=CMAP_AMPLITUDE, vmin=0, vmax=25,
                       shading="nearest", rasterized=True)
    for edge in GRACE_GAP:
        ax.axvline(edge, color="k", ls="--", lw=0.6)
    ax.set_ylabel("Latitude")
    ax.set_title("mid-depth cell strength (Sv; training baseline + "
                 "reconstructed anomaly); dashed: GRACE gap", loc="left")
    cbar = fig.colorbar(pm, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_title("Sv", fontsize=plt.rcParams["font.size"], pad=4)

    ax = axes[1]
    j = find_nearest_index(lat, 26.5)
    shade_gap(ax)
    ax.fill_between(t, strength[:, j] - spread[:, j], strength[:, j] + spread[:, j],
                    color=COLORS["prediction"], alpha=0.15, linewidth=0)
    ax.plot(t, strength[:, j], color=COLORS["prediction"], lw=1.2,
            label="NeurMOC (band: network spread)")
    try:
        # Use the 2004-2009-referenced RAPID anomaly over its full record.
        rapid_rec = load_rapid(edge_months=0)
        obs = rapid_rec.anomaly
        ax.plot(rapid_rec.t_years, obs, color=COLORS["truth"], lw=1.0,
                label="RAPID (anomaly)")
        # Align series by calendar month.
        if time_month is None:
            raise RuntimeError("overview_figure requires the reconstruction's "
                               "time_month for the RAPID comparison")
        _, i_rec, i_obs = np.intersect1d(
            np.asarray(time_month).astype("datetime64[M]"),
            rapid_rec.time_month, return_indices=True)
        rec, obs_c = strength[i_rec, j], obs[i_obs]
        ax.text(0.02, 0.05,
                f"vs RAPID ({i_rec.size} common months): "
                f"corr = {np.corrcoef(rec, obs_c)[0, 1]:.2f}, "
                f"bias = {(rec - obs_c).mean():+.2f} Sv",
                transform=ax.transAxes, va="bottom", bbox=TEXT_BBOX)
    except (FileNotFoundError, KeyError):
        print("  (RAPID series unavailable; overview panel omits the comparison)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(r"$\Psi$ at 26.5" "\N{DEGREE SIGN}N (Sv)")
    ax.set_title("AMOC at 26.5\N{DEGREE SIGN}N", loc="left")
    ax.legend(loc="upper right", ncol=2)
    ax.tick_params(top=False, right=False)

    # Compare OSNAP with nearby latitude cores after demeaning matched months.
    ax = axes[2]
    shade_gap(ax)
    try:
        osnap = load_npz_or_mat(
            OSNAP_DIR / "OSNAP_LPF",
            [
                "OSNAP_monthly_LPF", "t_year", "time_month",
                "product_schema_version", "source_variable",
            ],
        )
        osnap_schema = int(
            np.asarray(osnap["product_schema_version"]).squeeze()
        )
        osnap_source_variable = str(
            np.asarray(osnap["source_variable"]).squeeze()
        )
        if osnap_schema != 2 or osnap_source_variable != "MOC_ALL":
            raise RuntimeError(
                "OSNAP_LPF must be derived from the provider's MOC_ALL series"
            )
        t_o = np.asarray(osnap["t_year"]).squeeze()
        obs_o = np.asarray(osnap["OSNAP_monthly_LPF"]).squeeze()
        if time_month is None:
            raise RuntimeError("overview_figure requires the reconstruction's "
                               "time_month for the OSNAP comparison")
        _, i_rec, i_obs = np.intersect1d(
            np.asarray(time_month).astype("datetime64[M]"),
            np.asarray(osnap["time_month"]).astype("datetime64[M]"),
            return_indices=True)
        ax.plot(t_o, obs_o - obs_o[i_obs].mean(), color=COLORS["truth"],
                lw=1.0, label="OSNAP total (anomaly)")
        stats = []
        for lat0, shade in zip(OSNAP_LATS, (1.0, 0.65, 0.35)):
            j = find_nearest_index(lat, lat0)
            rec = strength[:, j] - strength[i_rec, j].mean()
            ax.plot(t, rec, color=COLORS["prediction"], lw=1.1, alpha=shade,
                    label=f"NeurMOC {lat0:.1f}\N{DEGREE SIGN}N")
            stats.append(f"{lat0:.1f}N "
                         f"{np.corrcoef(rec[i_rec], obs_o[i_obs] - obs_o[i_obs].mean())[0, 1]:.2f}")
        ax.text(0.02, 0.05, f"corr over {i_rec.size} common months: "
                + ", ".join(stats),
                transform=ax.transAxes, va="bottom", bbox=TEXT_BBOX)
        ax.legend(loc="upper right", ncol=2)
    except FileNotFoundError:
        print("  (OSNAP series unavailable; overview panel omits the comparison)")
    except KeyError as exc:
        raise RuntimeError(
            "OSNAP_LPF is missing required provenance fields"
        ) from exc
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(r"$\Psi$ (Sv)")
    ax.set_xlabel("Year")
    ax.set_title("subpolar AMOC at the OSNAP latitudes", loc="left")
    ax.tick_params(top=False, right=False)
    save_figure(fig, out_stem, formats=("png",))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-t", "--trained-on", default=TRAINED_ON,
                        help="training experiment whose networks to use")
    parser.add_argument("--obp-source", default=None,
                        choices=["GRACE", "GRACE_CSR", "GRACE_GSFC"],
                        help="GRACE OBP product feeding the reconstruction "
                             "(default: the JPL mascons; variants suffix "
                             "every output, e.g. _obpCSR)")
    parser.add_argument("--ssh-source", default=None,
                        choices=["DUACS", "NASASSH", "ZeroIce"],
                        help="SSH product feeding the reconstruction "
                             "(default: DUACS; NASASSH suffixes every "
                             "output with _sshNASASSH)")
    parser.add_argument("--wind-source", default=None,
                        choices=["CCMP", "ERA5"],
                        help="wind product feeding the reconstruction "
                             "(default: CCMP; ERA5 sets USE_ERA5_WINDS and "
                             "suffixes every output with _ERA5wind)")
    parser.add_argument("-E", "--experiment", default=None,
                        help="hyperparameter-experiment folder name to "
                             "reconstruct with (default: the profile "
                             "baseline), e.g. FullDepth_PCAinY64_ResNet_"
                             "Neur192x96x48_5foldCV_Reg0.01Drop0.2_"
                             "swishActivation_LPF2Year")
    parser.add_argument("-x", "--covariates", default=COVARIATE_NAMES,
                        help="comma-separated input set - selects the trained "
                             "ablation network (default: all three inputs)")
    return parser.parse_args()


def main(trained_on: str = TRAINED_ON,
         experiment: str | None = None,
         obp_source: str | None = None,
         ssh_source: str | None = None,
         wind_source: str | None = None,
         covariate_names: str = COVARIATE_NAMES) -> None:
    global OBP_SOURCE, SSH_SOURCE, USE_ERA5_WINDS
    if obp_source:
        OBP_SOURCE = obp_source
    if ssh_source:
        SSH_SOURCE = ssh_source
    if wind_source:
        USE_ERA5_WINDS = wind_source == "ERA5"
    configure_tensorflow_runtime(seed=DEFAULT_TRAINING.random_seed)
    apply_style()
    covariates = prepare_covariate_config(covariate_names)

    reference = load_npz_or_mat(REFERENCE_GRID, ["rho2_full", "lat_psi"])
    rho2 = np.asarray(reference["rho2_full"]).squeeze()
    lat = np.asarray(reference["lat_psi"]).squeeze()
    sigma2 = rho2 - 1000 if rho2[0] > 1000 else rho2

    for experiment in ([experiment] if experiment else EXPERIMENTS):
        lpf_tag = lpf_tag_from_name(experiment)
        nn_path = require_dir(
            RUN_ROOT / trained_on / experiment
            / covariates.input_var,
            f"Trained model {experiment}",
        )
        suffix = permute_suffix()
        analysis_nn_path = (
            results_dir(trained_on, lpf_tag) / experiment / covariates.input_var
        )
        out_dir = ensure_dir(analysis_nn_path / "RealWorld")
        out_file = out_dir / f"Pred_RealWorld{suffix}.mat"
        print("===", experiment)
        training_config = validate_training_covariates(
            nn_path, tuple(covariates.names))
        trained_moc_convention = training_config.get("moc_convention", "absolute")
        if trained_moc_convention == "anomaly_2004_2009":
            # The training reference state locates the cell cores.
            moc_baseline = load_training_moc_baseline(
                nn_path, lat, rho2, expected_period=BASELINE_YEARS
            )
        else:
            moc_baseline = None
        assembled = assemble_inputs(covariates, lpf=bool(lpf_tag))
        x = assembled.values
        print("input:", x.shape)

        # Confirm mascon coordinates and order, not just feature count.
        info = load_npz_or_mat(nn_path / "inputs_info",
                               ["mascon_lon", "mascon_lat"])
        train_lon = np.asarray(info["mascon_lon"]).squeeze()
        train_lat = np.asarray(info["mascon_lat"]).squeeze()
        if not (assembled.lon.shape == train_lon.shape
                and np.allclose(assembled.lon, train_lon, equal_nan=True)
                and np.allclose(assembled.lat, train_lat, equal_nan=True)):
            raise RuntimeError(
                "Observation mascon coordinates/order differ from the "
                f"training mascons recorded in {nn_path / 'inputs_info.mat'}")

        n_folds = int(training_config["num_folds"])
        n_ensembles = int(training_config["nn_repeats"])
        if n_folds < 1 or n_ensembles < 1:
            raise RuntimeError(
                f"Invalid ensemble dimensions in {nn_path / 'training_config.json'}: "
                f"num_folds={n_folds}, nn_repeats={n_ensembles}")
        print(f"ensemble: {n_folds} folds x {n_ensembles} members per fold")
        ensemble = TrainedEnsemble.load(nn_path, n_folds, n_ensembles)
        for fold, scaler in enumerate(ensemble.scalers_x, start=1):
            expected = getattr(scaler, "n_features_in_", x.shape[1])
            if expected != x.shape[1]:
                raise RuntimeError(
                    f"Fold {fold} scaler expects {expected} input features, "
                    f"but aligned observations provide {x.shape[1]}")
        members = ensemble.predict_all_members(x)  # [25, T, n_out]
        mask = np.asarray(
            load_npz_or_mat(nn_path / "Psi_mask", ["Psi_mask"])["Psi_mask"]
        ).squeeze().astype(bool)
        n_cells = rho2.size * lat.size
        if mask.shape != (n_cells,):
            raise RuntimeError(
                f"Training MOC mask has shape {mask.shape}; expected ({n_cells},)"
            )
        flat_members = members.reshape(-1, members.shape[-1])
        members = unflatten(flat_members, mask, rho2.size, lat.size).reshape(
            members.shape[0], members.shape[1], rho2.size, lat.size
        )

        pred_yz = members.mean(axis=0)   # [T, lev, lat]
        pred_yz_std = members.std(axis=0)

        # Fix the YYYY-MM string width to avoid padded MAT-file dates.
        time_labels = np.datetime_as_string(
            assembled.time_month, unit="M"
        ).astype("U7")
        t_year = decimal_year(assembled.time_month)
        payload = {"pred_yz": pred_yz, "pred_yz_std": pred_yz_std,
                   "lat": lat, "rho2": rho2,
                   "time_month": time_labels, "t_year": t_year,
                   "time_calendar": "proleptic_gregorian",
                   "input_covariates": np.asarray(
                       assembled.covariate_names, dtype="U"),
                   "input_sources": np.asarray(
                       assembled.source_names, dtype="U"),
                   "input_source_files": np.asarray(
                       assembled.source_files, dtype="U"),
                   "input_baseline_specs": np.asarray(
                       assembled.baseline_specs, dtype="U"),
                    "wind_convention": (
                        "anomaly_2004_2009"
                        if _wind_covariate_name(assembled.covariate_names)
                        is not None else "none"),
                    "moc_convention": trained_moc_convention,
                    "run_id": ACTIVE_RUN_ID,
                   "training_source_run_id": ACTIVE_RUN_ID,
                   "cmip_dataset_id": CMIP_DATASET_ID,
                   "satellite_dataset_id": SATELLITE_DATASET_ID,
                   "trained_on": trained_on,
                   "training_experiment": experiment,
                    "ensemble_num_folds": np.int64(n_folds),
                    "ensemble_members_per_fold": np.int64(n_ensembles)}
        if _wind_covariate_name(assembled.covariate_names) is not None:
            payload["wind_baseline_years"] = np.asarray(BASELINE_YEARS)
        if moc_baseline is not None:
            payload["moc_baseline"] = moc_baseline
            payload["moc_baseline_years"] = np.asarray(BASELINE_YEARS)
        if not suffix:
            # Stage 15 uses baseline member trajectories for trend spread.
            payload["pred_yz_members"] = members.astype(np.float32)
        save_mat(out_file, payload)
        print("saved", out_file)

        # The contextual mean panel adds the training baseline; the network
        # reconstructs anomalies, not the absolute real-ocean mean MOC.
        qc_baseline = moc_baseline
        mean_display = pred_yz.mean(axis=0)
        mean_title = "reconstructed anomaly mean (no baseline available)"
        if qc_baseline is not None:
            mean_display = mean_display + qc_baseline
            mean_title = ("ACCESS 2004-2009 baseline + reconstructed "
                          "anomaly mean")
        for field, cmap, vlim, label, stem, title in [
            (mean_display, CMAP_DIVERGING, (-20, 20), "Sv",
             "Pred_meanMOC_RealWorld", mean_title),
            (pred_yz.std(axis=0), CMAP_AMPLITUDE, (0, 5), "Sv",
             "Pred_STD_RealWorld", None),
        ]:
            fig, _, _ = section_row(field, lat, sigma2, cmap=cmap,
                                    vmin=vlim[0], vmax=vlim[1], cbar_label=label)
            if title:
                fig.suptitle(title, y=1.02,
                             fontsize=plt.rcParams["font.size"])
            save_figure(fig, out_dir / f"{stem}{suffix}", formats=("png",))

        overview_figure(pred_yz, pred_yz_std, lat, sigma2, t_year,
                        out_dir / f"Pred_Overview_RealWorld{suffix}",
                        time_month=assembled.time_month,
                        baseline=moc_baseline)

        print(f"reconstruction: min {np.nanmin(pred_yz):.2f}, "
              f"max {np.nanmax(pred_yz):.2f}, std {np.nanstd(pred_yz):.2f} Sv")


if __name__ == "__main__":
    _args = parse_args()
    main(_args.trained_on, _args.experiment, _args.obp_source,
         _args.ssh_source, _args.wind_source, _args.covariates)
