"""Stage 06: build the NN training arrays from the per-realization files.

For each variable the realizations are loaded, reduced to
Atlantic + Southern Ocean mascons, optionally basin-demeaned, low-pass
filtered per realization, concatenated, and saved as one .npz.

Run `python scripts/06_make_model_ready_data.py --help` for options. The
selected wind variable must match the stage-04 product.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.cmip_io import (
    load_mascon_realization,
    load_moc_realization,
    parse_realizations,
    wind_output_slot,
    wind_units,
)
from neurmoc.config import (
    BASELINE_YEARS,
    LPF_TRAIN,
    MOC_CONVENTION,
    TEST_REALIZATIONS,
    TRAIN_REALIZATIONS,
    cmip_interim_dir,
    experiment_dir,
    mascon_var,
)
from neurmoc.filtering import lowpass, mask_incomplete_series
from neurmoc.io_utils import ensure_dir, require_dir
from neurmoc.plotting.style import apply_style, save_figure

# ========== User settings (overridable from the command line) ==========
#: CMIP experiment folder: "ACCESS_historical" | "ACCESS_SSP126/245/370/585" |
#: "MRI_SSP245" | ...
EXPERIMENT = "ACCESS_historical"
#: Realization subset: train, test, all, test_ext, r1, or a range such as 1-5.
MODE = "train"
QC_PLOTS = True

#: Wind input to assemble from stage-04 products: uas, tauu, curltau, or ua@<Pa>.
WIND_VAR = "uas"


def predictors(wind_var: str = WIND_VAR):
    """(output name, per-realization file tag, remove basin mean?).

    The wind row follows the selected wind product and its output slot.
    """
    slot = wind_output_slot(wind_var)
    return [
        (mascon_var("obp"), "OBP", True),
        (mascon_var("ssh"), "SSH", True),
        (mascon_var(slot), slot, False),
    ]


MODES = {
    "train": (TRAIN_REALIZATIONS, "_r1_r35"),
    "test": (TEST_REALIZATIONS, "_r36_r40"),
    "all": (range(1, 41), "_r1_r40"),
    "test_ext": (range(1, 6), "_r1_r5"),  # 2100-2300 extension runs
    "r1": (range(1, 2), "_r1_r1"),        # single realization (cross-model tests)
}


def resolve_mode(mode: str):
    """Named MODES entry, or an explicit realization spec like "1-5".

    Non-contiguous specs get an extra `_n<count>` tag suffix (matching the
    stage-03 baseline convention), so e.g. "1,6-10" -> `_r1_r10_n6` cannot
    collide with or overwrite a true contiguous r1-r10 dataset."""
    if mode in MODES:
        return MODES[mode]
    realizations = parse_realizations(mode)
    tag = f"_r{realizations[0]}_r{realizations[-1]}"
    if len(realizations) != realizations[-1] - realizations[0] + 1:
        tag += f"_n{len(realizations)}"
    return realizations, tag


def std_by_realization(arr, rlz_index):
    """Std over time within each realization block, averaged over blocks
    (a std across the concatenation would mix inter-realization offsets)."""
    return np.mean([arr[rlz_index == r].std(axis=0)
                    for r in np.unique(rlz_index)], axis=0)


def qc_scatter(lon, lat, values, title, out_stem):
    fig, ax = plt.subplots(figsize=(6, 2.8), constrained_layout=True)
    sc = ax.scatter(lon, lat, c=values, s=4, cmap="OrRd", linewidths=0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, fraction=0.03)
    save_figure(fig, out_stem, formats=("png",))


def process_predictor(data_dir, name, file_tag, demean, realizations, tag, out_dir):
    print(f"=== {name} ===")
    raw_list, lpf_list, rlz_index, time_list = [], [], [], []
    lon = lat = None
    provenance: dict[str, dict[int, str]] = {}
    for r in realizations:
        loaded = load_mascon_realization(data_dir, file_tag, r)
        atl = loaded["basin_id"] == 1
        data = loaded["data"][:, atl]
        lon, lat = loaded["lon"][atl], loaded["lat"][atl]
        if loaded.get("time_month") is None:
            raise ValueError(
                f"{name} r{r} has no time_month coordinate; rerun stage 04")
        time_list.append(np.asarray(loaded["time_month"]).astype("U7"))
        for key, value in loaded.get("provenance", {}).items():
            provenance.setdefault(key, {})[r] = value

        if demean:
            data = data - np.nanmean(data, axis=1, keepdims=True)

        # Zero-fill permanently dry mascons; reject intermittent gaps.
        missing = np.isnan(data)
        partial_nan = missing.any(axis=0) & ~missing.all(axis=0)
        if partial_nan.any():
            examples = np.flatnonzero(partial_nan)[:5].tolist()
            raise RuntimeError(
                f"{name} r{r}: {int(partial_nan.sum())} mascons have partial "
                f"monthly coverage (column examples {examples}); repair the "
                "stage-04 source or choose an explicit gap-filling method"
            )
        all_nan = missing.all(axis=0)
        if all_nan.any():
            print(
                f"  WARNING: {int(all_nan.sum())} of {data.shape[1]} mascons "
                "have no wet cells on this model grid; filled with 0 after "
                "demeaning"
            )
            data = data.copy()
            data[:, all_nan] = 0.0
        data_lpf = lowpass(data, LPF_TRAIN)

        raw_list.append(data)
        lpf_list.append(data_lpf)
        rlz_index.extend([r] * data.shape[0])
        print(f"  r{r}: {data.shape[0]} months, {data.shape[1]} mascons")

    # all realizations must have been built alike (same baseline, same wind)
    for key, by_r in provenance.items():
        if len(set(by_r.values())) > 1:
            raise RuntimeError(
                f"{name}: inconsistent {key} across realizations - some "
                f"stage-04 files were built with different settings: {by_r}. "
                "Rerun stage 04 for the full realization range.")
        print(f"  {key}: {next(iter(by_r.values()))}")

    raw = np.concatenate(raw_list, axis=0)
    lpf = np.concatenate(lpf_list, axis=0)
    time_month = np.concatenate(time_list)

    out_file = out_dir / f"{name}{tag}.npz"
    np.savez(out_file, **{
        f"{name}_ALL": raw, f"{name}_LPF_ALL": lpf,
        "mascon_lon": lon, "mascon_lat": lat,
        "realization_index": np.asarray(rlz_index),
        "time_month": time_month,
        # carry the stage-04 provenance forward (when present)
        **{key: next(iter(by_r.values())) for key, by_r in provenance.items()},
    })
    print("  saved", out_file)

    rlz = np.asarray(rlz_index)
    if QC_PLOTS:
        qc_scatter(lon, lat, std_by_realization(raw, rlz), f"STD of {name}",
                   out_dir / f"STD_{name}{tag}")

    # Select the highest-variance mascon in the first realization for QC.
    first = rlz == rlz[0]
    j = int(np.nanargmax(raw[first].std(axis=0)))
    label = f"{name.split('_')[0]} @ ({lon[j]:.0f}\N{DEGREE SIGN}, {lat[j]:.0f}\N{DEGREE SIGN})"
    return label, raw[first, j], lpf[first, j]


def historical_experiment_of(experiment: str) -> str:
    """Historical experiment supplying an experiment's baseline."""
    if experiment.endswith("_historical"):
        return experiment
    return f"{experiment.rsplit('_', 1)[0]}_historical"


def moc_baseline_2004_2009(experiment: str):
    """Cached 2004-2009 historical ensemble-mean MOC on the target grid."""
    hist_dir = cmip_interim_dir(historical_experiment_of(experiment))
    cache = hist_dir / "MOC_TimeMean_2004_2009.npz"
    members = sorted(int(p.stem.split("_r")[-1])
                     for p in hist_dir.glob("FullDepth_ASMOC_interp_gr_r*.npz"))
    if cache.is_file():
        with np.load(cache) as fh:
            cached_members = fh["baseline_realizations"].tolist()
            if members and cached_members != members:
                # Reject a cached baseline from a different member set.
                raise RuntimeError(
                    f"{cache}: cached baseline uses members {cached_members} "
                    f"but the stage-05 folder now holds {members}. Delete the "
                    "cache to rebuild the baseline from the current ensemble "
                    "(this changes the anomaly reference of every dataset "
                    "derived from it).")
            cached_baseline = fh["MOC_2004_2009_mean"]
        member_paths = [
            hist_dir / f"FullDepth_ASMOC_interp_gr_r{member}.npz"
            for member in cached_members
        ]
        if not any(
            path.stat().st_mtime_ns > cache.stat().st_mtime_ns
            for path in member_paths
        ):
            return cached_baseline, cached_members
        print(f"  historical MOC is newer than {cache.name}; rebuilding baseline")
    if not members:
        raise FileNotFoundError(
            f"no historical stage-05 MOC files in {hist_dir}; run stage 05 on "
            f"{historical_experiment_of(experiment)} before building MOC "
            "anomalies (the 2004-2009 baseline needs the target-grid MOC)")
    lo = np.datetime64(f"{BASELINE_YEARS[0]:04d}-01", "M")
    hi = np.datetime64(f"{BASELINE_YEARS[1]:04d}-12", "M")
    per_member = []
    for r in members:
        loaded = load_moc_realization(hist_dir, r)
        psi = np.transpose(loaded["psi"], (0, 2, 1)) / 1e6  # [time, lat, lev], Sv
        months = np.asarray(loaded["time_month"]).astype("datetime64[M]")
        sel = (months >= lo) & (months <= hi)
        n_expected = 12 * (BASELINE_YEARS[1] - BASELINE_YEARS[0] + 1)
        if sel.sum() != n_expected:
            raise RuntimeError(
                f"{historical_experiment_of(experiment)} r{r}: found "
                f"{sel.sum()} baseline months, expected {n_expected} "
                f"({BASELINE_YEARS[0]}-01..{BASELINE_YEARS[1]}-12)")
        # Retain valid months in intermittently outcropping cells.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            per_member.append(np.nanmean(psi[sel], axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        baseline = np.nanmean(per_member, axis=0)   # [lat, lev]
    np.savez(cache, MOC_2004_2009_mean=baseline,
             baseline_realizations=np.asarray(members, dtype=int),
             baseline_period=np.asarray(BASELINE_YEARS, dtype=int),
             source_experiment=historical_experiment_of(experiment))
    print(f"  computed MOC 2004-2009 baseline from {len(members)} "
          f"{historical_experiment_of(experiment)} members -> {cache.name}")
    return baseline, members


def process_moc(data_dir, realizations, tag, out_dir, experiment):
    print("=== MOC ===")
    raw_list, lpf_list, rlz_index, time_list = [], [], [], []
    rho2 = lat = None
    provenance: dict[str, dict[int, str]] = {}
    for r in realizations:
        loaded = load_moc_realization(data_dir, r)
        psi = np.transpose(loaded["psi"], (0, 2, 1)) / 1e6  # [time, lat, lev], Sv
        rho2, lat = loaded["rho2"], loaded["lat"]
        if loaded.get("time_month") is None:
            raise ValueError(
                f"MOC r{r} has no time_month coordinate; rerun stage 05")
        time_list.append(np.asarray(loaded["time_month"]).astype("U7"))
        for key, value in loaded.get("provenance", {}).items():
            provenance.setdefault(key, {})[r] = value
        raw_list.append(psi)
        rlz_index.extend([r] * psi.shape[0])
        print(f"  r{r}: {psi.shape[0]} months")

    # the realizations being stacked must come from one model/experiment
    for key, by_r in provenance.items():
        if len(set(by_r.values())) > 1:
            raise RuntimeError(
                f"MOC: realizations disagree on {key}: {by_r}")

    raw = np.concatenate(raw_list, axis=0)

    moc_extra = {}
    if MOC_CONVENTION == "anomaly_2004_2009":
        # Reference MOC to its own historical 2004-2009 ensemble mean.
        baseline, base_members = moc_baseline_2004_2009(experiment)  # [lat, lev]
        if baseline.shape != raw.shape[1:]:
            raise RuntimeError(
                f"MOC baseline shape {baseline.shape} != record {raw.shape[1:]}; "
                "the historical baseline and this experiment are on different grids")
        raw = raw - baseline
        moc_extra = {
            "moc_convention": np.str_(MOC_CONVENTION),
            # Preserve the absolute reference state for cell-core selection.
            "MOC_baseline_mean": baseline,
            "moc_baseline_period": np.asarray(BASELINE_YEARS, dtype=int),
            "moc_baseline_realizations": np.asarray(base_members, dtype=int),
            "moc_baseline_experiment": np.str_(
                historical_experiment_of(experiment)),
        }
    else:
        moc_extra = {"moc_convention": np.str_("absolute")}

    # Exclude incomplete target series before filtering.
    raw, n_partial_target = mask_incomplete_series(raw)
    if n_partial_target:
        print(
            f"  WARNING: masked {n_partial_target} MOC cells with intermittent "
            "outcrop/missing months before LPF"
        )

    # LPF each realization block after the (constant) baseline subtraction
    lpf_list = [lowpass(raw[np.asarray(rlz_index) == r], LPF_TRAIN)
                for r in realizations]
    lpf = np.concatenate(lpf_list, axis=0)
    time_month = np.concatenate(time_list)
    out_file = out_dir / f"MOC{tag}.npz"
    np.savez(out_file, MOC_ALL=raw, MOC_LPF_ALL=lpf, rho2_full=rho2, lat_psi=lat,
             realization_index=np.asarray(rlz_index), time_month=time_month,
             **moc_extra,
             # carry the stage-05 provenance forward (when present)
             **{key: next(iter(by_r.values()))
                for key, by_r in provenance.items()})
    print(f"  saved {out_file}  ({moc_extra['moc_convention']})")

    rlz = np.asarray(rlz_index)
    if QC_PLOTS:
        sigma2 = rho2 - 1000 if rho2[0] > 1000 else rho2
        # Show the full time-mean MOC in the QC panel.
        mean_display = raw.mean(axis=0)
        if "MOC_baseline_mean" in moc_extra:
            mean_display = mean_display + moc_extra["MOC_baseline_mean"]
        panels = [
            (mean_display, "RdBu_r", -25, 25, "time-mean MOC (Sv)"),
            (std_by_realization(raw, rlz), "Spectral_r", 0, 5,
             "std of the raw monthly MOC (Sv) - includes seasonal cycle "
             "and forced trend"),
            (std_by_realization(lpf, rlz), "Spectral_r", 0, 5,
             "std of the 2-year low-passed MOC (Sv) - what the network "
             "is trained on"),
        ]
        fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.2), constrained_layout=True)
        for ax, (field, cmap, vmin, vmax, title) in zip(axes, panels):
            ax.set_facecolor("#d9d9d9")
            pm = ax.pcolormesh(lat, sigma2, np.ma.masked_invalid(field).T,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               shading="nearest", rasterized=True)
            ax.invert_yaxis()
            ax.set_ylabel(r"$\sigma_2$ (kg m$^{-3}$)")
            ax.set_title(title, loc="left")
            fig.colorbar(pm, ax=ax, fraction=0.03)
        axes[-1].set_xlabel("Latitude")
        save_figure(fig, out_dir / f"QC_FullDepth_MOC{tag}", formats=("png",))

    # Use the 26.5 degrees N AMOC core as the MOC QC series.
    first = rlz == rlz[0]
    j26 = int(np.argmin(np.abs(lat - 26.5)))
    lev = int(np.nanargmax(raw[first].mean(axis=0)[j26]))
    label = (f"MOC @ 26.5\N{DEGREE SIGN}N, $\\sigma_2$ = "
             f"{(rho2[lev] - 1000 if rho2[0] > 1000 else rho2[lev]):.2f}")
    return label, raw[first, j26, lev], lpf[first, j26, lev]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-e", "--experiment", default=EXPERIMENT,
                        help="CMIP experiment folder under the data root")
    parser.add_argument("-m", "--mode", default=MODE,
                        help="realization subset: " + " | ".join(sorted(MODES))
                             + " | explicit spec like 1-5 or 1,6-10")
    parser.add_argument("-w", "--wind-var", default=WIND_VAR,
                        help="wind input assembled: uas | tauu | ua@<Pa> "
                             "(must match the stage-04 run)")
    parser.add_argument(
        "--wind-only", action="store_true",
        help="build ONLY the wind array, reusing the existing OBP/SSH/MOC "
             "ones (they do not depend on the wind input, and rewriting "
             "them would give byte-identical files new mtimes, which the "
             "staleness gate reads as a change)")
    return parser.parse_args()


def qc_lpf_figure(series, units, tag, out_dir):
    """Raw monthly vs 2-year-filtered series at representative points
    (first realization), one row per variable - illustrates what LPF_TRAIN
    removes (seasonal cycle and monthly noise) and keeps (interannual
    variability and trend)."""
    # Keep axes two-dimensional for single-series runs.
    fig, axes = plt.subplots(len(series), 1, figsize=(6.5, 1.8 * len(series)),
                             sharex=True, constrained_layout=True,
                             squeeze=False)
    for ax, (label, raw_1d, lpf_1d), unit in zip(axes[:, 0], series, units):
        t = np.arange(raw_1d.size) / 12.0
        ax.plot(t, raw_1d, color="0.75", lw=0.5, label="raw monthly")
        ax.plot(t, lpf_1d, color="#0072B2", lw=1.2, label="2-year low-pass")
        ax.set_title(label, loc="left")
        ax.set_ylabel(unit)
        ax.tick_params(top=False, right=False)
    axes[0, 0].legend(ncol=2, loc="upper right")
    axes[-1, 0].set_xlabel("Years since record start")
    save_figure(fig, out_dir / f"QC_LPF{tag}", formats=("png",))


def main(experiment: str = EXPERIMENT, mode: str = MODE,
         wind_var: str = WIND_VAR, wind_only: bool = False) -> None:
    apply_style()
    data_dir = require_dir(cmip_interim_dir(experiment), "Interim CMIP experiment data")
    realizations, tag = resolve_mode(mode)
    out_dir = ensure_dir(experiment_dir(experiment))

    series = []
    predictor_rows = predictors(wind_var)
    if wind_only:
        # Rebuild only the wind array; require and align the existing
        # wind-independent arrays without changing their timestamps.
        build_rows = predictor_rows[-1:]
        missing = [out_dir / f"{name}{tag}.npz"
                   for name, *_ in predictor_rows[:-1]
                   if not (out_dir / f"{name}{tag}.npz").is_file()]
        if not (out_dir / f"MOC{tag}.npz").is_file():
            missing.append(out_dir / f"MOC{tag}.npz")
        if missing:
            raise FileNotFoundError(
                "--wind-only reuses the existing OBP/SSH/MOC arrays, but "
                f"{[p.name for p in missing]} are absent. Run stage 06 once "
                "without --wind-only for this experiment first.")
        print(f"wind-only: building {build_rows[0][0]} only; OBP/SSH/MOC "
              "reused untouched")
    else:
        build_rows = predictor_rows
    for name, file_tag, demean in build_rows:
        series.append(process_predictor(data_dir, name, file_tag, demean,
                                        realizations, tag, out_dir))
    if not wind_only:
        series.append(process_moc(data_dir, realizations, tag, out_dir,
                                  experiment))

    # All model-ready arrays must describe exactly the same member/month rows.
    expected_time = expected_realizations = None
    for name, *_ in predictor_rows:
        path = out_dir / f"{name}{tag}.npz"
        with np.load(path) as data:
            time = np.asarray(data["time_month"])
            members = np.asarray(data["realization_index"])
        if expected_time is None:
            expected_time, expected_realizations = time, members
        elif not (np.array_equal(time, expected_time)
                  and np.array_equal(members, expected_realizations)):
            raise RuntimeError(f"{path}: rows are not aligned with the other predictors")
    with np.load(out_dir / f"MOC{tag}.npz") as data:
        if not (np.array_equal(data["time_month"], expected_time)
                and np.array_equal(data["realization_index"], expected_realizations)):
            raise RuntimeError("MOC rows are not aligned with predictor rows")

    if QC_PLOTS:
        # One panel per series actually built, in the order they were built.
        units = ["Pa", "m", wind_units(wind_var), "Sv"]
        if wind_only:
            units = [wind_units(wind_var)]
        qc_lpf_figure(series, units, tag, out_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args.experiment, args.mode, args.wind_var, args.wind_only)
