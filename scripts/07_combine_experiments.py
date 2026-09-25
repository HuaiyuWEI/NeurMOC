"""Stage 07: concatenate the training data of several CMIP experiments.

For example, build `ACCESS_hist+SSP585` from `ACCESS_historical` and
`ACCESS_SSP585`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.cmip_io import (
    canonical_wind_var,
    wind_output_slot,
)
from neurmoc.config import (
    SCIENTIFIC_CONFIG,
    TRAIN_REALIZATIONS,
    experiment_dir,
    mascon_var,
)
from neurmoc.io_utils import ensure_dir, require_dir, require_file

# ========== User settings ==========
INPUT_EXPERIMENTS = list(SCIENTIFIC_CONFIG.get(
    "training_input_datasets", ["ACCESS_historical", "ACCESS_SSP585"]
))
OUTPUT_EXPERIMENT = str(
    SCIENTIFIC_CONFIG.get("training_dataset", "ACCESS_hist+SSP585")
)
TAG = f"_r{TRAIN_REALIZATIONS.start}_r{TRAIN_REALIZATIONS.stop - 1}"

#: Wind input combined: "uas" | "tauu" | "curltau"
#: | "ua@<Pa>". Must match the stage-06 arrays being combined.
WIND_VAR = "uas"


def variables(wind_var: str = WIND_VAR) -> list[str]:
    """The four combined arrays, for the selected wind input."""
    return [mascon_var("obp"), mascon_var("ssh"),
            mascon_var(wind_output_slot(wind_var)), "MOC"]


COORD_KEYS = {
    "MOC": ["rho2_full", "lat_psi"],
    "default": ["mascon_lon", "mascon_lat"],
}
#: Optional coordinate and baseline arrays must match across experiments.
OPTIONAL_COORD_KEYS = {
    "MOC": ["MOC_baseline_mean", "moc_baseline_period",
            "moc_baseline_realizations"],
    "default": [],
}


def _arrays_match(a: np.ndarray, b: np.ndarray) -> bool:
    """array_equal with NaN==NaN for float arrays (int-safe)."""
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def combine_variable(varname: str, input_dirs: list[Path], out_dir: Path) -> None:
    print(f"=== {varname} ===")
    stacked = {
        f"{varname}_ALL": [], f"{varname}_LPF_ALL": [],
        "realization_index": [], "time_month": [], "source_experiment": [],
    }
    coords = {}
    coord_keys = COORD_KEYS["MOC" if varname == "MOC" else "default"]
    optional_seen: dict[str, int] = {}
    provenance: dict[str, dict[str, str]] = {}

    for data_dir in input_dirs:
        path = require_file(data_dir / f"{varname}{TAG}.npz", varname)
        print("  loading", path)
        with np.load(path) as data:
            for key in stacked:
                if key == "source_experiment":
                    n_samples = data[f"{varname}_ALL"].shape[0]
                    # Use the experiment directory, not its dataset parent.
                    stacked[key].append(np.full(
                        n_samples, data_dir.name, dtype="U64"))
                else:
                    stacked[key].append(data[key])
            for key in coord_keys:
                value = np.asarray(data[key])
                if key in coords and not np.array_equal(coords[key], value):
                    raise ValueError(
                        f"{path}: coordinate {key!r} differs from the other "
                        "input experiment"
                    )
                coords[key] = value
            for key in OPTIONAL_COORD_KEYS["MOC" if varname == "MOC" else "default"]:
                if key not in data.files:
                    continue
                value = np.asarray(data[key])
                if key in coords and not _arrays_match(coords[key], value):
                    raise ValueError(
                        f"{path}: {key!r} differs between the input "
                        "experiments - they do not share one 2004-2009 "
                        "MOC baseline")
                coords[key] = value
                optional_seen[key] = optional_seen.get(key, 0) + 1
            for key in (
                "baseline_spec", "baseline_period", "source_model",
                "wind_source", "wind_convention", "wind_anomaly",
                "wind_calibration", "moc_convention",
                "moc_baseline_experiment",
            ):
                if key in data.files:
                    provenance.setdefault(key, {})[data_dir.name] = str(data[key])

    # the experiments being concatenated must have been built alike
    for key, by_experiment in provenance.items():
        if len(set(by_experiment.values())) > 1:
            raise RuntimeError(
                f"{varname}: the input experiments were built with different "
                f"{key} settings and must not be combined: {by_experiment}")
    for key, count in optional_seen.items():
        if count != len(input_dirs):
            raise RuntimeError(
                f"{varname}: {key!r} present in {count}/{len(input_dirs)} "
                "input experiments - mixed MOC conventions must not be "
                "combined (rebuild both with the same stage-06 settings)")

    payload = {key: np.concatenate(vals, axis=0) for key, vals in stacked.items()}
    payload.update(coords)
    payload.update({key: next(iter(by_e.values()))
                    for key, by_e in provenance.items()})
    out_file = out_dir / f"{varname}{TAG}.npz"
    np.savez(out_file, **payload)
    print("  saved", out_file, "->", payload[f"{varname}_ALL"].shape)


def main(wind_var: str = WIND_VAR, wind_only: bool = False) -> None:
    all_variables = variables(wind_var)
    input_dirs = [require_dir(experiment_dir(name), name)
                  for name in INPUT_EXPERIMENTS]
    out_dir = ensure_dir(experiment_dir(OUTPUT_EXPERIMENT))
    if wind_only:
        # Reuse wind-independent arrays without changing their timestamps;
        # require and align them with the new wind array.
        built = [all_variables[2]]
        absent = [out_dir / f"{v}{TAG}.npz" for v in all_variables
                  if v != built[0] and not (out_dir / f"{v}{TAG}.npz").is_file()]
        if absent:
            raise FileNotFoundError(
                "--wind-only reuses the existing combined arrays, but "
                f"{[p.name for p in absent]} are absent. Run stage 07 once "
                "without --wind-only first.")
        print(f"wind-only: combining {built[0]} only; "
              "OBP/SSH/MOC reused untouched")
    else:
        built = all_variables
    for varname in built:
        combine_variable(varname, input_dirs, out_dir)

    # Require aligned experiment/member/month rows across all arrays.
    expected_rows = None
    for varname in all_variables:
        path = out_dir / f"{varname}{TAG}.npz"
        with np.load(path) as data:
            rows = (
                np.asarray(data["source_experiment"]),
                np.asarray(data["realization_index"]),
                np.asarray(data["time_month"]),
            )
        if expected_rows is None:
            expected_rows = rows
        elif not all(
            np.array_equal(actual, expected)
            for actual, expected in zip(rows, expected_rows)
        ):
            raise RuntimeError(f"{path}: combined rows are not aligned")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-w", "--wind-var", default=WIND_VAR,
        help="wind input combined: uas (default) | tauu | curltau | ua@<Pa>; "
             "must match the stage-06 arrays")
    parser.add_argument(
        "--wind-only", action="store_true",
        help="combine ONLY the wind array, reusing the existing combined "
             "OBP/SSH/MOC (rewriting them would give byte-identical files "
             "new mtimes, which the staleness gate reads as a change)")
    args = parser.parse_args()
    args.wind_var = canonical_wind_var(args.wind_var)
    return args


if __name__ == "__main__":
    _args = parse_args()
    main(_args.wind_var, _args.wind_only)
