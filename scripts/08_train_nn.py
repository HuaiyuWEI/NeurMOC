"""Stage 08: train the dual-branch neural network on CMIP6 data.

Hyperparameters are specified by `TrainingConfig`, which also determines the
experiment output directory. The ensemble is trained with five-fold,
realization-wise cross-validation and five random initializations per fold.

Start each training in a fresh Python process because TensorFlow graph mode
cannot be reset within an interpreter. Run `python scripts/08_train_nn.py
--help` for configuration overrides.
"""

import argparse
import dataclasses
import logging
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import SCIENTIFIC_CONFIG, experiment_dir, results_dir
from neurmoc.io_utils import ensure_dir, require_dir
from neurmoc.model import configure_tensorflow_runtime
from neurmoc.naming import model_output_dir, training_config_from_scientific
from neurmoc.training import load_training_data, plot_training_losses, train_experiment

# ========== User settings ==========
CONFIG = training_config_from_scientific(SCIENTIFIC_CONFIG)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--cmip-name", default=None,
                        help="training experiment (e.g. ACCESS_hist+SSP126)")
    parser.add_argument("-x", "--covariates", default=None,
                        help="comma-separated covariate list")
    parser.add_argument("--pca-y", type=int, default=None,
                        help="target PCA components (folder PCAinY<n>); "
                             "0 = no PCA (FullDepth_ResNet_... folder)")
    parser.add_argument("--set", dest="overrides", action="append",
                        metavar="FIELD=VALUE", default=None,
                        help="override any TrainingConfig field by name "
                             "(repeatable), e.g. --set reg_strength=0.003 "
                             "--set neurons=256x128x64; the experiment "
                             "folder name reflects every override")
    return parser.parse_args()


def _coerce(current, raw: str):
    """Parse a --set value using the field's current (default) type."""
    if isinstance(current, bool):  # before int: bool is an int subclass
        return raw.strip().lower() in ("true", "1", "yes")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):  # neurons: "512x256x128x64" or "512,256"
        return [int(part) for part in raw.replace("x", ",").split(",") if part]
    return raw


def apply_overrides(tc, overrides: list[str]):
    """Apply FIELD=VALUE overrides to a TrainingConfig."""
    valid = {f.name for f in dataclasses.fields(tc)}
    for item in overrides:
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep or name not in valid:
            raise ValueError(
                f"--set {item!r}: expected FIELD=VALUE with FIELD one of "
                f"{sorted(valid)}")
        tc = replace(tc, **{name: _coerce(getattr(tc, name), raw.strip())})
    return tc


def main(cmip_name: str | None = None, covariates: str | None = None,
         pca_y: int | None = None,
         overrides: list[str] | None = None) -> None:
    tc = CONFIG
    if overrides:
        tc = apply_overrides(tc, overrides)
    if cmip_name:
        tc = replace(tc, cmip_name=cmip_name)
    if covariates:
        tc = replace(tc, covariate_names=covariates)
    if pca_y is not None:
        tc = replace(tc, pca_y_num=pca_y)

    # Stage 06 stores unfiltered (_ALL) and 24-month filtered (_LPF_ALL) arrays.
    if tc.lpf_months not in (0, 24):
        raise ValueError(f"lpf_months must be 0 or 24, not {tc.lpf_months}")

    data_dir = require_dir(experiment_dir(tc.cmip_name), f"Training data for {tc.cmip_name}")
    data = load_training_data(tc, data_dir)
    output_dir = ensure_dir(model_output_dir(
        results_dir(tc.cmip_name, tc.lpf_tag),
        tc.experiment_name(), tc.covariates.input_var,
    ))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        force=True,  # Spyder/IPython may already have configured root handlers
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "training.log"),
        ],
    )
    logging.info("experiment: %s", tc.experiment_name())
    logging.info("output dir: %s", output_dir)
    configure_tensorflow_runtime(seed=tc.random_seed, require_fresh_console=True)

    summary = train_experiment(tc, data, output_dir)
    plot_training_losses(summary, tc, output_dir)

    # Keep a copy of this script with the trained networks.
    shutil.copy(__file__, output_dir / Path(__file__).name)
    logging.info("done.")


if __name__ == "__main__":
    args = parse_args()
    main(args.cmip_name, args.covariates, args.pca_y, args.overrides)
