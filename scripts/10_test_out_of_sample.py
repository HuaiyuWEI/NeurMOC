"""Stage 10: evaluate trained networks on held-out simulations.

Apply each trained ensemble to registered ACCESS or MRI cases and
save predictions and latitude-density skill fields. Evaluation inputs must
be prepared before this stage. Run with --help for case selection.
"""

import argparse
import gc
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.cmip_io import load_evaluation_data
from neurmoc.config import (
    CMIP_ROOT,
    LPF_DECADAL,
    SCIENTIFIC_CONFIG,
    results_dir,
)
from neurmoc.filtering import lowpass_by_realization, std_by_realization
from neurmoc.inference import TrainedEnsemble
from neurmoc.io_utils import load_npz_or_mat, require_dir, save_mat
from neurmoc.moc_utils import pointwise_r2, pointwise_rmse, unflatten
from neurmoc.model import configure_tensorflow_runtime
from neurmoc.model_test_cases import registered_case_specs, resolve_case_spec
from neurmoc.naming import (
    lpf_tag_from_name,
    prepare_covariate_config,
    training_config_from_scientific,
)
from neurmoc.plotting import (
    CMAP_AMPLITUDE,
    CMAP_DIVERGING,
    CMAP_R2,
    apply_style,
    section_row,
)
from neurmoc.plotting.style import save_figure

# ========== User settings ==========
#: Experiment the networks were trained on: "ACCESS_historical" |
#: "ACCESS_hist+SSP126" | "ACCESS_hist+SSP585" | ...
DEFAULT_TRAINING = training_config_from_scientific(SCIENTIFIC_CONFIG)
TRAINED_ON = DEFAULT_TRAINING.cmip_name
#: Input set - selects the trained (ablation) network to evaluate: any
#: comma-separated subset of the three inputs.
COVARIATE_NAMES = DEFAULT_TRAINING.covariate_names

EXPERIMENTS = [DEFAULT_TRAINING.experiment_name()]

#: Evaluation cases: (experiment, output tag, realization tag, count).
EVALUATION_CASES = list(registered_case_specs())

#: Also show the 10-year post-LPF diagnostics (extra TestR2 figure +
#: printed median). False = show only the 2-year-LPF results; the saved
#: .mat files contain both either way.
SHOW_10Y_LPF = False

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-t", "--trained-on", default=TRAINED_ON,
                        help="training experiment whose networks to evaluate")
    parser.add_argument("-c", "--case", default=None,
                        help="registered name or exact custom "
                             "NAME:OUT_TAG:RLZ_TAG:N_RLZ specification")
    parser.add_argument("-x", "--covariates", default=COVARIATE_NAMES,
                        help="comma-separated input set (selects the trained "
                             "ablation network, e.g. obp_mascon_V7,ssh_mascon_V7)")
    parser.add_argument("--show-10y-lpf", action="store_true",
                        default=SHOW_10Y_LPF,
                        help="also plot/print the 10-year post-LPF skill")
    parser.add_argument("-E", "--experiment", default=None,
                        help="hyperparameter-experiment folder name to "
                             "evaluate (default: the profile baseline), e.g. "
                             "FullDepth_PCAinY64_ResNet_Neur192x96x48_"
                             "5foldCV_Reg0.01Drop0.2_swishActivation_LPF2Year")
    return parser.parse_args()


def load_training_mask(nn_path: Path, n_outputs: int) -> np.ndarray:
    """Validity mask the network was trained on (defines its output layout)."""
    try:
        mask = np.asarray(load_npz_or_mat(nn_path / "Psi_mask", ["Psi_mask"])
                          ["Psi_mask"]).squeeze().astype(bool)
    except FileNotFoundError:
        mask = np.ones(n_outputs, dtype=bool)
    return mask


def evaluate_case(nn_path: Path, ensemble, cmip_name: str, out_tag: str,
                  covariates, rlz_tag: str, n_realizations: int,
                   show_10y_lpf: bool = SHOW_10Y_LPF,
                   lpf_key: str = "_LPF_ALL"):
    print(f"--- {cmip_name} ({out_tag}, {rlz_tag}, {n_realizations} realizations) ---")
    data_dir = require_dir(CMIP_ROOT / cmip_name, cmip_name)
    expected_wind = (
        ensemble.training_config.get("wind_convention")
        if ensemble.training_config is not None
        else None
    )
    expected_moc = (
        ensemble.training_config.get("moc_convention")
        if ensemble.training_config is not None
        else None
    )
    case = load_evaluation_data(
        data_dir,
        covariates.names,
        rlz_tag,
        lpf_key,
        expected_wind_convention=expected_wind,
        expected_moc_convention=expected_moc,
    )

    def decadal_lpf(series):
        return lowpass_by_realization(series, n_realizations, LPF_DECADAL)

    pred = ensemble.predict(case.x)

    # Compare fields on the grid using their respective validity masks.
    train_mask = load_training_mask(nn_path, case.n_lev * case.n_lat)
    truth_g = unflatten(case.y, case.psi_mask, case.n_lev, case.n_lat)
    pred_g = unflatten(pred, train_mask, case.n_lev, case.n_lat)
    truth_g_lpf, pred_g_lpf = decadal_lpf(truth_g), decadal_lpf(pred_g)

    r2 = pointwise_r2(truth_g, pred_g)
    r2_lpf = pointwise_r2(truth_g_lpf, pred_g_lpf)

    # Save full RMSE, mean bias, and realization-demeaned RMSE separately.
    err = pred_g - truth_g
    with warnings.catch_warnings():          # cells invalid in every month
        warnings.simplefilter("ignore", RuntimeWarning)
        rmse = pointwise_rmse(truth_g, pred_g)
        bias = np.nanmean(err, axis=0)
        err_m = err.reshape(n_realizations, -1, *err.shape[1:])
        err_db = err_m - np.nanmean(err_m, axis=1, keepdims=True)
        rmse_debiased = np.sqrt(
            np.nanmean(err_db.reshape(err.shape) ** 2, axis=0))

    pred_path = nn_path / f"Pred_{out_tag}.mat"
    testr2_path = nn_path / f"TestR2_{out_tag}.mat"
    save_mat(pred_path,
             {"y_pred": pred_g, "y": truth_g,
              "y_pred_10LPF": pred_g_lpf, "y_10LPF": truth_g_lpf})
    testr2 = {"r2_mean_yz": r2, "r2_mean_yz_LPFafter": r2_lpf,
              "rmse_yz": rmse, "rmse_debiased_yz": rmse_debiased,
              "bias_yz": bias,
              "rho2": case.rho2, "lat_psi": case.lat}
    if case.moc_baseline is not None:
        # Preserve the absolute reference state for cell-core selection.
        testr2["moc_baseline"] = case.moc_baseline
    save_mat(testr2_path, testr2)

    sigma2 = case.rho2 - 1000 if case.rho2[0] > 1000 else case.rho2
    # Hide low-variability cells (per-member std < 0.5 Sv) in plots; the
    # saved data retain every cell.
    hide = std_by_realization(truth_g, n_realizations) < 0.5
    panels = [(r2, "", "monthly")]
    if show_10y_lpf:
        panels.append((r2_lpf, "_10YLPFafter", "10-yr filtered"))
    for field, suffix, title in panels:
        shown = field.copy()
        shown[hide] = np.nan
        fig, _, _ = section_row(shown, case.lat, sigma2, cmap=CMAP_R2,
                                vmin=0, vmax=1, cbar_label=r"$R^2$")
        fig.suptitle(f"{out_tag}: reconstruction skill ({title})", y=1.02)
        save_figure(fig, nn_path / f"TestR2_{out_tag}{suffix}", formats=("png",))

    # Plot transfer-error diagnostics.
    fig = plt.figure(figsize=(9, 7.5))
    gs = fig.add_gridspec(3, 1, hspace=0.55,
                          left=0.08, right=0.9, top=0.92, bottom=0.06)
    fig.suptitle(f"{out_tag}: cross-model transfer error", y=0.97)
    vmax = float(np.nanpercentile(rmse, 98))
    for row, (field, title, cmap, vmin, vmax_, label) in enumerate([
        (rmse, "full RMSE (includes the static bias)",
         CMAP_AMPLITUDE, 0, vmax, "Sv"),
        (rmse_debiased, "debiased (per-member mean error removed)",
         CMAP_AMPLITUDE, 0, vmax, None),
        (bias, "static mean error (bias)",
         CMAP_DIVERGING, -vmax, vmax, "Sv"),
    ]):
        _, (ax_s, _), _ = section_row(field, case.lat, sigma2, cmap=cmap,
                                      vmin=vmin, vmax=vmax_, cbar_label=label,
                                      fig=fig, subplot_spec=gs[row])
        ax_s.set_title(title, loc="left", fontsize=plt.rcParams["font.size"])
    save_figure(fig, nn_path / f"RMSE_{out_tag}", formats=("png",))

    line = f"  median R2 = {np.nanmedian(r2):.3f}"
    line += (f" | RMSE full/debiased = {np.nanmedian(rmse):.2f}/"
             f"{np.nanmedian(rmse_debiased):.2f} Sv")
    if show_10y_lpf:
        line += f" | median R2 (10y LPF) = {np.nanmedian(r2_lpf):.3f}"
    print(line)


def main(trained_on: str = TRAINED_ON, single_case: str | None = None,
         covariate_names: str = COVARIATE_NAMES,
         show_10y_lpf: bool = SHOW_10Y_LPF,
         experiment: str | None = None) -> None:
    cases = [resolve_case_spec(single_case)] if single_case else EVALUATION_CASES
    configure_tensorflow_runtime(seed=DEFAULT_TRAINING.random_seed)
    apply_style()
    covariates = prepare_covariate_config(covariate_names)
    experiments = [experiment] if experiment else EXPERIMENTS

    for experiment in experiments:
        nn_path = require_dir(
            results_dir(trained_on, lpf_tag_from_name(experiment))
            / experiment / covariates.input_var,
            f"Trained model {experiment}",
        )
        print("===", experiment)
        # Match evaluation filtering to the training configuration.
        lpf_key = "_LPF_ALL" if lpf_tag_from_name(experiment) else "_ALL"
        # Load fold and member counts from the saved training configuration.
        ensemble = TrainedEnsemble.load(nn_path)
        for cmip_name, out_tag, rlz_tag, n_rlz in cases:
            evaluate_case(nn_path, ensemble, cmip_name, out_tag, covariates,
                          rlz_tag, n_rlz, show_10y_lpf, lpf_key)

        import tensorflow as tf

        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == "__main__":
    args = parse_args()
    main(args.trained_on, args.case, args.covariates, args.show_10y_lpf,
         args.experiment)
