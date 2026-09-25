"""Stage 17: attribute a reconstructed MOC cell to satellite inputs.

Layer-wise relevance propagation (LRP) targets a physical latitude-density
MOC anomaly by composing each fold's output with its inverse PCA and target
scaling. Relevance is propagated through the linear and nonlinear branches.
Input, bias, and numerical-remainder contributions are stored separately.
Run with --help to select the target and propagation rule.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc import lrp_targets
from neurmoc.config import (
    ACTIVE_RUN_ID,
    BASELINE_YEARS,
    MOC_CONVENTION,
    SCIENTIFIC_CONFIG,
    RUN_ROOT,
    results_dir,
)
from neurmoc.inference import TrainedEnsemble
from neurmoc.io_utils import (
    ensure_dir,
    load_mat,
    load_npz_or_mat,
    require_dir,
    require_file,
    save_mat,
)
from neurmoc.lrp import (
    LRP0_DENOMINATOR_ATOL,
    LRP0_RELATIVE_DENOMINATOR_THRESHOLD,
    explain_dbnn_physical_target,
    lrp_method_name,
    physical_output_head,
)
from neurmoc.model import configure_tensorflow_runtime
from neurmoc.naming import (
    lpf_tag_from_name,
    prepare_covariate_config,
    training_config_from_scientific,
)
from neurmoc.results import load_training_moc_baseline
from neurmoc.timeaxis import normalize_month_axis

# =============================================================================
# User settings
# =============================================================================
DEFAULT_TRAINING = training_config_from_scientific(SCIENTIFIC_CONFIG)
TRAINED_ON = DEFAULT_TRAINING.cmip_name
EXPERIMENT = DEFAULT_TRAINING.experiment_name()
COVARIATE_NAMES = DEFAULT_TRAINING.covariate_names

# These must match the Stage-14 reconstruction being explained.
OBP_SOURCE = "GRACE"
SSH_SOURCE = "DUACS"
WIND_SOURCE = "CCMP"

# If density is unspecified, use the reference-state core at the target latitude.
TARGET_LATITUDE = 0.5
TARGET_CORE = "mid"  # "mid" | "abyssal"
TARGET_SIGMA2 = None  # number overrides TARGET_CORE
TARGET_SIGN = 1.0  # -1 explains -Psi instead of Psi

# LRP-0 is default; epsilon LRP is an optional sensitivity test.
LRP_EPSILON = 0.0
LRP0_REFERENCE_EPSILON = 1e-6
LRP0_RULE_REMAINDER_TOLERANCE_SV = 1e-10
CHECK_ONLY = False
VALIDATE_ALL = False
SAVE_MEMBER_RELEVANCE = False  # True also saves each network member's relevance
FORWARD_TOLERANCE_SV = 5e-3  # GPU-TF32 Stage-14 vs NumPy forward pass


def _load_stage14_module():
    """Load Stage 14 as a module so LRP uses exactly the same inputs."""
    path = Path(__file__).with_name("14_reconstruct_real_world.py")
    name = "_neurmoc_stage14_for_lrp"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Stage 14 from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _configure_stage14_sources(stage14, obp: str, ssh: str, wind: str) -> None:
    """Select the same product swap as the reconstruction being explained."""
    stage14.OBP_SOURCE = obp
    stage14.SSH_SOURCE = ssh
    stage14.USE_ERA5_WINDS = wind == "ERA5"
    stage14.PERMUTE = {"obp": False, "ssh": False, "uas": False}


def _validate_input_coordinates(nn_path: Path, assembled) -> None:
    info = load_npz_or_mat(nn_path / "inputs_info", ["mascon_lon", "mascon_lat"])
    train_lon = np.asarray(info["mascon_lon"]).squeeze()
    train_lat = np.asarray(info["mascon_lat"]).squeeze()
    if not (
        assembled.lon.shape == train_lon.shape
        and assembled.lat.shape == train_lat.shape
        and np.allclose(assembled.lon, train_lon, equal_nan=True)
        and np.allclose(assembled.lat, train_lat, equal_nan=True)
    ):
        raise RuntimeError("Observation mascon coordinates/order differ from inputs_info.mat")


def _load_stage14_result(path: Path, time_month: np.ndarray) -> dict:
    result = load_mat(
        require_file(path, "Stage-14 reconstruction to explain"),
        ["pred_yz", "pred_yz_std", "pred_yz_members", "time_month"],
    )
    required = {"pred_yz", "pred_yz_std", "time_month"}
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(f"{path}: missing Stage-14 fields {missing}")
    saved_months = normalize_month_axis(
        result["time_month"],
        np.asarray(result["pred_yz"]).shape[0],
        f"{path} time_month",
    )
    if not np.array_equal(saved_months, time_month):
        raise RuntimeError(
            "LRP observation dates differ from the Stage-14 reconstruction: "
            f"LRP={time_month[0]}..{time_month[-1]} ({time_month.size}), "
            f"Stage14={saved_months[0]}..{saved_months[-1]} ({saved_months.size})"
        )
    return result


def _target_variance(scaler_y, output_index: int) -> float:
    variance = np.asarray(getattr(scaler_y, "var_", []), dtype=float)
    if variance.ndim != 1 or variance.size <= output_index:
        raise TypeError("Target scaler does not expose a compatible var_ array")
    return float(variance[output_index])


def _atomic_save_mat(path: Path, payload: dict) -> None:
    """Write beside the final file, then atomically replace it."""
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        save_mat(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--trained-on", default=TRAINED_ON)
    parser.add_argument("-E", "--experiment", default=EXPERIMENT)
    parser.add_argument("-x", "--covariates", default=COVARIATE_NAMES)
    parser.add_argument(
        "--obp-source",
        default=OBP_SOURCE,
        choices=["GRACE", "GRACE_CSR"],
    )
    parser.add_argument("--ssh-source", default=SSH_SOURCE, choices=["DUACS", "NASASSH"])
    parser.add_argument("--wind-source", default=WIND_SOURCE, choices=["CCMP", "ERA5"])
    parser.add_argument("--target-lat", type=float, default=TARGET_LATITUDE)
    parser.add_argument(
        "--target-core",
        choices=["mid", "abyssal"],
        default=TARGET_CORE,
        help="baseline cell core used when --target-sigma2 is omitted",
    )
    parser.add_argument(
        "--target-sigma2",
        type=float,
        default=TARGET_SIGMA2,
        help="explicit sigma2 level; overrides --target-core",
    )
    parser.add_argument(
        "--target-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=TARGET_SIGN,
        help="use -1 to explain -Psi",
    )
    parser.add_argument("--epsilon", type=float, default=LRP_EPSILON)
    parser.add_argument(
        "--save-member-relevance",
        action=argparse.BooleanOptionalAction,
        default=SAVE_MEMBER_RELEVANCE,
    )
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument(
        "--check-only", dest="validation_mode", action="store_const",
        const="check", help="analyze fold 1/member 1, but write nothing",
    )
    validation.add_argument(
        "--validate-all", dest="validation_mode", action="store_const",
        const="all", help="analyze all members, but write nothing",
    )
    validation.add_argument(
        "--full-product", dest="validation_mode", action="store_const",
        const="product", help="write the complete selected-rule product",
    )
    default_mode = "check" if CHECK_ONLY else ("all" if VALIDATE_ALL else "product")
    parser.set_defaults(validation_mode=default_mode)
    args = parser.parse_args()
    args.check_only = args.validation_mode == "check"
    args.validate_all = args.validation_mode == "all"
    return args


def main(
    *,
    trained_on: str = TRAINED_ON,
    experiment: str = EXPERIMENT,
    covariate_names: str = COVARIATE_NAMES,
    obp_source: str = OBP_SOURCE,
    ssh_source: str = SSH_SOURCE,
    wind_source: str = WIND_SOURCE,
    target_latitude: float = TARGET_LATITUDE,
    target_core: str = TARGET_CORE,
    target_sigma2: float | None = TARGET_SIGMA2,
    target_sign: float = TARGET_SIGN,
    epsilon: float = LRP_EPSILON,
    save_member_relevance: bool = SAVE_MEMBER_RELEVANCE,
    check_only: bool = CHECK_ONLY,
    validate_all: bool = VALIDATE_ALL,
) -> Path | None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon!r}")
    if check_only and validate_all:
        raise ValueError("check_only and validate_all are mutually exclusive")
    propagation_rule = "lrp0" if epsilon == 0.0 else "epsilon"
    method_name = lrp_method_name(epsilon, propagation_rule)
    comparison_epsilon = (
        LRP0_REFERENCE_EPSILON
        if propagation_rule == "lrp0"
        else epsilon * 10.0
    )
    comparison_method = lrp_method_name(comparison_epsilon, "epsilon")
    if target_sign not in (-1.0, 1.0):
        raise ValueError("target_sign must be +1 or -1")

    configure_tensorflow_runtime(seed=DEFAULT_TRAINING.random_seed)
    stage14 = _load_stage14_module()
    _configure_stage14_sources(stage14, obp_source, ssh_source, wind_source)
    covariates = prepare_covariate_config(covariate_names)
    lpf_tag = lpf_tag_from_name(experiment)
    training_nn_path = require_dir(
        RUN_ROOT / trained_on / experiment
        / covariates.input_var,
        f"Trained model from source run {ACTIVE_RUN_ID}",
    )
    analysis_nn_path = (
        results_dir(trained_on, lpf_tag) / experiment / covariates.input_var
    )
    training_config = stage14.validate_training_covariates(
        training_nn_path, tuple(covariates.names)
    )
    lpf_months = int(training_config.get("lpf_months", 0))
    if lpf_months not in (0, 24):
        raise RuntimeError(f"Stage-14 observation files support 0 or 24 months, not {lpf_months}")

    reference = load_npz_or_mat(stage14.REFERENCE_GRID, ["rho2_full", "lat_psi"])
    rho2 = np.asarray(reference["rho2_full"]).squeeze()
    lat = np.asarray(reference["lat_psi"]).squeeze()
    sigma2 = rho2 - 1000.0 if rho2[0] > 1000 else rho2.astype(float)
    baseline = None
    if training_config.get("moc_convention", "absolute") == "anomaly_2004_2009":
        baseline = load_training_moc_baseline(
            training_nn_path, lat, rho2, expected_period=BASELINE_YEARS
        )

    assembled = stage14.assemble_inputs(covariates, lpf=(lpf_months == 24))
    _validate_input_coordinates(training_nn_path, assembled)
    x_raw = np.asarray(assembled.values, dtype=float)
    if not np.isfinite(x_raw).all():
        raise RuntimeError("Aligned real-world input contains NaN or infinite values")

    mask = (
        np.asarray(
            load_npz_or_mat(training_nn_path / "Psi_mask", ["Psi_mask"])[
                "Psi_mask"
            ]
        )
        .squeeze()
        .astype(bool)
    )
    if mask.shape != (rho2.size * lat.size,):
        raise RuntimeError(f"Psi_mask has shape {mask.shape}; expected {(rho2.size * lat.size,)}")
    target = lrp_targets.resolve_target(
        lat, sigma2, mask, baseline, target_latitude, target_core, target_sigma2
    )

    source_suffix = lrp_targets.source_suffix(obp_source, ssh_source, wind_source)
    if source_suffix != stage14.permute_suffix():
        raise RuntimeError(
            "Stage-17 product suffix differs from Stage 14"
        )
    stage14_file = (
        analysis_nn_path / "RealWorld" / f"Pred_RealWorld{source_suffix}.mat"
    )
    stage14_result = _load_stage14_result(stage14_file, assembled.time_month)
    pred_yz = np.asarray(stage14_result["pred_yz"], dtype=float)
    pred_yz_std = np.asarray(stage14_result["pred_yz_std"], dtype=float)
    expected_grid_shape = (x_raw.shape[0], rho2.size, lat.size)
    if pred_yz.shape != expected_grid_shape or pred_yz_std.shape != expected_grid_shape:
        raise RuntimeError(
            f"{stage14_file}: prediction grids have shapes {pred_yz.shape} and "
            f"{pred_yz_std.shape}; expected {expected_grid_shape}"
        )
    stage14_members = stage14_result.get("pred_yz_members")
    if stage14_members is not None:
        stage14_members = np.asarray(stage14_members, dtype=float)

    target_directory = lrp_targets.target_slug(target, target_sign)
    output_dir = (
        analysis_nn_path
        / "RealWorld"
        / "LRP"
        / target_directory
        / lrp_targets.epsilon_slug(epsilon)
    )
    output_file = output_dir / f"Relevance{source_suffix}.mat"

    n_folds = int(training_config["num_folds"])
    n_ensembles = int(training_config["nn_repeats"])
    ensemble = TrainedEnsemble.load(training_nn_path, n_folds, n_ensembles)
    if ensemble.pcas_x is not None:
        raise RuntimeError(
            "LRP is implemented for networks without input PCA; relevance "
            "would first have to be propagated through each input-PCA block."
        )
    if ensemble.pcas_y is None:
        raise RuntimeError("LRP requires the fold-specific target EOF (PCA-Y) basis")

    n_time, n_features = x_raw.shape
    n_covariates = len(covariates.names)
    n_mascons = assembled.lon.size
    if n_features != n_covariates * n_mascons:
        raise RuntimeError(
            f"Input width {n_features} is not {n_covariates} covariates x "
            f"{n_mascons} registered mascons"
        )
    n_members = n_folds * n_ensembles
    if stage14_members is not None:
        expected_member_shape = (n_members, *expected_grid_shape)
        if stage14_members.shape != expected_member_shape:
            raise RuntimeError(
                f"{stage14_file}: pred_yz_members has {stage14_members.shape}; "
                f"expected {expected_member_shape}"
            )

    print("=" * 78)
    print("REAL-WORLD PHYSICAL-TARGET LRP")
    print(f"run/model : {ACTIVE_RUN_ID} / {experiment}")
    print(f"network   : {training_nn_path}")
    print(f"analysis  : {analysis_nn_path}")
    print(
        f"inputs    : {n_time} months x {n_features} features = "
        f"{n_covariates} x {n_mascons}; {assembled.time_month[0]}.."
        f"{assembled.time_month[-1]}"
    )
    print(
        f"target    : {target.mode}, Psi({target.sigma2:.4f}, "
        f"{target.latitude:+.1f} deg), sign={target_sign:+.0f}; "
        f"masked output index={target.valid_output_index}"
    )
    if check_only:
        output_mode = "one-member validation; no output will be written"
    elif validate_all:
        output_mode = "full-ensemble validation; no output will be written"
    else:
        output_mode = f"complete product -> {output_file}"
    if propagation_rule == "lrp0":
        print(
            "method    : branch-wise LRP-0 (exact unstabilized z-rule); "
            "inverse-transform offset and internal biases kept separate"
        )
    else:
        print(
            f"method    : branch-wise epsilon LRP, epsilon={epsilon:g}; "
            "inverse-transform offset and internal biases kept separate"
        )
    print(f"mode      : {output_mode}")
    print(
        f"comparison: {comparison_method}, epsilon={comparison_epsilon:g} "
        "(fold 1/member 1, first 8 months)"
    )
    print(f"ensemble  : {n_folds} folds x {n_ensembles} members")
    print("=" * 78)

    sum_relevance = np.zeros((n_time, n_features), dtype=float)
    sum_relevance_sq = np.zeros_like(sum_relevance)
    sum_abs_relevance = np.zeros_like(sum_relevance)
    target_prediction = np.empty((n_members, n_time), dtype=float)
    target_centered = np.empty_like(target_prediction)
    target_offsets = np.empty(n_members, dtype=float)
    internal_bias = np.empty_like(target_prediction)
    stabilizer_remainder = np.empty_like(target_prediction)
    feature_residual = np.empty_like(target_prediction)
    accounted_residual = np.empty_like(target_prediction)
    branch_scores = np.empty((n_members, n_time, 2), dtype=float)
    covariate_sum = np.empty((n_members, n_time, n_covariates), dtype=float)
    covariate_abs_sum = np.empty_like(covariate_sum)
    member_fold = np.empty(n_members, dtype=np.int16)
    member_ensemble = np.empty(n_members, dtype=np.int16)
    minimum_absolute_denominator = np.empty(n_members, dtype=float)
    minimum_relative_denominator = np.empty(n_members, dtype=float)
    maximum_absolute_message = np.empty(n_members, dtype=float)
    maximum_absolute_relevance = np.empty(n_members, dtype=float)
    inactive_zero_denominator_count = np.empty(n_members, dtype=np.int64)
    low_relative_denominator_count = np.empty(n_members, dtype=np.int64)
    rule_remainder_max_abs = np.empty(n_members, dtype=float)
    member_relevance = (
        np.empty((n_members, n_time, n_features), dtype=np.float32)
        if save_member_relevance
        else None
    )
    forward_differences: list[float] = []
    target_variances = np.empty(n_folds, dtype=float)
    pca_variance_fraction = np.empty(n_folds, dtype=float)
    sign_check_error = np.nan
    batch_check_error = np.nan
    comparison_relative_l2 = np.nan
    first_explanation = None

    member_index = 0
    for fold_index in range(n_folds):
        scaler_x = ensemble.scalers_x[fold_index]
        scaler_y = ensemble.scalers_y[fold_index]
        pca_y = ensemble.pcas_y[fold_index]
        expected_features = int(getattr(scaler_x, "n_features_in_", n_features))
        if expected_features != n_features:
            raise RuntimeError(
                f"Fold {fold_index + 1} scaler expects {expected_features} "
                f"features; observations supply {n_features}"
            )
        if int(getattr(scaler_y, "n_features_in_", mask.sum())) != int(mask.sum()):
            raise RuntimeError(f"Fold {fold_index + 1} target scaler does not match Psi_mask")
        target_variances[fold_index] = _target_variance(scaler_y, target.valid_output_index)
        if target_variances[fold_index] <= 1e-12:
            raise ValueError(
                f"The requested target has zero training variance in fold "
                f"{fold_index + 1}; it is masked as geometrically valid but was "
                "not learnable. Choose another density/latitude cell."
            )
        pca_variance_fraction[fold_index] = float(np.sum(pca_y.explained_variance_ratio_))
        x_fold = ensemble.transform_inputs(fold_index, x_raw)
        head = physical_output_head(scaler_y, pca_y, target.valid_output_index, sign=target_sign)

        for ensemble_index, model in enumerate(ensemble.models[fold_index]):
            member_label = (
                f"fold {fold_index + 1}/{n_folds}, member "
                f"{ensemble_index + 1}/{n_ensembles}"
            )
            try:
                explanation = explain_dbnn_physical_target(
                    model,
                    x_fold,
                    head,
                    epsilon=epsilon,
                    propagation_rule=propagation_rule,
                )
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f"{member_label}: {method_name} failed on the complete "
                    f"{n_time}-month record for target lat={target.latitude:+.1f}, "
                    f"sigma2={target.sigma2:.4f}: {exc}"
                ) from exc
            if explanation.relevance.shape != (n_time, n_features):
                raise RuntimeError(
                    f"{member_label}: "
                    f"relevance shape {explanation.relevance.shape}; expected "
                    f"{(n_time, n_features)}"
                )

            # Check the physical output transform against sklearn.
            physical_via_sklearn = (
                scaler_y.inverse_transform(pca_y.inverse_transform(explanation.model_output))[
                    :, target.valid_output_index
                ]
                * target_sign
            )
            head_error = float(
                np.max(np.abs(explanation.physical_prediction - physical_via_sklearn))
            )
            if head_error > 1e-10:
                raise RuntimeError(
                    f"{member_label}: "
                    f"physical PCA/scaler head mismatch {head_error:.3g} Sv"
                )

            if stage14_members is not None:
                saved_member = (
                    stage14_members[member_index, :, target.level_index, target.latitude_index]
                    * target_sign
                )
                forward_error = float(
                    np.max(np.abs(explanation.physical_prediction - saved_member))
                )
                forward_differences.append(forward_error)
                if forward_error > FORWARD_TOLERANCE_SV:
                    raise RuntimeError(
                        f"{member_label}: "
                        f"NumPy LRP forward pass differs from the Stage-14 "
                        f"member by {forward_error:.4g} Sv (limit "
                        f"{FORWARD_TOLERANCE_SV:g})"
                    )

            if member_index == 0:
                first_explanation = explanation
                # Check batch invariance and target-sign symmetry.
                subset = min(8, n_time)
                try:
                    repeated = explain_dbnn_physical_target(
                        model,
                        x_fold[:subset],
                        head,
                        epsilon=epsilon,
                        propagation_rule=propagation_rule,
                    )
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        f"{member_label}: {method_name} failed during the "
                        f"first-{subset}-month batch-invariance check: {exc}"
                    ) from exc
                batch_check_error = float(
                    np.max(np.abs(repeated.relevance - explanation.relevance[:subset]))
                )
                opposite_head = physical_output_head(
                    scaler_y, pca_y, target.valid_output_index, sign=-target_sign
                )
                try:
                    opposite = explain_dbnn_physical_target(
                        model,
                        x_fold[:subset],
                        opposite_head,
                        epsilon=epsilon,
                        propagation_rule=propagation_rule,
                    )
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        f"{member_label}: {method_name} failed during the "
                        f"first-{subset}-month sign-equivariance check: {exc}"
                    ) from exc
                sign_check_error = float(
                    max(
                        np.max(np.abs(opposite.relevance + repeated.relevance)),
                        np.max(np.abs(opposite.physical_prediction + repeated.physical_prediction)),
                    )
                )
                try:
                    sensitivity = explain_dbnn_physical_target(
                        model,
                        x_fold[:subset],
                        head,
                        epsilon=comparison_epsilon,
                        propagation_rule="epsilon",
                    )
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        f"{member_label}: comparison method "
                        f"{comparison_method} failed during the first-{subset}-month "
                        f"rule-sensitivity check: {exc}"
                    ) from exc
                denominator = np.linalg.norm(repeated.relevance)
                comparison_relative_l2 = (
                    float(np.linalg.norm(sensitivity.relevance - repeated.relevance) / denominator)
                    if denominator
                    else 0.0
                )
                if batch_check_error > 1e-10 or sign_check_error > 1e-10:
                    raise RuntimeError(
                        f"{member_label}: LRP implementation self-check failed: "
                        f"batch={batch_check_error:.3g}, sign={sign_check_error:.3g}"
                    )

            relevance = explanation.relevance
            sum_relevance += relevance
            sum_relevance_sq += relevance**2
            sum_abs_relevance += np.abs(relevance)
            if member_relevance is not None:
                member_relevance[member_index] = relevance.astype(np.float32)

            blocks = relevance.reshape(n_time, n_covariates, n_mascons)
            covariate_sum[member_index] = blocks.sum(axis=2)
            covariate_abs_sum[member_index] = np.abs(blocks).sum(axis=2)
            target_prediction[member_index] = explanation.physical_prediction
            target_centered[member_index] = explanation.centered_score
            target_offsets[member_index] = head.offset
            internal_bias[member_index] = explanation.internal_bias_relevance
            stabilizer_remainder[member_index] = explanation.stabilizer_remainder
            feature_residual[member_index] = explanation.feature_conservation_residual
            accounted_residual[member_index] = explanation.accounted_conservation_residual
            branch_scores[member_index] = explanation.branch_scores
            member_fold[member_index] = fold_index + 1
            member_ensemble[member_index] = ensemble_index + 1
            minimum_absolute_denominator[member_index] = (
                explanation.minimum_absolute_denominator
            )
            minimum_relative_denominator[member_index] = (
                explanation.minimum_relative_denominator
            )
            maximum_absolute_message[member_index] = (
                explanation.maximum_absolute_message
            )
            maximum_absolute_relevance[member_index] = float(
                np.max(np.abs(relevance))
            )
            inactive_zero_denominator_count[member_index] = (
                explanation.inactive_zero_denominator_count
            )
            low_relative_denominator_count[member_index] = (
                explanation.low_relative_denominator_count
            )
            rule_remainder_max_abs[member_index] = float(
                np.max(np.abs(explanation.stabilizer_remainder))
            )

            if propagation_rule == "lrp0":
                safety_values = np.asarray(
                    [
                        minimum_absolute_denominator[member_index],
                        minimum_relative_denominator[member_index],
                        maximum_absolute_message[member_index],
                        maximum_absolute_relevance[member_index],
                        rule_remainder_max_abs[member_index],
                    ],
                    dtype=float,
                )
                if not np.isfinite(safety_values).all():
                    raise FloatingPointError(
                        f"{member_label}: LRP-0 produced a non-finite full-record "
                        f"safety diagnostic: {safety_values.tolist()}"
                    )
                if (
                    rule_remainder_max_abs[member_index]
                    > LRP0_RULE_REMAINDER_TOLERANCE_SV
                ):
                    raise FloatingPointError(
                        f"{member_label}: exact LRP-0 rule remainder is "
                        f"{rule_remainder_max_abs[member_index]:.3e} Sv, exceeding "
                        f"the roundoff tolerance "
                        f"{LRP0_RULE_REMAINDER_TOLERANCE_SV:.1e} Sv"
                    )

            print(
                f"fold {fold_index + 1}/{n_folds}, member "
                f"{ensemble_index + 1}/{n_ensembles}: "
                f"target={np.mean(explanation.physical_prediction):+.3f} Sv; "
                f"feature-only residual RMS="
                f"{np.sqrt(np.mean(explanation.feature_conservation_residual**2)):.3f} Sv; "
                f"accounted residual max="
                f"{np.max(np.abs(explanation.accounted_conservation_residual)):.2e} Sv"
            )

            if check_only:
                print("-" * 78)
                print("CHECK ONLY PASSED - one member analyzed; nothing was written")
                print(f"physical-head parity : {head_error:.3e} Sv")
                if forward_differences:
                    print(f"Stage-14 member parity: {forward_differences[-1]:.3e} Sv")
                print(f"batch invariance     : {batch_check_error:.3e}")
                print(f"sign equivariance    : {sign_check_error:.3e}")
                print(
                    f"comparison to {comparison_method}, "
                    f"epsilon={comparison_epsilon:g}: "
                    f"{comparison_relative_l2:.3e} "
                    "(relative L2, first 8 months)"
                )
                if propagation_rule == "lrp0" and first_explanation is not None:
                    print(
                        "LRP-0 denominator QA: "
                        f"hard |z|>{LRP0_DENOMINATOR_ATOL:.1e}; warn when "
                        f"relative |z|<={LRP0_RELATIVE_DENOMINATOR_THRESHOLD:.1e}"
                    )
                    print(
                        "minimum active |z|  : "
                        f"{first_explanation.minimum_absolute_denominator:.3e}"
                    )
                    print(
                        "minimum relative |z|: "
                        f"{first_explanation.minimum_relative_denominator:.3e}"
                    )
                    print(
                        "maximum |message|   : "
                        f"{first_explanation.maximum_absolute_message:.3e}"
                    )
                    print(
                        "maximum |relevance| : "
                        f"{np.max(np.abs(first_explanation.relevance)):.3e} Sv"
                    )
                    print(
                        "maximum rule remainder: "
                        f"{rule_remainder_max_abs[member_index]:.3e} Sv "
                        f"(limit {LRP0_RULE_REMAINDER_TOLERANCE_SV:.1e})"
                    )
                    print(
                        "inactive exact 0/0  : "
                        f"{first_explanation.inactive_zero_denominator_count} "
                        "(assigned zero message)"
                    )
                    print(
                        "low-relative warnings: "
                        f"{first_explanation.low_relative_denominator_count} "
                        f"(relative |z| <= "
                        f"{LRP0_RELATIVE_DENOMINATOR_THRESHOLD:.1e})"
                    )
                return None

            member_index += 1
            gc.collect()

    if member_index != n_members:
        raise RuntimeError(f"Analyzed {member_index} members; expected {n_members}")

    relevance_mean_flat = sum_relevance / n_members
    relevance_variance = np.maximum(sum_relevance_sq / n_members - relevance_mean_flat**2, 0.0)
    relevance_std_flat = np.sqrt(relevance_variance)
    relevance_abs_mean_flat = sum_abs_relevance / n_members

    stage14_target_mean = pred_yz[:, target.level_index, target.latitude_index] * target_sign
    stage14_target_std = pred_yz_std[:, target.level_index, target.latitude_index]
    mean_error = float(np.max(np.abs(target_prediction.mean(axis=0) - stage14_target_mean)))
    std_error = float(np.max(np.abs(target_prediction.std(axis=0) - stage14_target_std)))
    if max(mean_error, std_error) > FORWARD_TOLERANCE_SV:
        raise RuntimeError(
            "LRP ensemble target does not reproduce Stage 14: "
            f"mean max diff={mean_error:.4g} Sv, std max diff={std_error:.4g} Sv"
        )

    if propagation_rule == "lrp0":
        lrp0_minimum_absolute = float(np.min(minimum_absolute_denominator))
        lrp0_minimum_relative = float(np.min(minimum_relative_denominator))
        lrp0_maximum_message = float(np.max(maximum_absolute_message))
        lrp0_maximum_relevance = float(np.max(maximum_absolute_relevance))
        lrp0_inactive_zero_count = int(np.sum(inactive_zero_denominator_count))
        lrp0_low_relative_count = int(np.sum(low_relative_denominator_count))
        lrp0_rule_remainder_maximum = float(np.max(rule_remainder_max_abs))

    if validate_all:
        print("=" * 78)
        print(
            f"FULL-ENSEMBLE {method_name} VALIDATION PASSED - "
            "nothing was written"
        )
        print(f"Stage-14 parity: mean={mean_error:.3e} Sv, std={std_error:.3e} Sv")
        print(f"max accounted conservation residual: {np.max(np.abs(accounted_residual)):.3e} Sv")
        print(
            f"first-8 comparison to {comparison_method}, "
            f"epsilon={comparison_epsilon:g}: relative L2="
            f"{comparison_relative_l2:.3e}"
        )
        if propagation_rule == "lrp0":
            print(
                "LRP-0 full-record diagnostics: "
                f"min |z|={lrp0_minimum_absolute:.3e}, "
                f"min relative |z|={lrp0_minimum_relative:.3e}, "
                f"max |message|={lrp0_maximum_message:.3e}, "
                f"max |relevance|={lrp0_maximum_relevance:.3e} Sv"
            )
            print(
                "LRP-0 zero/remainder diagnostics: "
                f"inactive 0/0={lrp0_inactive_zero_count}, "
                f"low-relative warnings={lrp0_low_relative_count}, "
                f"max rule remainder={lrp0_rule_remainder_maximum:.3e} Sv "
                f"(limit {LRP0_RULE_REMAINDER_TOLERANCE_SV:.1e})"
            )
        print("=" * 78)
        return None

    ensure_dir(output_dir)

    block_shape = (n_time, n_covariates, n_mascons)
    payload = {
        "relevance_mean": relevance_mean_flat.reshape(block_shape).astype(np.float32),
        "relevance_std": relevance_std_flat.reshape(block_shape).astype(np.float32),
        "relevance_abs_mean": relevance_abs_mean_flat.reshape(block_shape).astype(np.float32),
        "relevance_covariate_sum_members": covariate_sum.astype(np.float32),
        "relevance_covariate_abs_sum_members": covariate_abs_sum.astype(np.float32),
        "target_prediction_members": target_prediction.astype(np.float32),
        "target_centered_members": target_centered.astype(np.float32),
        "target_output_offset_members": target_offsets.astype(np.float32),
        "internal_bias_relevance_members": internal_bias.astype(np.float32),
        "stabilizer_remainder_members": stabilizer_remainder.astype(np.float32),
        "feature_conservation_residual_members": feature_residual.astype(np.float32),
        "accounted_conservation_residual_members": accounted_residual.astype(np.float32),
        "branch_centered_scores_members": branch_scores.astype(np.float32),
        "member_fold": member_fold,
        "member_ensemble": member_ensemble,
        "stage14_target_mean": stage14_target_mean.astype(np.float32),
        "stage14_target_std": stage14_target_std.astype(np.float32),
        "time_month": np.datetime_as_string(assembled.time_month, unit="M").astype("U7"),
        "relevance_dimensions": np.asarray(["time", "covariate", "mascon"], dtype="U"),
        "covariate_names": np.asarray(assembled.covariate_names, dtype="U"),
        "input_source_names": np.asarray(assembled.source_names, dtype="U"),
        "input_source_files": np.asarray(assembled.source_files, dtype="U"),
        "input_baseline_specs": np.asarray(assembled.baseline_specs, dtype="U"),
        "mascon_lon": np.asarray(assembled.lon, dtype=float),
        "mascon_lat": np.asarray(assembled.lat, dtype=float),
        "feature_block_offsets": np.arange(n_covariates + 1) * n_mascons,
        "moc_latitude_grid": lat,
        "moc_sigma2_grid": sigma2,
        "psi_mask": mask,
        "target_mode": target.mode,
        "target_sign": np.float64(target_sign),
        "target_latitude": np.float64(target.latitude),
        "target_sigma2": np.float64(target.sigma2),
        "target_latitude_index_python": np.int64(target.latitude_index),
        "target_level_index_python": np.int64(target.level_index),
        "target_flat_grid_index_python": np.int64(target.flat_grid_index),
        "target_valid_output_index_python": np.int64(target.valid_output_index),
        "target_baseline_psi_2004_2009": np.float64(target.baseline_value),
        "target_signed_baseline_2004_2009": np.float64(target.baseline_value * target_sign),
        "moc_convention": MOC_CONVENTION,
        "lrp_method": method_name,
        "lrp_epsilon": np.float64(epsilon),
        "lrp_propagation_rule": propagation_rule,
        "activation_relevance_rule": "identity pass-through (including swish)",
        "bias_policy": "saved separately; not assigned to mascon inputs",
        "output_offset_policy": "inverse-PCA/scaler offset saved separately",
        "input_relevance_domain": (
            "fold-standardized model inputs; feature identities remain the registered mascons"
        ),
        "accounting_identity": (
            "prediction = sum(input relevance) + internal bias + stabilizer + output offset"
        ),
        "target_training_variance_by_fold": target_variances,
        "pca_explained_variance_fraction_by_fold": pca_variance_fraction,
        "stage14_mean_max_abs_difference_sv": np.float64(mean_error),
        "stage14_std_max_abs_difference_sv": np.float64(std_error),
        "batch_invariance_max_abs_difference": np.float64(batch_check_error),
        "sign_equivariance_max_abs_difference": np.float64(sign_check_error),
        "lrp_comparison_method_first8": comparison_method,
        "lrp_comparison_epsilon_first8": np.float64(comparison_epsilon),
        "lrp_comparison_relative_l2_first8": np.float64(
            comparison_relative_l2
        ),
        "run_id": ACTIVE_RUN_ID,
        "training_source_run_id": ACTIVE_RUN_ID,
        "trained_on": trained_on,
        "training_experiment": experiment,
    }
    if propagation_rule == "epsilon":
        # Comparison between epsilon and 10 x epsilon on the first 8 samples.
        payload["epsilon_x10_relative_l2_first8"] = np.float64(
            comparison_relative_l2
        )
    else:
        payload.update(
            {
                "lrp0_denominator_atol": np.float64(LRP0_DENOMINATOR_ATOL),
                "lrp0_relative_denominator_threshold": np.float64(
                    LRP0_RELATIVE_DENOMINATOR_THRESHOLD
                ),
                "lrp0_rule_remainder_tolerance_sv": np.float64(
                    LRP0_RULE_REMAINDER_TOLERANCE_SV
                ),
                "lrp0_minimum_absolute_denominator_by_member": (
                    minimum_absolute_denominator
                ),
                "lrp0_minimum_relative_denominator_by_member": (
                    minimum_relative_denominator
                ),
                "lrp0_maximum_absolute_message_by_member": (
                    maximum_absolute_message
                ),
                "lrp0_maximum_absolute_relevance_by_member": (
                    maximum_absolute_relevance
                ),
                "lrp0_inactive_zero_denominator_count_by_member": (
                    inactive_zero_denominator_count
                ),
                "lrp0_low_relative_denominator_count_by_member": (
                    low_relative_denominator_count
                ),
                "lrp0_rule_remainder_max_abs_sv_by_member": (
                    rule_remainder_max_abs
                ),
            }
        )
    if member_relevance is not None:
        payload["relevance_members"] = member_relevance
        payload["relevance_members_dimensions"] = np.asarray(
            ["member", "time", "feature"], dtype="U"
        )

    _atomic_save_mat(output_file, payload)

    print("=" * 78)
    print("LRP COMPLETE")
    print(f"saved     : {output_file}")
    print(f"Stage-14 parity: mean={mean_error:.3e} Sv, std={std_error:.3e} Sv")
    print(
        "Accounting: max |features + internal bias + stabilizer - centered "
        f"target| = {np.max(np.abs(accounted_residual)):.3e} Sv"
    )
    print(
        "Interpret relevance_mean as signed ensemble-mean contribution and "
        "relevance_abs_mean as magnitude; they are not interchangeable."
    )
    print("=" * 78)
    return output_file


if __name__ == "__main__":
    args = parse_args()
    main(
        trained_on=args.trained_on,
        experiment=args.experiment,
        covariate_names=args.covariates,
        obp_source=args.obp_source,
        ssh_source=args.ssh_source,
        wind_source=args.wind_source,
        target_latitude=args.target_lat,
        target_core=args.target_core,
        target_sigma2=args.target_sigma2,
        target_sign=args.target_sign,
        epsilon=args.epsilon,
        save_member_relevance=args.save_member_relevance,
        check_only=args.check_only,
        validate_all=args.validate_all,
    )
