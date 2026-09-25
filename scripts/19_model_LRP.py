"""Stage 19: attribute a cross-model MOC prediction to model inputs.

Apply physical-target LRP to a completed Stage-10 evaluation case, where
the true MOC is available. The target cell is defined from the training
model's reference state rather than the evaluation model. Input, bias, and
numerical-remainder contributions are stored separately.
Run with --help to select a case and target.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc import lrp_targets
from neurmoc.cmip_io import EvaluationData, load_evaluation_data
from neurmoc.config import (
    ACTIVE_RUN_ID,
    RUN_ROOT,
    BASELINE_YEARS,
    CMIP_ROOT,
    MOC_CONVENTION,
    SCIENTIFIC_CONFIG,
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
    lrp_method_name,
    physical_output_head,
    prepare_dbnn_physical_target_explainer,
)
from neurmoc.moc_utils import locate_cell_cores
from neurmoc.model import configure_tensorflow_runtime
from neurmoc.model_test_cases import (
    stage06_lpf_key,
    training_dataset_root,
    contiguous_realization_groups,
    resolve_case_spec,
)
from neurmoc.naming import prepare_covariate_config, training_config_from_scientific
from neurmoc.results import load_training_moc_baseline

DEFAULT_TRAINING = training_config_from_scientific(SCIENTIFIC_CONFIG)

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
TRAINED_ON = DEFAULT_TRAINING.cmip_name
EXPERIMENT = DEFAULT_TRAINING.experiment_name()
COVARIATE_NAMES = DEFAULT_TRAINING.covariate_names

# Custom cases require NAME:OUT_TAG:RLZ_TAG:N_REALIZATIONS.
CASE = "MRI_SSP245"

TARGET_LATITUDE = 26.5
TARGET_CORE = "mid"          # "mid" | "abyssal"; ignored with sigma2
TARGET_SIGMA2 = None
TARGET_SIGN = 1.0            # +1 explains Psi, -1 explains -Psi
# LRP-0 is default; epsilon LRP is an optional sensitivity test.
LRP_EPSILON = 0.0
LRP0_REFERENCE_EPSILON = 1e-6
LRP_BATCH_SIZE = 256
LRP_CHECK_SAMPLES = 64  # evenly spaced across all realizations and months
CHECK_ONLY = False
VALIDATE_ALL = False

# Stream full relevance arrays to chunked HDF5 when requested.
SAVE_MEMBER_RELEVANCE = False

FORWARD_TOLERANCE_SV = 5e-3
HEAD_TOLERANCE_SV = 1e-10
SELF_CHECK_TOLERANCE = 1e-10
ACCOUNTING_TOLERANCE_SV = 1e-8


def _target_variance(scaler_y, output_index: int) -> float:
    variance = np.asarray(getattr(scaler_y, "var_", []), dtype=float)
    if variance.ndim != 1 or variance.size <= output_index:
        raise TypeError("Target scaler does not expose a compatible var_ array")
    value = float(variance[output_index])
    if not np.isfinite(value):
        raise ValueError("Target training variance is not finite")
    return value


def _atomic_save_mat(path: Path, payload: dict) -> None:
    """Write beside the final file, then atomically replace it."""
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        save_mat(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_seed_map_json(
    nn_path: Path,
    *,
    training_seed: int,
    n_folds: int,
    n_ensembles: int,
) -> str:
    """Load and validate the exact PCA/fold/member seed schedule."""
    path = require_file(nn_path / "seed_map.json", "Training seed map")
    raw = path.read_text(encoding="utf-8")
    try:
        seed_map = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(seed_map, dict):
        raise RuntimeError(f"{path} does not contain a JSON object")
    if seed_map.get("base_random_seed") != training_seed:
        raise RuntimeError(
            f"{path}: base_random_seed={seed_map.get('base_random_seed')!r}, "
            f"but training_config.json records {training_seed}"
        )
    folds = seed_map.get("folds")
    if not isinstance(folds, list) or len(folds) != n_folds:
        raise RuntimeError(
            f"{path}: expected {n_folds} fold seed records, got "
            f"{len(folds) if isinstance(folds, list) else 'a non-list'}"
        )
    for expected_fold, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict) or fold.get("fold") != expected_fold:
            raise RuntimeError(
                f"{path}: fold seed records are not ordered 1..{n_folds}"
            )
        pca_y = fold.get("pca_y")
        if not isinstance(pca_y, dict) or not isinstance(pca_y.get("seed"), int):
            raise RuntimeError(
                f"{path}: fold {expected_fold} has no integer PCA-Y seed"
            )
        members = fold.get("members")
        if not isinstance(members, list) or len(members) != n_ensembles:
            raise RuntimeError(
                f"{path}: fold {expected_fold} must record {n_ensembles} members"
            )
        for expected_member, member in enumerate(members, start=1):
            if (
                not isinstance(member, dict)
                or member.get("ensemble") != expected_member
                or not isinstance(member.get("seed"), int)
            ):
                raise RuntimeError(
                    f"{path}: fold {expected_fold} member seeds are not an "
                    f"ordered 1..{n_ensembles} map"
                )
    return raw


def _validate_training_input_schema(
    nn_path: Path,
    training_config: dict,
    covariate_names: list[str],
    case: EvaluationData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recorded_covariates = [
        item.strip()
        for item in str(training_config.get("covariate_names", "")).split(",")
        if item.strip()
    ]
    if recorded_covariates != list(covariate_names):
        raise RuntimeError(
            "Selected covariates do not match training_config.json: "
            f"selected={list(covariate_names)}, recorded={recorded_covariates}"
        )

    info = load_npz_or_mat(
        nn_path / "inputs_info",
        ["mascon_lon", "mascon_lat", "lat_psi", "InputNumIndCum"],
    )
    train_lon = np.asarray(info["mascon_lon"]).squeeze()
    train_lat_mascon = np.asarray(info["mascon_lat"]).squeeze()
    train_moc_lat = np.asarray(info["lat_psi"]).squeeze()
    train_offsets = np.asarray(info["InputNumIndCum"]).squeeze().astype(np.int64)
    if case.mascon_lon is None or case.mascon_lat is None:
        raise RuntimeError("Evaluation inputs do not expose mascon coordinates")
    if not (
        np.array_equal(np.asarray(case.mascon_lon).squeeze(), train_lon)
        and np.array_equal(np.asarray(case.mascon_lat).squeeze(), train_lat_mascon)
    ):
        raise RuntimeError(
            "Evaluation mascon coordinates/order differ from trained inputs_info"
        )
    offsets = np.r_[0, np.cumsum(np.asarray(case.block_sizes, dtype=np.int64))]
    if not np.array_equal(offsets, train_offsets):
        raise RuntimeError(
            f"Evaluation feature offsets {offsets.tolist()} differ from trained "
            f"offsets {train_offsets.tolist()}"
        )
    if not np.array_equal(np.asarray(case.lat).squeeze(), train_moc_lat):
        raise RuntimeError("Evaluation MOC latitude grid differs from inputs_info")
    return train_lon, train_lat_mascon, offsets


def _month_strings(values: np.ndarray | None, n_time: int) -> np.ndarray:
    if values is None:
        raise RuntimeError("Evaluation data are missing time_month metadata")
    months = np.asarray(values).reshape(-1)
    if months.size != n_time:
        raise RuntimeError(
            f"time_month has {months.size} rows; expected {n_time}"
        )
    try:
        return np.datetime_as_string(months.astype("datetime64[M]"), unit="M")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Evaluation time_month metadata are invalid") from exc


def _load_stage10_target(
    pred_path: Path,
    target,
    expected_shape: tuple[int, int, int],
    target_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred_grid = np.asarray(load_mat(pred_path, ["y_pred"])["y_pred"])
    if pred_grid.shape != expected_shape:
        raise RuntimeError(
            f"{pred_path}: y_pred shape {pred_grid.shape}; expected {expected_shape}"
        )
    pred = np.asarray(
        pred_grid[:, target.level_index, target.latitude_index], dtype=float
    ).copy() * target_sign
    del pred_grid

    truth_grid = np.asarray(load_mat(pred_path, ["y"])["y"])
    if truth_grid.shape != expected_shape:
        raise RuntimeError(
            f"{pred_path}: y shape {truth_grid.shape}; expected {expected_shape}"
        )
    truth = np.asarray(
        truth_grid[:, target.level_index, target.latitude_index], dtype=float
    ).copy() * target_sign
    del truth_grid
    return pred, truth


def _physical_from_scores(scaler_y, pca_y, scores, output_index, sign):
    return (
        scaler_y.inverse_transform(pca_y.inverse_transform(scores))[
            :, output_index
        ]
        * sign
    )


def _load_check_member(nn_path: Path, training_config: dict):
    """Load only fold 1/member 1 for the lightweight validation path."""
    recorded = training_config.get("software_versions", {}).get("scikit-learn")
    if recorded and recorded != "not installed" and version("scikit-learn") != recorded:
        raise RuntimeError(
            f"The trained networks require scikit-learn {recorded}; this "
            f"environment provides {version('scikit-learn')}"
        )
    if any(nn_path.glob("pca_x_fold1_block*.pkl")):
        raise RuntimeError("Stage-19 LRP does not support PCA-X models")
    if not (nn_path / "pca_y_fold1.pkl").is_file():
        raise RuntimeError("Stage-19 requires the fold-specific target EOF (PCA-Y) basis")

    import joblib
    from tensorflow.keras.models import load_model

    scaler_x = joblib.load(require_file(nn_path / "scaler_x_fold1.pkl"))
    scaler_y = joblib.load(require_file(nn_path / "scaler_y_fold1.pkl"))
    pca_y = joblib.load(require_file(nn_path / "pca_y_fold1.pkl"))
    model = load_model(
        require_file(nn_path / "model_fold1_ens1.h5"), compile=False
    )
    return scaler_x, scaler_y, pca_y, model


def _self_checks(
    model,
    x,
    scaler_y,
    pca_y,
    target,
    target_sign,
    epsilon,
    propagation_rule,
    *,
    explainer=None,
):
    head = physical_output_head(
        scaler_y, pca_y, target.valid_output_index, sign=target_sign
    )
    if explainer is None:
        explainer = prepare_dbnn_physical_target_explainer(
            model,
            head,
            epsilon=epsilon,
            propagation_rule=propagation_rule,
        )
    explanation = explainer.explain(x)
    via_sklearn = _physical_from_scores(
        scaler_y, pca_y, explanation.model_output,
        target.valid_output_index, target_sign,
    )
    head_error = float(np.max(np.abs(
        explanation.physical_prediction - via_sklearn
    )))
    keras_scores = np.asarray(model.predict(x, verbose=0), dtype=float)
    keras_physical = _physical_from_scores(
        scaler_y, pca_y, keras_scores, target.valid_output_index, target_sign
    )
    keras_error = float(np.max(np.abs(
        explanation.physical_prediction - keras_physical
    )))

    subset = min(8, x.shape[0])
    repeated = explainer.explain(x[:subset])
    batch_error = float(np.max(np.abs(
        repeated.relevance - explanation.relevance[:subset]
    )))
    opposite_head = physical_output_head(
        scaler_y, pca_y, target.valid_output_index, sign=-target_sign
    )
    opposite = prepare_dbnn_physical_target_explainer(
        model,
        opposite_head,
        epsilon=epsilon,
        propagation_rule=propagation_rule,
    ).explain(x[:subset])
    sign_error = float(max(
        np.max(np.abs(opposite.relevance + repeated.relevance)),
        np.max(np.abs(
            opposite.physical_prediction + repeated.physical_prediction
        )),
    ))
    comparison_epsilon = (
        LRP0_REFERENCE_EPSILON
        if propagation_rule == "lrp0"
        else epsilon * 10.0
    )
    sensitivity = prepare_dbnn_physical_target_explainer(
        model,
        head,
        epsilon=comparison_epsilon,
        propagation_rule="epsilon",
    ).explain(x[:subset])
    denominator = np.linalg.norm(repeated.relevance)
    comparison_relative_l2 = (
        float(np.linalg.norm(
            sensitivity.relevance - repeated.relevance
        ) / denominator)
        if denominator else 0.0
    )
    accounting_error = float(np.max(np.abs(
        explanation.accounted_conservation_residual
    )))
    rule_remainder_error = float(np.max(np.abs(
        explanation.stabilizer_remainder
    )))
    if head_error > HEAD_TOLERANCE_SV:
        raise RuntimeError(f"physical-head parity failed: {head_error:.3g} Sv")
    if keras_error > FORWARD_TOLERANCE_SV:
        raise RuntimeError(
            f"NumPy LRP trace differs from Keras by {keras_error:.3g} Sv"
        )
    if max(batch_error, sign_error) > SELF_CHECK_TOLERANCE:
        raise RuntimeError(
            f"LRP self-check failed: batch={batch_error:.3g}, "
            f"sign={sign_error:.3g}"
        )
    if accounting_error > ACCOUNTING_TOLERANCE_SV:
        raise RuntimeError(
            f"LRP accounting residual is {accounting_error:.3g} Sv"
        )
    if (
        propagation_rule == "lrp0"
        and rule_remainder_error > ACCOUNTING_TOLERANCE_SV
    ):
        raise RuntimeError(
            "Exact LRP-0 produced a non-roundoff rule remainder of "
            f"{rule_remainder_error:.3g} Sv"
        )
    return {
        "head": head_error,
        "keras": keras_error,
        "batch": batch_error,
        "sign": sign_error,
        "comparison_method": lrp_method_name(
            comparison_epsilon, "epsilon"
        ),
        "comparison_relative_l2": comparison_relative_l2,
        "comparison_epsilon": comparison_epsilon,
        "accounting": accounting_error,
        "rule_remainder": rule_remainder_error,
        "minimum_absolute_denominator": (
            explanation.minimum_absolute_denominator
        ),
        "minimum_relative_denominator": (
            explanation.minimum_relative_denominator
        ),
        "maximum_absolute_message": explanation.maximum_absolute_message,
        "maximum_absolute_relevance": float(
            np.max(np.abs(explanation.relevance))
        ),
        "inactive_zero_denominator_count": (
            explanation.inactive_zero_denominator_count
        ),
        "low_relative_denominator_count": (
            explanation.low_relative_denominator_count
        ),
    }


def _skill(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ok = np.isfinite(prediction) & np.isfinite(truth)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan
    error = prediction[ok] - truth[ok]
    rmse = float(np.sqrt(np.mean(error**2)))
    bias = float(np.mean(error))
    corr = np.nan
    if ok.sum() >= 2 and np.std(prediction[ok]) > 0 and np.std(truth[ok]) > 0:
        corr = float(np.corrcoef(prediction[ok], truth[ok])[0, 1])
    return corr, rmse, bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-t", "--trained-on", default=TRAINED_ON)
    parser.add_argument("-E", "--experiment", default=EXPERIMENT)
    parser.add_argument("-x", "--covariates", default=COVARIATE_NAMES)
    parser.add_argument(
        "--case", default=CASE,
        help="registered name or NAME:OUT_TAG:RLZ_TAG:N_REALIZATIONS",
    )
    parser.add_argument("--target-lat", type=float, default=TARGET_LATITUDE)
    parser.add_argument(
        "--target-core", default=TARGET_CORE, choices=["mid", "abyssal"]
    )
    parser.add_argument("--target-sigma2", type=float, default=TARGET_SIGMA2)
    parser.add_argument(
        "--target-sign", type=float, default=TARGET_SIGN,
        choices=[-1.0, 1.0],
    )
    parser.add_argument("--epsilon", type=float, default=LRP_EPSILON)
    parser.add_argument("--batch-size", type=int, default=LRP_BATCH_SIZE)
    parser.add_argument(
        "--save-member-relevance",
        action=argparse.BooleanOptionalAction,
        default=SAVE_MEMBER_RELEVANCE,
        help="stream the full member cube to a separate chunked HDF5 file",
    )
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument(
        "--check-only", dest="validation_mode", action="store_const",
        const="check", help="validate fold 1/member 1; write nothing",
    )
    validation.add_argument(
        "--validate-all", dest="validation_mode", action="store_const",
        const="all", help="validate the complete ensemble; write nothing",
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
    case_spec: str = CASE,
    target_latitude: float = TARGET_LATITUDE,
    target_core: str = TARGET_CORE,
    target_sigma2: float | None = TARGET_SIGMA2,
    target_sign: float = TARGET_SIGN,
    epsilon: float = LRP_EPSILON,
    batch_size: int = LRP_BATCH_SIZE,
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
    if target_sign not in (-1.0, 1.0):
        raise ValueError("target_sign must be +1 or -1")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    batch_size = int(batch_size)

    cmip_name, out_tag, realization_tag, n_realizations = resolve_case_spec(
        case_spec
    )
    covariates = prepare_covariate_config(covariate_names.replace("+", ","))

    model_root = training_dataset_root(trained_on)
    nn_path = require_dir(
        model_root / experiment / covariates.input_var,
        f"Trained model under source run {ACTIVE_RUN_ID}",
    )
    config_path = require_file(nn_path / "training_config.json")
    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise RuntimeError(f"{config_path} does not contain a JSON object")
    training_seed = int(training_config.get("random_seed", 0))
    n_folds = int(training_config["num_folds"])
    n_ensembles = int(training_config["nn_repeats"])
    if n_folds <= 0 or n_ensembles <= 0:
        raise RuntimeError(
            "training_config.json fold/member counts must be positive"
        )
    seed_map_json = _validated_seed_map_json(
        nn_path,
        training_seed=training_seed,
        n_folds=n_folds,
        n_ensembles=n_ensembles,
    )
    tensorflow_runtime = configure_tensorflow_runtime(seed=training_seed)
    software_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        "tensorflow": str(tensorflow_runtime["tensorflow_version"]),
    }
    lpf_key = stage06_lpf_key(int(training_config.get("lpf_months", 0)))

    data_dir = require_dir(CMIP_ROOT / cmip_name, cmip_name)
    case = load_evaluation_data(
        data_dir,
        covariates.names,
        realization_tag,
        lpf_key,
        expected_wind_convention=training_config.get("wind_convention"),
        expected_moc_convention=training_config.get("moc_convention"),
    )
    x_raw = np.asarray(case.x, dtype=float)
    if x_raw.ndim != 2 or not np.isfinite(x_raw).all():
        raise RuntimeError("Evaluation inputs must be a finite [time,feature] array")
    n_time, n_features = x_raw.shape
    time_month = _month_strings(case.time_month, n_time)
    if case.realization_index is None:
        raise RuntimeError("Evaluation data are missing realization_index metadata")
    realization_index = np.asarray(case.realization_index).reshape(-1)
    if realization_index.size != n_time:
        raise RuntimeError("realization_index length differs from evaluation rows")
    realization_groups = contiguous_realization_groups(
        realization_index, expected_count=n_realizations
    )
    realization_labels = np.asarray([
        realization_index[group[0]] for group in realization_groups
    ])
    realization_counts = np.asarray(
        [group.size for group in realization_groups], dtype=np.int64
    )

    train_lon, train_lat_mascon, feature_offsets = _validate_training_input_schema(
        nn_path, training_config, covariates.names, case
    )
    rho2 = np.asarray(case.rho2).squeeze()
    lat = np.asarray(case.lat).squeeze()
    sigma2 = rho2 - 1000.0 if rho2[0] > 1000 else rho2.astype(float)
    training_baseline = None
    if training_config.get("moc_convention", MOC_CONVENTION) == "anomaly_2004_2009":
        training_baseline = load_training_moc_baseline(
            nn_path, lat, rho2, expected_period=BASELINE_YEARS
        )

    train_mask = (
        np.asarray(load_npz_or_mat(nn_path / "Psi_mask", ["Psi_mask"])["Psi_mask"])
        .squeeze().astype(bool)
    )
    eval_mask = np.asarray(case.psi_mask).squeeze().astype(bool)
    expected_plane = rho2.size * lat.size
    if train_mask.shape != (expected_plane,) or eval_mask.shape != (expected_plane,):
        raise RuntimeError(
            f"Training/evaluation masks must both span {expected_plane} cells; "
            f"got {train_mask.shape} and {eval_mask.shape}"
        )
    target = lrp_targets.resolve_target(
        lat, sigma2, train_mask, training_baseline,
        target_latitude, target_core, target_sigma2,
    )
    truth_available = bool(eval_mask[target.flat_grid_index])
    evaluation_output_index = -1
    if truth_available:
        evaluation_output_index = int(
            np.cumsum(eval_mask)[target.flat_grid_index] - 1
        )

    n_members = n_folds * n_ensembles
    pred_path = require_file(nn_path / f"Pred_{out_tag}.mat", "Stage-10 prediction")
    expected_grid_shape = (n_time, rho2.size, lat.size)
    stage10_target_mean, stage10_target_truth = _load_stage10_target(
        pred_path, target, expected_grid_shape, target_sign
    )
    if truth_available:
        loaded_truth = np.asarray(
            case.y[:, evaluation_output_index], dtype=float
        ) * target_sign
        truth_difference = float(np.max(np.abs(
            loaded_truth - stage10_target_truth
        )))
        if truth_difference > HEAD_TOLERANCE_SV:
            raise RuntimeError(
                f"Stage-19 truth differs from Stage 10 by {truth_difference:.3g} Sv"
            )
    else:
        truth_difference = np.nan
        if np.isfinite(stage10_target_truth).any():
            raise RuntimeError(
                "Evaluation mask marks the target unavailable but Stage-10 truth is finite"
            )

    n_covariates = len(covariates.names)
    n_mascons = train_lon.size
    if not np.all(np.asarray(case.block_sizes) == n_mascons):
        raise RuntimeError(
            "Stage-19 maps require every input variable to use the same "
            f"registered mascons; block sizes={case.block_sizes.tolist()}"
        )
    if n_features != n_covariates * n_mascons:
        raise RuntimeError(
            f"Input width {n_features} is not {n_covariates} x {n_mascons}"
        )

    analysis_nn_path = (
        RUN_ROOT / trained_on / experiment / covariates.input_var
    )
    output_dir = (
        analysis_nn_path / "LRP_model" / out_tag
        / lrp_targets.target_slug(target, target_sign)
        / lrp_targets.epsilon_slug(epsilon)
    )
    output_file = output_dir / f"lrp_model_{out_tag}.mat"
    member_file = output_dir / f"lrp_model_{out_tag}_members.h5"
    member_tmp = output_dir / f".lrp_model_{out_tag}_members.tmp.h5"
    if not (check_only or validate_all):
        ensure_dir(output_dir)

    print("=" * 78)
    print("MODEL-TEST PHYSICAL-TARGET LRP")
    print(f"analysis/training source: {ACTIVE_RUN_ID} / {ACTIVE_RUN_ID}")
    print(f"network   : {nn_path}")
    print(
        f"case      : {cmip_name} ({out_tag}, {realization_tag}); "
        f"{n_realizations} realizations, lengths {realization_counts.tolist()}"
    )
    print(
        f"inputs    : {n_time} samples x {n_features} features = "
        f"{n_covariates} x {n_mascons}; batch size {batch_size}"
    )
    print(
        f"target    : training-defined {target.mode}, Psi({target.sigma2:.4f}, "
        f"{target.latitude:+.1f} deg), sign={target_sign:+.0f}; "
        f"truth={'yes' if truth_available else 'NO'}"
    )
    if propagation_rule == "lrp0":
        print(
            "method    : branch-wise LRP-0 (exact z-rule); no denominator "
            "stabilizer"
        )
        print(
            "safety    : hard fail for relevance-active "
            f"|z|<={LRP0_DENOMINATOR_ATOL:.1e}; record warnings for relative "
            f"|z|<={LRP0_RELATIVE_DENOMINATOR_THRESHOLD:.1e}"
        )
    else:
        print(
            f"method    : branch-wise epsilon LRP, epsilon={epsilon:g}; "
            "bias/stabilizer/offset kept separate"
        )
    if check_only:
        print("mode      : check-only (fold 1/member 1; nothing written)")
    elif validate_all:
        print("mode      : full-ensemble validation (nothing written)")
    else:
        print(f"mode      : full product -> {output_file}")
    print(f"ensemble  : {n_folds} folds x {n_ensembles} members")
    print("=" * 78)

    if check_only:
        scaler_x, scaler_y, pca_y, model = _load_check_member(
            nn_path, training_config
        )
        if int(getattr(scaler_x, "n_features_in_", n_features)) != n_features:
            raise RuntimeError("Fold-1 input scaler width differs from evaluation data")
        variance = _target_variance(scaler_y, target.valid_output_index)
        if variance <= 1e-12:
            raise ValueError("Requested target has zero fold-1 training variance")
        n_check = min(LRP_CHECK_SAMPLES, n_time)
        check_indices = np.unique(
            np.rint(np.linspace(0, n_time - 1, n_check)).astype(np.int64)
        )
        x_check = scaler_x.transform(x_raw[check_indices])
        head = physical_output_head(
            scaler_y, pca_y, target.valid_output_index, sign=target_sign
        )
        explainer = prepare_dbnn_physical_target_explainer(
            model,
            head,
            epsilon=epsilon,
            propagation_rule=propagation_rule,
        )
        try:
            checks = _self_checks(
                model, x_check, scaler_y, pca_y, target,
                target_sign, epsilon, propagation_rule, explainer=explainer,
            )
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Stage-19 check-only LRP failed for fold 1/member 1; "
                f"global rows={check_indices.tolist()}, months="
                f"{time_month[check_indices].tolist()}: {exc}"
            ) from exc
        print(
            "CHECK ONLY PASSED - fold 1/member 1; "
            f"{check_indices.size} samples spanning "
            f"{np.unique(realization_index[check_indices]).size} realization(s); "
            "nothing written"
        )
        print(f"physical-head parity : {checks['head']:.3e} Sv")
        print(f"Keras-trace parity   : {checks['keras']:.3e} Sv")
        print(f"accounting residual  : {checks['accounting']:.3e} Sv")
        print(f"rule remainder max   : {checks['rule_remainder']:.3e} Sv")
        print(f"batch invariance     : {checks['batch']:.3e}")
        print(f"sign equivariance    : {checks['sign']:.3e}")
        print(
            f"comparison to {checks['comparison_method']} "
            f"(epsilon={checks['comparison_epsilon']:g}): "
            f"{checks['comparison_relative_l2']:.3e} "
            "(relative L2, first 8 check samples)"
        )
        if propagation_rule == "lrp0":
            print(
                "LRP-0 denominator QA: "
                f"hard |z|>{LRP0_DENOMINATOR_ATOL:.1e}; warn when "
                f"relative |z|<={LRP0_RELATIVE_DENOMINATOR_THRESHOLD:.1e}"
            )
            print(
                "minimum active |z|  : "
                f"{checks['minimum_absolute_denominator']:.3e}"
            )
            print(
                "minimum relative |z|: "
                f"{checks['minimum_relative_denominator']:.3e}"
            )
            print(
                "maximum |message|   : "
                f"{checks['maximum_absolute_message']:.3e}"
            )
            print(
                "maximum |relevance| : "
                f"{checks['maximum_absolute_relevance']:.3e} Sv"
            )
            print(
                "inactive exact 0/0  : "
                f"{checks['inactive_zero_denominator_count']} "
                "(assigned zero message)"
            )
            print(
                "low-relative warnings: "
                f"{checks['low_relative_denominator_count']}"
            )
        return None

    ensemble = TrainedEnsemble.load(nn_path, n_folds, n_ensembles)
    if ensemble.pcas_x is not None:
        raise RuntimeError("Stage-19 LRP does not support PCA-X models")
    if ensemble.pcas_y is None:
        raise RuntimeError("Stage-19 currently requires fold-specific PCA-Y")

    # Accumulate moments in float32; calculate batch relevance in float64.
    relevance_mean = np.zeros((n_time, n_features), dtype=np.float32)
    relevance_m2 = np.zeros_like(relevance_mean)
    relevance_abs_mean = np.zeros_like(relevance_mean)
    target_prediction = np.empty((n_members, n_time), dtype=np.float64)
    target_centered = np.empty_like(target_prediction)
    target_offsets = np.empty(n_members, dtype=np.float64)
    internal_bias = np.empty_like(target_prediction)
    stabilizer_remainder = np.empty_like(target_prediction)
    feature_residual = np.empty_like(target_prediction)
    accounted_residual = np.empty_like(target_prediction)
    branch_scores = np.empty((n_members, n_time, 2), dtype=np.float64)
    covariate_sum = np.empty(
        (n_members, n_time, n_covariates), dtype=np.float32
    )
    covariate_abs_sum = np.empty_like(covariate_sum)
    member_fold = np.repeat(
        np.arange(1, n_folds + 1, dtype=np.int16), n_ensembles
    )
    member_ensemble = np.tile(
        np.arange(1, n_ensembles + 1, dtype=np.int16), n_folds
    )
    target_variances = np.empty(n_folds, dtype=np.float64)
    pca_variance_fraction = np.empty(n_folds, dtype=np.float64)
    keras_trace_error = np.empty(n_members, dtype=np.float64)
    head_error_members = np.zeros(n_members, dtype=np.float64)
    minimum_absolute_denominator_members = np.full(
        n_members, np.inf, dtype=np.float64
    )
    minimum_relative_denominator_members = np.full(
        n_members, np.inf, dtype=np.float64
    )
    maximum_absolute_message_members = np.zeros(
        n_members, dtype=np.float64
    )
    maximum_absolute_relevance_members = np.zeros(
        n_members, dtype=np.float64
    )
    inactive_zero_denominator_count_members = np.zeros(
        n_members, dtype=np.int64
    )
    low_relative_denominator_count_members = np.zeros(
        n_members, dtype=np.int64
    )
    rule_remainder_max_abs_members = np.zeros(
        n_members, dtype=np.float64
    )
    batch_check_error = np.nan
    sign_check_error = np.nan
    comparison_relative_l2 = np.nan
    comparison_epsilon = (
        LRP0_REFERENCE_EPSILON
        if propagation_rule == "lrp0"
        else epsilon * 10.0
    )
    comparison_method = lrp_method_name(comparison_epsilon, "epsilon")

    member_h5 = None
    member_dataset = None
    if save_member_relevance and not validate_all:
        import h5py

        member_tmp.unlink(missing_ok=True)
        member_h5 = h5py.File(member_tmp, "w")
        chunks = (1, min(batch_size, n_time), min(512, n_features))
        member_dataset = member_h5.create_dataset(
            "relevance_members",
            shape=(n_members, n_time, n_features),
            dtype="f4",
            chunks=chunks,
            compression="lzf",
            shuffle=True,
        )
        member_dataset.attrs["dimensions"] = np.asarray(
            ["member", "time", "feature"], dtype="S"
        )
        member_dataset.attrs["lrp_method"] = method_name
        member_dataset.attrs["lrp_epsilon"] = epsilon

    run_succeeded = False
    try:
        for fold_index in range(n_folds):
            scaler_x = ensemble.scalers_x[fold_index]
            scaler_y = ensemble.scalers_y[fold_index]
            pca_y = ensemble.pcas_y[fold_index]
            if int(getattr(scaler_x, "n_features_in_", n_features)) != n_features:
                raise RuntimeError(
                    f"Fold {fold_index + 1} input scaler width differs from inputs"
                )
            if int(getattr(scaler_y, "n_features_in_", train_mask.sum())) != int(
                train_mask.sum()
            ):
                raise RuntimeError(
                    f"Fold {fold_index + 1} target scaler differs from Psi_mask"
                )
            target_variances[fold_index] = _target_variance(
                scaler_y, target.valid_output_index
            )
            if target_variances[fold_index] <= 1e-12:
                raise ValueError(
                    f"Requested target has zero training variance in fold "
                    f"{fold_index + 1}"
                )
            pca_variance_fraction[fold_index] = float(
                np.sum(pca_y.explained_variance_ratio_)
            )
            head = physical_output_head(
                scaler_y, pca_y, target.valid_output_index, sign=target_sign
            )
            explainers = [
                prepare_dbnn_physical_target_explainer(
                    model,
                    head,
                    epsilon=epsilon,
                    propagation_rule=propagation_rule,
                )
                for model in ensemble.models[fold_index]
            ]
            for ensemble_index in range(n_ensembles):
                target_offsets[fold_index * n_ensembles + ensemble_index] = head.offset

            for start in range(0, n_time, batch_size):
                stop = min(start + batch_size, n_time)
                section = slice(start, stop)
                x_fold = ensemble.transform_inputs(fold_index, x_raw[section])
                for ensemble_index, (model, explainer) in enumerate(
                    zip(ensemble.models[fold_index], explainers)
                ):
                    member_index = fold_index * n_ensembles + ensemble_index
                    try:
                        explanation = explainer.explain(x_fold)
                    except FloatingPointError as exc:
                        batch_realizations = np.unique(
                            realization_index[section]
                        ).tolist()
                        raise FloatingPointError(
                            "Stage-19 LRP failed for "
                            f"fold {fold_index + 1}/member "
                            f"{ensemble_index + 1}; global rows "
                            f"{start}:{stop - 1}, months "
                            f"{time_month[start]}..{time_month[stop - 1]}, "
                            f"realizations={batch_realizations}: {exc}"
                        ) from exc
                    expected_relevance_shape = (stop - start, n_features)
                    if explanation.relevance.shape != expected_relevance_shape:
                        raise RuntimeError(
                            f"Fold {fold_index + 1}/member {ensemble_index + 1}: "
                            f"relevance shape {explanation.relevance.shape}; "
                            f"expected {expected_relevance_shape}"
                        )
                    via_sklearn = _physical_from_scores(
                        scaler_y, pca_y, explanation.model_output,
                        target.valid_output_index, target_sign,
                    )
                    head_error = float(np.max(np.abs(
                        explanation.physical_prediction - via_sklearn
                    )))
                    head_error_members[member_index] = max(
                        head_error_members[member_index], head_error
                    )
                    if head_error > HEAD_TOLERANCE_SV:
                        raise RuntimeError(
                            f"Fold {fold_index + 1}/member {ensemble_index + 1}: "
                            f"physical-head mismatch {head_error:.3g} Sv"
                        )
                    accounting_error = float(np.max(np.abs(
                        explanation.accounted_conservation_residual
                    )))
                    if accounting_error > ACCOUNTING_TOLERANCE_SV:
                        raise RuntimeError(
                            f"Fold {fold_index + 1}/member {ensemble_index + 1}: "
                            f"accounting residual {accounting_error:.3g} Sv"
                        )
                    rule_remainder_error = float(np.max(np.abs(
                        explanation.stabilizer_remainder
                    )))
                    if (
                        propagation_rule == "lrp0"
                        and rule_remainder_error > ACCOUNTING_TOLERANCE_SV
                    ):
                        raise RuntimeError(
                            f"Fold {fold_index + 1}/member "
                            f"{ensemble_index + 1}, rows {start}:{stop - 1}: "
                            "exact LRP-0 produced a non-roundoff rule "
                            f"remainder of {rule_remainder_error:.3g} Sv"
                        )
                    minimum_absolute_denominator_members[member_index] = min(
                        minimum_absolute_denominator_members[member_index],
                        explanation.minimum_absolute_denominator,
                    )
                    minimum_relative_denominator_members[member_index] = min(
                        minimum_relative_denominator_members[member_index],
                        explanation.minimum_relative_denominator,
                    )
                    maximum_absolute_message_members[member_index] = max(
                        maximum_absolute_message_members[member_index],
                        explanation.maximum_absolute_message,
                    )
                    maximum_absolute_relevance_members[member_index] = max(
                        maximum_absolute_relevance_members[member_index],
                        float(np.max(np.abs(explanation.relevance))),
                    )
                    inactive_zero_denominator_count_members[member_index] += (
                        explanation.inactive_zero_denominator_count
                    )
                    low_relative_denominator_count_members[member_index] += (
                        explanation.low_relative_denominator_count
                    )
                    rule_remainder_max_abs_members[member_index] = max(
                        rule_remainder_max_abs_members[member_index],
                        rule_remainder_error,
                    )
                    if start == 0:
                        keras_scores = np.asarray(
                            model.predict(x_fold, verbose=0), dtype=float
                        )
                        keras_physical = _physical_from_scores(
                            scaler_y, pca_y, keras_scores,
                            target.valid_output_index, target_sign,
                        )
                        keras_trace_error[member_index] = float(np.max(np.abs(
                            explanation.physical_prediction - keras_physical
                        )))
                        if keras_trace_error[member_index] > FORWARD_TOLERANCE_SV:
                            raise RuntimeError(
                                f"Fold {fold_index + 1}/member "
                                f"{ensemble_index + 1}: NumPy trace differs "
                                f"from Keras by {keras_trace_error[member_index]:.3g} Sv"
                            )
                    if member_index == 0 and start == 0:
                        checks = _self_checks(
                            model, x_fold[:min(8, x_fold.shape[0])],
                            scaler_y, pca_y, target, target_sign, epsilon,
                            propagation_rule,
                            explainer=explainer,
                        )
                        batch_check_error = checks["batch"]
                        sign_check_error = checks["sign"]
                        comparison_relative_l2 = checks[
                            "comparison_relative_l2"
                        ]
                        comparison_epsilon = checks["comparison_epsilon"]
                        comparison_method = checks["comparison_method"]

                    relevance = explanation.relevance.astype(np.float32)
                    if not np.isfinite(relevance).all():
                        raise FloatingPointError(
                            f"Fold {fold_index + 1}/member "
                            f"{ensemble_index + 1}, rows {start}:{stop - 1}: "
                            "relevance overflowed when converted to the "
                            "float32 output representation"
                        )
                    count = np.float32(member_index + 1)
                    delta = relevance - relevance_mean[section]
                    relevance_mean[section] += delta / count
                    relevance_m2[section] += delta * (
                        relevance - relevance_mean[section]
                    )
                    absolute = np.abs(relevance)
                    relevance_abs_mean[section] += (
                        absolute - relevance_abs_mean[section]
                    ) / count
                    blocks = relevance.reshape(
                        stop - start, n_covariates, n_mascons
                    )
                    covariate_sum[member_index, section] = blocks.sum(axis=2)
                    covariate_abs_sum[member_index, section] = np.abs(blocks).sum(axis=2)
                    target_prediction[member_index, section] = explanation.physical_prediction
                    target_centered[member_index, section] = explanation.centered_score
                    internal_bias[member_index, section] = explanation.internal_bias_relevance
                    stabilizer_remainder[member_index, section] = explanation.stabilizer_remainder
                    feature_residual[member_index, section] = (
                        explanation.feature_conservation_residual
                    )
                    accounted_residual[member_index, section] = (
                        explanation.accounted_conservation_residual
                    )
                    branch_scores[member_index, section] = explanation.branch_scores
                    if member_dataset is not None:
                        member_dataset[member_index, section, :] = relevance
                    del explanation, relevance, blocks
                del x_fold
            for ensemble_index in range(n_ensembles):
                member_index = fold_index * n_ensembles + ensemble_index
                print(
                    f"fold {fold_index + 1}/{n_folds}, member "
                    f"{ensemble_index + 1}/{n_ensembles}: "
                    f"target={np.mean(target_prediction[member_index]):+.3f} Sv; "
                    f"feature residual RMS="
                    f"{np.sqrt(np.mean(feature_residual[member_index]**2)):.3f} Sv; "
                    f"accounted max="
                    f"{np.max(np.abs(accounted_residual[member_index])):.2e} Sv"
                )
                if propagation_rule == "lrp0":
                    print(
                        "  LRP-0 safety: min |z|="
                        f"{minimum_absolute_denominator_members[member_index]:.3e}, "
                        "min relative |z|="
                        f"{minimum_relative_denominator_members[member_index]:.3e}, "
                        "max |message|="
                        f"{maximum_absolute_message_members[member_index]:.3e}, "
                        "max |R|="
                        f"{maximum_absolute_relevance_members[member_index]:.3e} Sv, "
                        "inactive 0/0="
                        f"{inactive_zero_denominator_count_members[member_index]}, "
                        "low-relative warnings="
                        f"{low_relative_denominator_count_members[member_index]}, "
                        "rule remainder max="
                        f"{rule_remainder_max_abs_members[member_index]:.3e} Sv"
                    )
            del explainers
            gc.collect()

        relevance_std = np.sqrt(
            np.maximum(relevance_m2 / np.float32(n_members), 0.0)
        ).astype(np.float32)
        prediction_mean = target_prediction.mean(axis=0)
        prediction_std = target_prediction.std(axis=0)
        stage10_mean_error = float(np.max(np.abs(
            prediction_mean - stage10_target_mean
        )))
        if stage10_mean_error > FORWARD_TOLERANCE_SV:
            raise RuntimeError(
                "LRP ensemble target does not reproduce completed Stage 10: "
                f"max difference={stage10_mean_error:.4g} Sv"
            )

        if propagation_rule == "lrp0":
            lrp0_diagnostic_arrays = (
                minimum_absolute_denominator_members,
                minimum_relative_denominator_members,
                maximum_absolute_message_members,
                maximum_absolute_relevance_members,
                rule_remainder_max_abs_members,
            )
            if not all(np.isfinite(values).all() for values in lrp0_diagnostic_arrays):
                raise FloatingPointError(
                    "Full-record LRP-0 diagnostics contain NaN or Inf"
                )
            if np.min(minimum_absolute_denominator_members) <= LRP0_DENOMINATOR_ATOL:
                raise FloatingPointError(
                    "Full-record LRP-0 minimum active |z| does not exceed "
                    f"{LRP0_DENOMINATOR_ATOL:.1e}"
                )

        corr, rmse, bias = _skill(prediction_mean, stage10_target_truth)
        corr_by_realization = np.empty(n_realizations, dtype=float)
        rmse_by_realization = np.empty(n_realizations, dtype=float)
        bias_by_realization = np.empty(n_realizations, dtype=float)
        for i, group in enumerate(realization_groups):
            values = _skill(
                prediction_mean[group], stage10_target_truth[group]
            )
            corr_by_realization[i], rmse_by_realization[i], bias_by_realization[i] = values

        block_shape = (n_time, n_covariates, n_mascons)
        relevance_mean_3d = relevance_mean.reshape(block_shape)
        relevance_std_3d = relevance_std.reshape(block_shape)
        relevance_abs_mean_3d = relevance_abs_mean.reshape(block_shape)
        relevance_mean_by_realization = np.stack([
            relevance_mean[group].mean(axis=0)
            for group in realization_groups
        ]).reshape(n_realizations, n_covariates, n_mascons)
        relevance_abs_by_realization = np.stack([
            relevance_abs_mean[group].mean(axis=0)
            for group in realization_groups
        ]).reshape(n_realizations, n_covariates, n_mascons)

        evaluation_baseline_value = np.nan
        evaluation_core_sigma2 = np.nan
        if case.moc_baseline is not None:
            evaluation_baseline_value = float(
                np.asarray(case.moc_baseline)[
                    target.level_index, target.latitude_index
                ]
            )
            evaluation_cores = locate_cell_cores(
                np.asarray(case.moc_baseline), lat, sigma2
            )
            core_values = (
                evaluation_cores.mid_sigma2
                if target_core == "mid" else evaluation_cores.abyssal_sigma2
            )
            evaluation_core_sigma2 = float(
                core_values[target.latitude_index]
            )

        member_file_value = str(member_file.resolve()) if member_dataset is not None else ""
        rule_remainder_policy = (
            "exact LRP-0 has no stabilizer; saved values are numerical roundoff only"
            if propagation_rule == "lrp0"
            else "epsilon-stabilizer remainder saved separately from input relevance"
        )
        payload = {
            "relevance_mean": relevance_mean_3d,
            "relevance_std": relevance_std_3d,
            "relevance_abs_mean": relevance_abs_mean_3d,
            "relevance_mean_by_realization": relevance_mean_by_realization.astype(np.float32),
            "relevance_abs_mean_by_realization": relevance_abs_by_realization.astype(np.float32),
            "relevance_by_realization": relevance_mean_by_realization.astype(np.float32),
            "relevance_covariate_sum_members": covariate_sum,
            "relevance_covariate_abs_sum_members": covariate_abs_sum,
            "target_prediction_members": target_prediction.astype(np.float32),
            "target_prediction_mean": prediction_mean.astype(np.float32),
            "target_prediction_std": prediction_std.astype(np.float32),
            "prediction_mean": prediction_mean.astype(np.float32),
            "target_truth": stage10_target_truth.astype(np.float32),
            "target_centered_members": target_centered.astype(np.float32),
            "target_output_offset_members": target_offsets.astype(np.float32),
            "internal_bias_relevance_members": internal_bias.astype(np.float32),
            "stabilizer_remainder_members": stabilizer_remainder.astype(np.float32),
            "feature_conservation_residual_members": feature_residual.astype(np.float32),
            "accounted_conservation_residual_members": accounted_residual.astype(np.float32),
            "branch_centered_scores_members": branch_scores.astype(np.float32),
            "member_fold": member_fold,
            "member_ensemble": member_ensemble,
            "target_corr": np.float64(corr),
            "target_rmse": np.float64(rmse),
            "target_bias": np.float64(bias),
            "target_corr_by_realization": corr_by_realization,
            "target_rmse_by_realization": rmse_by_realization,
            "target_bias_by_realization": bias_by_realization,
            "stage10_target_mean": stage10_target_mean.astype(np.float32),
            "stage10_target_truth": stage10_target_truth.astype(np.float32),
            "time_month": time_month.astype("U7"),
            "realization_index": realization_index,
            "realization_labels": realization_labels,
            "realization_sample_counts": realization_counts,
            "n_realizations": np.int64(n_realizations),
            "months_per_realization": np.int64(
                realization_counts[0]
                if np.all(realization_counts == realization_counts[0]) else -1
            ),
            "relevance_dimensions": np.asarray(
                ["time", "covariate", "mascon"], dtype="U"
            ),
            "relevance_by_realization_dimensions": np.asarray(
                ["realization", "covariate", "mascon"], dtype="U"
            ),
            "covariate_names": np.asarray(covariates.names, dtype="U"),
            "input_source_names": np.asarray(case.input_source_names, dtype="U"),
            "input_source_files": np.asarray(case.input_files, dtype="U"),
            "input_baseline_specs": np.asarray(case.input_baseline_specs, dtype="U"),
            "input_wind_sources": np.asarray(case.input_wind_sources, dtype="U"),
            "mascon_lon": train_lon,
            "mascon_lat": train_lat_mascon,
            "feature_block_offsets": feature_offsets,
            "moc_latitude_grid": lat,
            "moc_sigma2_grid": sigma2,
            "psi_mask": train_mask,
            "evaluation_psi_mask": eval_mask,
            "target_mode": np.str_(target.mode),
            "target_core_reference": np.str_("training_baseline"),
            "target_sign": np.float64(target_sign),
            "target_latitude": np.float64(target.latitude),
            "target_sigma2": np.float64(target.sigma2),
            "target_latitude_index_python": np.int64(target.latitude_index),
            "target_level_index_python": np.int64(target.level_index),
            "target_flat_grid_index_python": np.int64(target.flat_grid_index),
            "target_valid_output_index_python": np.int64(target.valid_output_index),
            "target_evaluation_output_index_python": np.int64(evaluation_output_index),
            "target_truth_available": np.bool_(truth_available),
            "target_baseline_psi_2004_2009": np.float64(target.baseline_value),
            "target_signed_baseline_2004_2009": np.float64(
                target.baseline_value * target_sign
            ),
            "target_evaluation_baseline_psi_2004_2009": np.float64(
                evaluation_baseline_value
            ),
            "target_evaluation_core_sigma2": np.float64(evaluation_core_sigma2),
            "moc_convention": np.str_(training_config.get("moc_convention", MOC_CONVENTION)),
            "baseline_period": np.asarray(BASELINE_YEARS, dtype=np.int16),
            "lrp_method": np.str_(method_name),
            "lrp_epsilon": np.float64(epsilon),
            "lrp_propagation_rule": np.str_(propagation_rule),
            "lrp_comparison_method_first8": np.str_(comparison_method),
            "lrp_comparison_epsilon_first8": np.float64(comparison_epsilon),
            "lrp_comparison_relative_l2_first8": np.float64(
                comparison_relative_l2
            ),
            "activation_relevance_rule": np.str_(
                "identity pass-through (including swish)"
            ),
            "bias_policy": np.str_(
                "saved separately; not assigned to mascon inputs"
            ),
            "output_offset_policy": np.str_(
                "inverse-PCA/scaler offset saved separately"
            ),
            "input_relevance_domain": np.str_(
                "fold-standardized model inputs; feature identities remain registered mascons"
            ),
            "accounting_identity": np.str_(
                "prediction = sum(input relevance) + internal bias + stabilizer + output offset"
            ),
            "lrp_rule_remainder_policy": np.str_(rule_remainder_policy),
            "lrp_rule_remainder_max_abs_sv_by_member": (
                rule_remainder_max_abs_members
            ),
            "lrp_rule_remainder_max_abs_sv": np.float64(
                np.max(rule_remainder_max_abs_members)
            ),
            "lrp_maximum_absolute_relevance_sv_by_member": (
                maximum_absolute_relevance_members
            ),
            "target_training_variance_by_fold": target_variances,
            "pca_explained_variance_fraction_by_fold": pca_variance_fraction,
            "keras_trace_first_batch_max_abs_difference_sv_by_member": (
                keras_trace_error
            ),
            "physical_head_max_abs_difference_sv_by_member": head_error_members,
            "stage10_mean_max_abs_difference_sv": np.float64(stage10_mean_error),
            "stage10_truth_max_abs_difference_sv": np.float64(truth_difference),
            "batch_invariance_max_abs_difference": np.float64(batch_check_error),
            "sign_equivariance_max_abs_difference": np.float64(sign_check_error),
            "case_name": np.str_(cmip_name),
            "case_out_tag": np.str_(out_tag),
            "case_realization_tag": np.str_(realization_tag),
            "stage10_prediction_file": np.str_(str(pred_path.resolve())),
            "run_id": np.str_(ACTIVE_RUN_ID),
            "analysis_run_id": np.str_(ACTIVE_RUN_ID),
            "training_source_run_id": np.str_(ACTIVE_RUN_ID),
            "trained_on": np.str_(trained_on),
            "training_experiment": np.str_(experiment),
            "training_random_seed": np.int64(training_seed),
            "training_seed_map_json": np.str_(seed_map_json),
            "software_versions_json": np.str_(
                json.dumps(software_versions, sort_keys=True)
            ),
            "tensorflow_runtime_json": np.str_(
                json.dumps(tensorflow_runtime, sort_keys=True)
            ),
            "member_relevance_storage": np.str_(
                "external_hdf5" if member_dataset is not None else "not_saved"
            ),
            "member_relevance_file": np.str_(member_file_value),
            "created_utc": np.str_(datetime.now(timezone.utc).isoformat()),
            "stage": np.str_("19_model_LRP"),
        }

        if propagation_rule == "lrp0":
            payload.update({
                "lrp0_denominator_atol": np.float64(
                    LRP0_DENOMINATOR_ATOL
                ),
                "lrp0_relative_denominator_threshold": np.float64(
                    LRP0_RELATIVE_DENOMINATOR_THRESHOLD
                ),
                "lrp0_minimum_absolute_denominator_by_member": (
                    minimum_absolute_denominator_members
                ),
                "lrp0_minimum_relative_denominator_by_member": (
                    minimum_relative_denominator_members
                ),
                "lrp0_maximum_absolute_message_by_member": (
                    maximum_absolute_message_members
                ),
                "lrp0_maximum_absolute_relevance_by_member": (
                    maximum_absolute_relevance_members
                ),
                "lrp0_inactive_zero_denominator_count_by_member": (
                    inactive_zero_denominator_count_members
                ),
                "lrp0_low_relative_denominator_count_by_member": (
                    low_relative_denominator_count_members
                ),
                "lrp0_rule_remainder_max_abs_sv_by_member": (
                    rule_remainder_max_abs_members
                ),
                "lrp0_minimum_absolute_denominator": np.float64(
                    np.min(minimum_absolute_denominator_members)
                ),
                "lrp0_minimum_relative_denominator": np.float64(
                    np.min(minimum_relative_denominator_members)
                ),
                "lrp0_maximum_absolute_message": np.float64(
                    np.max(maximum_absolute_message_members)
                ),
                "lrp0_inactive_zero_denominator_count": np.int64(
                    np.sum(inactive_zero_denominator_count_members)
                ),
            })
        else:
            # Comparison between epsilon and 10 x epsilon on the first 8 samples.
            payload["epsilon_x10_relative_l2_first8"] = np.float64(
                comparison_relative_l2
            )

        if validate_all:
            print("=" * 78)
            print("FULL-ENSEMBLE VALIDATION PASSED - nothing written")
            print(f"Stage-10 mean parity : {stage10_mean_error:.3e} Sv")
            print(
                "accounting max       : "
                f"{np.max(np.abs(accounted_residual)):.3e} Sv"
            )
            print(
                "rule remainder max   : "
                f"{np.max(rule_remainder_max_abs_members):.3e} Sv"
            )
            print(
                f"comparison to {comparison_method} "
                f"(epsilon={comparison_epsilon:g}): "
                f"{comparison_relative_l2:.3e} relative L2 (first 8)"
            )
            if propagation_rule == "lrp0":
                print(
                    "LRP-0 global safety  : min |z|="
                    f"{np.min(minimum_absolute_denominator_members):.3e}, "
                    "min relative |z|="
                    f"{np.min(minimum_relative_denominator_members):.3e}, "
                    "max |message|="
                    f"{np.max(maximum_absolute_message_members):.3e}, "
                    "max |R|="
                    f"{np.max(maximum_absolute_relevance_members):.3e} Sv, "
                    "inactive 0/0="
                    f"{np.sum(inactive_zero_denominator_count_members)}, "
                    "low-relative warnings="
                    f"{np.sum(low_relative_denominator_count_members)}"
                )
            print("=" * 78)
            run_succeeded = True
            return None

        if member_h5 is not None:
            member_h5.flush()
            member_h5.close()
            member_h5 = None
            os.replace(member_tmp, member_file)

        _atomic_save_mat(output_file, payload)
        run_succeeded = True

        print("=" * 78)
        print("MODEL-TEST LRP COMPLETE")
        print(f"saved     : {output_file}")
        if save_member_relevance:
            print(f"members   : {member_file}")
        print(f"Stage-10 parity: {stage10_mean_error:.3e} Sv")
        print(
            "Accounting max: "
            f"{np.max(np.abs(accounted_residual)):.3e} Sv"
        )
        print(
            "Interpret relevance_mean as signed ensemble-mean contribution "
            "and relevance_abs_mean as magnitude; they are not interchangeable."
        )
        print("=" * 78)
        return output_file
    finally:
        if member_h5 is not None:
            member_h5.close()
        if not run_succeeded:
            member_tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        trained_on=args.trained_on,
        experiment=args.experiment,
        covariate_names=args.covariates,
        case_spec=args.case,
        target_latitude=args.target_lat,
        target_core=args.target_core,
        target_sigma2=args.target_sigma2,
        target_sign=args.target_sign,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        save_member_relevance=args.save_member_relevance,
        check_only=args.check_only,
        validate_all=args.validate_all,
    )
