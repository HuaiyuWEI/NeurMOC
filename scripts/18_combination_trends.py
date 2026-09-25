"""Stage 18: estimate trends for each satellite-product combination.

Use the Stage-15 trend estimator and budget to calculate product-specific
slopes, serial uncertainty, and ensemble uncertainty. Shared mapping and
product-choice terms are retained from the common budget.
"""

import argparse
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

# This stage gains little from threaded MKL. Select sequential execution
# before importing NumPy; an explicit environment setting takes precedence.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.evaluation import (
    BASELINE_EXPERIMENT,
    COVARIATES_ALL,
    PERF_ROOT,
    REALWORLD_ROOT,
    model_dir,
)
from neurmoc.io_utils import load_npz_or_mat
from neurmoc.moc_utils import TWO_SIGMA_ALPHA, robust_trend
from neurmoc.satellite_products import (
    DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC,
    uncertainty_product_registry,
)
from neurmoc.results import (
    exact_common_month_indices,
    load_real_world,
    trim_and_validate_member_predictions,
)

EDGE_MONTHS = 12
#: Use the same trend-estimator settings as Stage 15.
TREND_METHOD = "mbb"
TREND_BLOCK_MONTHS = 48
N_BOOT_TREND = 1000
TREND_SEED = 0
N_SIGMA = 2.0
ALPHA_FDR = 2.0 * TWO_SIGMA_ALPHA

COVARIATES = COVARIATES_ALL
EXPERIMENT = BASELINE_EXPERIMENT
# Match the Stage-15 output tag and product registry.
BUDGET_TAG = ""
_cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_cli.add_argument("--budget-tag", default=None)
_cli.add_argument(
    "--without-grace", action="store_true",
    help=(
        "write separate statistics with the GRACE measurement-noise term "
        "omitted from every product tuple"
    ),
)
_cli.add_argument("-E", "--experiment", default=None)
_cli.add_argument("-x", "--covariates", default=None)
_args = _cli.parse_known_args()[0]
COVARIATES = (
    _args.covariates
    or os.environ.get("NEURMOC_FIG_COVARIATES")
    or COVARIATES
).replace(",", "+")
EXPERIMENT = (
    _args.experiment
    or os.environ.get("NEURMOC_FIG_EXPERIMENT")
    or EXPERIMENT
)
NN_DIR = model_dir(
    root=REALWORLD_ROOT, experiment=EXPERIMENT, covariates=COVARIATES
)
SOURCE_NN_DIR = model_dir(
    root=PERF_ROOT, experiment=EXPERIMENT, covariates=COVARIATES
)
OUT_DIR = NN_DIR / "RealWorld"
BUDGET_TAG = (
    _args.budget_tag
    if _args.budget_tag is not None
    else os.environ.get("NEURMOC_TREND_BUDGET_TAG", BUDGET_TAG)
).strip()
if BUDGET_TAG.startswith("_") or any(c in BUDGET_TAG for c in ("/", "\\")):
    raise SystemExit("BUDGET_TAG must be a filename-safe tag without '_' prefix")
_SUFFIX = f"_{BUDGET_TAG}" if BUDGET_TAG else ""
BUDGET_FILE = OUT_DIR / f"trend_error_budget{_SUFFIX}.npz"
GRACE_FILE = OUT_DIR / f"grace_noise_budget{_SUFFIX}.npz"
CSR_GRACE_FILE = OUT_DIR / f"grace_noise_budget_obpCSR{_SUFFIX}.npz"
_GRACE_STEM = "_without_grace" if _args.without_grace else ""
REFERENCE_FILE = OUT_DIR / f"real_world_trend_stats{_GRACE_STEM}{_SUFFIX}.npz"
OUT_FILE = OUT_DIR / f"combination_trend_stats{_GRACE_STEM}{_SUFFIX}.npz"

# %% [1] Fixed budget terms ----------------------------------------------------
if not BUDGET_FILE.is_file():
    raise SystemExit(f"{BUDGET_FILE} missing - run 15_compute_trend_budget.py")
with np.load(BUDGET_FILE, allow_pickle=False) as _b:
    _required = {
        "sigma_map", "sigma_sate", "sigma_eps", "run_id",
        "training_source_run_id", "covariates",
        "cmip_dataset_id", "satellite_dataset_id", "training_experiment",
        "time_month_int", "n_combos", "combo_tags", "combo_obp_sources",
        "combo_ssh_sources", "combo_wind_sources",
        "product_spread_include_gsfc", "product_spread_definition",
    }
    _missing = sorted(_required - set(_b.files))
    if _missing:
        raise SystemExit(
            f"{BUDGET_FILE.name} predates the exact product/time registry "
            f"(missing {_missing}); rerun Stage 15"
        )
    sigma_map = np.asarray(_b["sigma_map"], dtype=float)
    sigma_sate = np.asarray(_b["sigma_sate"], dtype=float)
    sigma_eps = np.asarray(_b["sigma_eps"], dtype=float)
    budget_run_id = str(np.asarray(_b["run_id"]).item())
    budget_covariates = str(np.asarray(_b["covariates"]).item())
    _budget_provenance = {
        name: str(np.asarray(_b[name]).squeeze().item())
        for name in (
            "run_id", "training_source_run_id", "cmip_dataset_id",
            "satellite_dataset_id",
            "training_experiment",
        )
    }
    reference_months = np.asarray(
        _b["time_month_int"], dtype=np.int64
    ).reshape(-1).astype("datetime64[M]")
    _saved_registry = list(zip(
        np.asarray(_b["combo_obp_sources"]).astype(str).reshape(-1).tolist(),
        np.asarray(_b["combo_ssh_sources"]).astype(str).reshape(-1).tolist(),
        np.asarray(_b["combo_wind_sources"]).astype(str).reshape(-1).tolist(),
        np.asarray(_b["combo_tags"]).astype(str).reshape(-1).tolist(),
    ))
    _saved_n_combos = int(np.asarray(_b["n_combos"]).squeeze())
    product_spread_include_gsfc = bool(
        np.asarray(_b["product_spread_include_gsfc"]).squeeze().item()
    )
    product_spread_definition = str(
        np.asarray(_b["product_spread_definition"]).squeeze().item()
    )
if budget_covariates != COVARIATES:
    raise SystemExit(
        f"{BUDGET_FILE.name} covariates={budget_covariates!r}; expected "
        f"{COVARIATES!r}"
    )
if _budget_provenance["training_experiment"] != EXPERIMENT:
    raise SystemExit(
        f"{BUDGET_FILE.name} training_experiment="
        f"{_budget_provenance['training_experiment']!r}; expected "
        f"{EXPERIMENT!r}"
    )
if (
    not BUDGET_TAG
    and product_spread_include_gsfc
    != DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC
):
    raise SystemExit(
        f"untagged budget {BUDGET_FILE.name} has "
        f"product_spread_include_gsfc={product_spread_include_gsfc}, but "
        "the scientific settings require "
        f"{DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC}; rerun Stage 15 with the "
        "profile default or use a tagged output"
    )
_expected_registry = uncertainty_product_registry(
    COVARIATES, include_gsfc=product_spread_include_gsfc
)
if (
    _saved_n_combos != len(_saved_registry)
    or _saved_registry != _expected_registry
):
    raise SystemExit(
        f"{BUDGET_FILE.name} product registry differs from its saved "
        "GSFC-inclusion policy; rerun Stage 15"
    )
COMBINATION_REGISTRY = _saved_registry


def _npz_scalar_text(handle, name: str) -> str:
    """Return one required scalar text value without accepting object arrays."""
    value = np.asarray(handle[name])
    if value.size != 1:
        raise SystemExit(f"{name} must be scalar, got shape {value.shape}")
    item = value.squeeze().item()
    return item.decode() if isinstance(item, bytes) else str(item)


def _load_exact_grace_budget(path: Path, expected_tuple):
    """Load one Stage-16 term only when its saved reconstruction tuple matches."""
    if not path.is_file():
        return None
    point_estimate = OUT_DIR / f"Pred_RealWorld{expected_tuple[3]}.mat"
    if not point_estimate.is_file():
        raise SystemExit(
            f"{point_estimate.name} is missing; rerun Stage 14 before "
            "pricing its GRACE measurement-noise response"
        )
    if path.stat().st_mtime_ns < BUDGET_FILE.stat().st_mtime_ns:
        raise SystemExit(
            f"{path.name} predates {BUDGET_FILE.name}; rerun Stage 16"
        )
    if path.stat().st_mtime_ns < point_estimate.stat().st_mtime_ns:
        raise SystemExit(
            f"{path.name} predates {point_estimate.name}; rerun Stage 16 "
            "for this exact reconstruction"
        )
    with np.load(path, allow_pickle=False) as handle:
        required = {
            "time_month", "run_id", "training_source_run_id",
            "satellite_dataset_id", "training_experiment", "covariates",
            "sigma_grace", "n_combos", "combo_tags",
            "combo_obp_sources", "combo_ssh_sources", "combo_wind_sources",
            "product_spread_include_gsfc", "product_spread_definition",
            "reconstruction_product_tag", "reconstruction_obp_source",
            "reconstruction_ssh_source", "reconstruction_wind_source",
            "measurement_noise_source", "measurement_noise_source_center",
            "measurement_noise_is_proxy", "measurement_noise_definition",
        }
        missing = sorted(required - set(handle.files))
        if missing:
            raise SystemExit(
                f"{path.name} is missing {missing}; rerun Stage 16"
            )
        field = np.asarray(handle["sigma_grace"], dtype=float)
        months = np.asarray(handle["time_month"]).astype(
            "datetime64[M]"
        ).reshape(-1)
        provenance = {
            name: _npz_scalar_text(handle, name)
            for name in (
                "run_id", "training_source_run_id", "satellite_dataset_id",
                "training_experiment", "covariates",
            )
        }
        registry = list(zip(
            np.asarray(handle["combo_obp_sources"]).astype(
                str
            ).reshape(-1).tolist(),
            np.asarray(handle["combo_ssh_sources"]).astype(
                str
            ).reshape(-1).tolist(),
            np.asarray(handle["combo_wind_sources"]).astype(
                str
            ).reshape(-1).tolist(),
            np.asarray(handle["combo_tags"]).astype(str).reshape(-1).tolist(),
        ))
        n_combos = int(np.asarray(handle["n_combos"]).squeeze())
        include_gsfc = bool(
            np.asarray(handle["product_spread_include_gsfc"]).squeeze().item()
        )
        definition = _npz_scalar_text(handle, "product_spread_definition")
        saved_tuple = (
            _npz_scalar_text(handle, "reconstruction_obp_source"),
            _npz_scalar_text(handle, "reconstruction_ssh_source"),
            _npz_scalar_text(handle, "reconstruction_wind_source"),
            _npz_scalar_text(handle, "reconstruction_product_tag"),
        )
        noise_source = _npz_scalar_text(handle, "measurement_noise_source")
        noise_center = _npz_scalar_text(
            handle, "measurement_noise_source_center"
        )
        noise_definition = _npz_scalar_text(
            handle, "measurement_noise_definition"
        )
        proxy_value = np.asarray(handle["measurement_noise_is_proxy"])
        if proxy_value.size != 1 or proxy_value.dtype.kind != "b":
            raise SystemExit(
                f"{path.name} measurement_noise_is_proxy must be scalar bool"
            )
        is_proxy = bool(proxy_value.squeeze().item())

    if field.shape != sigma_sate.shape or not np.isfinite(field).all() \
            or np.any(field < 0):
        raise SystemExit(
            f"{path.name} sigma_grace must be a finite, nonnegative "
            f"{sigma_sate.shape} field"
        )
    if not np.array_equal(months, reference_months):
        raise SystemExit(
            f"{path.name} months differ from the Stage-15 analysis interval; "
            "rerun Stage 16"
        )
    expected_provenance = {
        "run_id": _budget_provenance["run_id"],
        "training_source_run_id": _budget_provenance["training_source_run_id"],
        "satellite_dataset_id": _budget_provenance["satellite_dataset_id"],
        "training_experiment": _budget_provenance["training_experiment"],
        "covariates": COVARIATES,
    }
    if provenance != expected_provenance:
        raise SystemExit(
            f"{path.name} provenance differs from {BUDGET_FILE.name}"
        )
    if (
        n_combos != _saved_n_combos
        or registry != COMBINATION_REGISTRY
        or include_gsfc != product_spread_include_gsfc
        or definition != product_spread_definition
    ):
        raise SystemExit(
            f"{path.name} product-spread registry differs from "
            f"{BUDGET_FILE.name}; rerun Stage 16"
        )
    if saved_tuple != expected_tuple:
        raise SystemExit(
            f"{path.name} prices {saved_tuple}, expected {expected_tuple}; "
            "do not apply a GRACE response to a different product tuple"
        )
    expected_proxy = expected_tuple[0] == "GRACE_CSR"
    if (
        noise_center != "JPL"
        or not noise_source.strip()
        or not noise_definition.strip()
        or is_proxy != expected_proxy
    ):
        raise SystemExit(
            f"{path.name} has inconsistent measurement-noise provenance; "
            "rerun Stage 16"
        )
    return {
        "sigma_grace": field,
        "file": path.name,
        "is_proxy": is_proxy,
        "definition": noise_definition,
    }


# Reuse a GRACE uncertainty term only for its exact satellite-product tuple.
_grace_candidates = {
    "": GRACE_FILE,
    "_obpCSR": CSR_GRACE_FILE,
}
grace_budgets_by_tag = {}
required_grace_tags = []
if not _args.without_grace:
    for combo_tuple in COMBINATION_REGISTRY:
        obp_source, ssh_source, wind_source, tag = combo_tuple
        if tag not in _grace_candidates or obp_source not in {
            "GRACE", "GRACE_CSR"
        }:
            continue
        required_grace_tags.append(tag)
        expected = (obp_source, ssh_source, wind_source, tag)
        loaded = _load_exact_grace_budget(_grace_candidates[tag], expected)
        if loaded is not None:
            grace_budgets_by_tag[tag] = loaded

    missing_grace_tags = [
        tag for tag in required_grace_tags if tag not in grace_budgets_by_tag
    ]
    if missing_grace_tags:
        commands = []
        for tag in missing_grace_tags:
            source = "GRACE" if tag == "" else "GRACE_CSR"
            tag_args = (
                f' --budget-tag "{BUDGET_TAG}" --out-tag "{BUDGET_TAG}"'
                if BUDGET_TAG else ""
            )
            commands.append(
                "python scripts/16_grace_noise_montecarlo.py "
                f"--obp-source {source} -E \"{EXPERIMENT}\" "
                f"-x \"{COVARIATES}\"{tag_args}"
            )
        raise SystemExit(
            "Stage 18 needs the GRACE Monte Carlo results for both the JPL "
            "and CSR reconstructions. Run Stage 16 first:\n  "
            + "\n  ".join(commands)
        )

if _args.without_grace:
    grace_state = "explicitly omitted (--without-grace)"
elif grace_budgets_by_tag:
    grace_state = "priced for " + ", ".join(
        tag or "(default)" for tag in grace_budgets_by_tag
    )
else:
    grace_state = "absent (run 16_grace_noise_montecarlo.py)"
print(
    f"budget: {BUDGET_FILE.name} (run {budget_run_id}); "
    f"GRACE term {grace_state}"
)


class _ExactMemberReconstructor:
    """Recompute members for product variants missing saved members."""

    def __init__(self):
        stage14_path = Path(__file__).resolve().parent / (
            "14_reconstruct_real_world.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_neurmoc_stage14_member_fallback", stage14_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import Stage-14 helpers from {stage14_path}")
        stage14 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage14)
        self.stage14 = stage14
        # Convert the stored input-set tag to Stage-14's CLI form.
        self.covariates = stage14.prepare_covariate_config(
            COVARIATES.replace("+", ",")
        )
        training_config = stage14.validate_training_covariates(
            SOURCE_NN_DIR, tuple(self.covariates.names)
        )
        stage14.configure_tensorflow_runtime(
            seed=int(training_config.get("random_seed", 0))
        )
        self.n_folds = int(training_config["num_folds"])
        self.n_ensembles = int(training_config["nn_repeats"])
        self.ensemble = stage14.TrainedEnsemble.load(
            SOURCE_NN_DIR, self.n_folds, self.n_ensembles
        )
        self.n_members = self.n_folds * self.n_ensembles
        self.lpf = bool(stage14.lpf_tag_from_name(EXPERIMENT))

        mask_raw = load_npz_or_mat(
            SOURCE_NN_DIR / "Psi_mask", ["Psi_mask"]
        )
        self.mask = np.asarray(mask_raw["Psi_mask"]).squeeze().astype(bool)
        info = load_npz_or_mat(
            SOURCE_NN_DIR / "inputs_info", ["mascon_lon", "mascon_lat"]
        )
        self.train_lon = np.asarray(info["mascon_lon"]).squeeze()
        self.train_lat = np.asarray(info["mascon_lat"]).squeeze()

    @staticmethod
    def _normalized_paths(paths) -> tuple[str, ...]:
        return tuple(
            os.path.normcase(str(Path(path).resolve(strict=False)))
            for path in paths
        )

    def predict_aligned(
        self,
        combo,
        obp_source: str,
        ssh_source: str,
        wind_source: str,
        combo_months: np.ndarray,
        combo_take: np.ndarray,
    ) -> np.ndarray:
        """Return exact flat members on the Stage-15 common month axis."""
        stage14 = self.stage14
        # Assign every source explicitly for each product tuple.
        stage14.OBP_SOURCE = obp_source or "GRACE"
        stage14.SSH_SOURCE = ssh_source or "DUACS"
        stage14.USE_ERA5_WINDS = wind_source == "ERA5"
        assembled = stage14.assemble_inputs(self.covariates, lpf=self.lpf)

        expected_metadata = (
            (
                "input_covariates",
                tuple(assembled.covariate_names),
                tuple(combo.input_covariates),
            ),
            (
                "input_sources",
                tuple(assembled.source_names),
                tuple(combo.input_sources),
            ),
            (
                "input_baseline_specs",
                tuple(assembled.baseline_specs),
                tuple(combo.input_baseline_specs),
            ),
        )
        for name, rebuilt, saved in expected_metadata:
            if rebuilt != saved:
                raise RuntimeError(
                    f"{name} differs while rebuilding members: "
                    f"current={rebuilt}, saved={saved}"
                )
        if self._normalized_paths(assembled.source_files) != (
            self._normalized_paths(combo.input_source_files)
        ):
            raise RuntimeError(
                "input_source_files differ while rebuilding members; rerun "
                "Stage 14 before Stage 18"
            )
        if not (
            assembled.lon.shape == self.train_lon.shape
            and np.allclose(assembled.lon, self.train_lon, equal_nan=True)
            and np.allclose(assembled.lat, self.train_lat, equal_nan=True)
        ):
            raise RuntimeError(
                "observation coordinates differ from the training mascons"
            )
        for fold, scaler in enumerate(self.ensemble.scalers_x, start=1):
            expected = getattr(
                scaler, "n_features_in_", assembled.values.shape[1]
            )
            if expected != assembled.values.shape[1]:
                raise RuntimeError(
                    f"fold {fold} expects {expected} features, but the exact "
                    f"Stage-14 inputs provide {assembled.values.shape[1]}"
                )

        members_full = self.ensemble.predict_all_members(assembled.values)
        if (
            members_full.shape[0] != self.n_members
            or members_full.shape[1] != assembled.time_month.size
            or members_full.shape[2] != int(self.mask.sum())
        ):
            raise RuntimeError(
                "recomputed member predictions have an unexpected shape: "
                f"{members_full.shape}"
            )
        edge = slice(EDGE_MONTHS, -EDGE_MONTHS) if EDGE_MONTHS else slice(None)
        members_trimmed = members_full[:, edge]
        rebuilt_months = np.asarray(assembled.time_month)[edge].astype(
            "datetime64[M]"
        )
        if not np.array_equal(rebuilt_months, combo_months):
            raise RuntimeError(
                "recomputed member month axis differs from the saved Stage-14 "
                "reconstruction"
            )
        if combo.pred.shape[1:] != (combo.sigma2.size, combo.lat.size):
            raise RuntimeError("saved reconstruction grid dimensions disagree")
        if self.mask.shape != (combo.sigma2.size * combo.lat.size,):
            raise RuntimeError(
                f"Psi_mask shape {self.mask.shape} differs from the saved grid"
            )
        saved_flat = combo.pred.reshape(combo.pred.shape[0], -1)[:, self.mask]
        if not np.allclose(
            members_trimmed.mean(axis=0),
            saved_flat,
            atol=1e-3,
            rtol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                "recomputed member mean differs from the saved Stage-14 "
                "reconstruction"
            )
        return members_trimmed[:, combo_take]


_member_reconstructor = None


def _exact_member_reconstructor() -> _ExactMemberReconstructor:
    global _member_reconstructor
    if _member_reconstructor is None:
        _member_reconstructor = _ExactMemberReconstructor()
        print(
            "  loaded the trained ensemble to recompute member predictions"
        )
    return _member_reconstructor


# %% [2] One trend per input combination ---------------------------------------
display_tags, slopes, intervals, sig_point, sig_fdr = [], [], [], [], []
intervals_no_sate, sig_point_no_sate, sig_fdr_no_sate = [], [], []
sigma_total, sigma_total_no_sate, sigma_serial = [], [], []
sigma_eps_per_combo, sigma_grace_per_combo = [], []
member_prediction_sources = []
member_prediction_recomputed = []
member_counts = []
trimmed, testables, testables_no_sate = [], [], []
grace_priced_per_combo = []
source_grace_files = []
measurement_noise_proxy_per_combo = []
measurement_noise_definition_per_combo = []
_log = io.StringIO()

for _obp_source, _ssh_source, _wind_source, tag in COMBINATION_REGISTRY:
    stem = f"Pred_RealWorld{tag}"
    with redirect_stdout(_log):          # scenario-RMSE stub warnings
        combo = load_real_world(NN_DIR, rmse_scenario=None,
                                edge_months=EDGE_MONTHS, file_stem=stem)
    _combo_provenance = {
        "run_id": combo.run_id or "unknown",
        "training_source_run_id": combo.training_source_run_id or "unknown",
        "cmip_dataset_id": combo.cmip_dataset_id or "unknown",
        "satellite_dataset_id": combo.satellite_dataset_id or "unknown",
        "training_experiment": combo.training_experiment or "unknown",
    }
    if _combo_provenance != _budget_provenance:
        raise SystemExit(
            f"{stem} provenance differs from {BUDGET_FILE.name}"
        )
    if combo.time_month is None:
        raise SystemExit(f"{stem} has no calendar month coordinate")
    months = np.asarray(combo.time_month).astype("datetime64[M]")
    common, (_combo_take, _budget_take) = exact_common_month_indices(
        (months, reference_months), labels=(stem, BUDGET_FILE.name)
    )
    if not np.array_equal(common, reference_months):
        raise SystemExit(
            f"{stem} does not contain the exact Stage-15 analysis interval"
        )
    pred = combo.pred[_combo_take]
    t_years = np.asarray(combo.t_years)[_combo_take]
    trimmed.append(int(months.size - reference_months.size))

    # Estimate member-trend spread separately for each product tuple.
    raw_combo = load_npz_or_mat(OUT_DIR / stem, ["pred_yz_members"])
    xc = t_years - t_years.mean()
    weights = xc / np.sum(xc**2)
    if "pred_yz_members" in raw_combo:
        members_full = trim_and_validate_member_predictions(
            raw_combo["pred_yz_members"], combo.pred, EDGE_MONTHS
        )
        members = members_full[:, _combo_take]
        if not np.allclose(
            members.mean(axis=0), pred, atol=1e-3, rtol=0.0, equal_nan=True
        ):
            raise RuntimeError(f"{stem} aligned member mean differs from pred")
        member_trends = np.einsum("t,ktij->kij", weights, members)
        combo_sigma_eps = member_trends.std(axis=0, ddof=1)
        member_source = "saved_pred_yz_members"
        n_combo_members = members.shape[0]
    else:
        print(
            f"  {tag or '(default)'}: Stage 14 saves member predictions "
            "for the default product only; recomputing them"
        )
        reconstructor = _exact_member_reconstructor()
        flat_members = reconstructor.predict_aligned(
            combo,
            _obp_source,
            _ssh_source,
            _wind_source,
            months,
            _combo_take,
        )
        pred_flat = pred.reshape(pred.shape[0], -1)[
            :, reconstructor.mask
        ]
        if not np.allclose(
            flat_members.mean(axis=0),
            pred_flat,
            atol=1e-3,
            rtol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"{stem} recomputed aligned member mean differs from pred"
            )
        member_trends_flat = np.einsum("t,ktp->kp", weights, flat_members)
        sigma_flat = member_trends_flat.std(axis=0, ddof=1)
        combo_sigma_eps = np.zeros(combo.sigma2.size * combo.lat.size)
        combo_sigma_eps[reconstructor.mask] = sigma_flat
        combo_sigma_eps = combo_sigma_eps.reshape(
            combo.sigma2.size, combo.lat.size
        )
        member_source = "recomputed_exact_stage14_pipeline"
        n_combo_members = flat_members.shape[0]
    member_prediction_sources.append(member_source)
    member_prediction_recomputed.append(
        member_source == "recomputed_exact_stage14_pipeline"
    )
    member_counts.append(n_combo_members)
    if n_combo_members < 2:
        raise RuntimeError(f"{stem} has fewer than two ensemble members")
    sigma_eps_per_combo.append(combo_sigma_eps)

    # False means this exact tuple has no Stage-16 uncertainty estimate.
    grace_budget = grace_budgets_by_tag.get(tag)
    combo_grace_priced = grace_budget is not None
    combo_sigma_grace = (
        grace_budget["sigma_grace"]
        if combo_grace_priced
        else np.zeros_like(sigma_sate)
    )
    grace_priced_per_combo.append(combo_grace_priced)
    sigma_grace_per_combo.append(combo_sigma_grace)
    source_grace_files.append(
        grace_budget["file"] if combo_grace_priced else ""
    )
    measurement_noise_proxy_per_combo.append(
        grace_budget["is_proxy"] if combo_grace_priced else False
    )
    measurement_noise_definition_per_combo.append(
        grace_budget["definition"] if combo_grace_priced else ""
    )

    trend = robust_trend(pred, t_years,
                         method=TREND_METHOD,
                         block_months=TREND_BLOCK_MONTHS,
                         n_boot=N_BOOT_TREND, seed=TREND_SEED,
                         sigma_map=sigma_map, sigma_sate=sigma_sate,
                         sigma_eps=combo_sigma_eps,
                         sigma_grace=combo_sigma_grace)
    no_sate_total = np.sqrt(
        trend.sigma_serial**2 + combo_sigma_eps**2 + sigma_map**2
        + combo_sigma_grace**2
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        no_sate_z = trend.slope_mean / no_sate_total
    no_sate_p = 2.0 * ndtr(-np.abs(no_sate_z))
    no_sate_interval = np.stack([
        trend.slope_mean - N_SIGMA * no_sate_total,
        trend.slope_mean + N_SIGMA * no_sate_total,
    ])
    trend_no_sate = replace(
        trend,
        slope_interval_2sigma=no_sate_interval,
        slope_pval=no_sate_p,
        sigma_sate=np.zeros_like(sigma_sate),
        sigma_total=no_sate_total,
    )
    display_tags.append(tag or "(default)")
    slopes.append(trend.slope_mean)
    intervals.append(trend.slope_interval_2sigma)
    intervals_no_sate.append(no_sate_interval)
    sigma_total.append(trend.sigma_total)
    sigma_total_no_sate.append(no_sate_total)
    sigma_serial.append(trend.sigma_serial)
    # Exclude structural-zero cells from multiple-testing correction.
    testable = (np.isfinite(trend.slope_pval)
                & (pred.std(axis=0, ddof=0) > 0))
    testables.append(testable)
    testable_no_sate = (np.isfinite(no_sate_p)
                        & (pred.std(axis=0, ddof=0) > 0))
    testables_no_sate.append(testable_no_sate)
    sig_point.append(trend.is_significant(N_SIGMA))
    sig_fdr.append(trend.is_significant_fdr(
        ALPHA_FDR, n_sigma=N_SIGMA, test_mask=testable))
    sig_point_no_sate.append(trend_no_sate.is_significant(N_SIGMA))
    sig_fdr_no_sate.append(trend_no_sate.is_significant_fdr(
        ALPHA_FDR, n_sigma=N_SIGMA, test_mask=testable_no_sate))
    _valid = np.isfinite(trend.slope_mean)
    print(f"  {display_tags[-1]:38s} significant {sig_point[-1][_valid].mean():6.1%} "
          f"per-point -> {sig_fdr[-1][_valid].mean():6.1%} after FDR"
          + (f"  ({trimmed[-1]} later months trimmed)" if trimmed[-1] else ""))

slopes = np.stack(slopes)
intervals = np.stack(intervals)
intervals_no_sate = np.stack(intervals_no_sate)
sigma_total = np.stack(sigma_total)
sigma_total_no_sate = np.stack(sigma_total_no_sate)
sigma_serial = np.stack(sigma_serial)
sig_point = np.stack(sig_point)
sig_fdr = np.stack(sig_fdr)
testables = np.stack(testables)
sig_point_no_sate = np.stack(sig_point_no_sate)
sig_fdr_no_sate = np.stack(sig_fdr_no_sate)
testables_no_sate = np.stack(testables_no_sate)
sigma_eps_per_combo = np.stack(sigma_eps_per_combo)
sigma_grace_per_combo = np.stack(sigma_grace_per_combo)

if not np.allclose(
    sigma_eps_per_combo[0], sigma_eps, atol=1e-12, rtol=0.0,
    equal_nan=True,
):
    raise SystemExit(
        "combination 0 member-trend sigma_eps does not reproduce the "
        "Stage-15 budget"
    )

# %% [3] Check: combination 0 reproduces the Stage-15/16 trend statistics ----
if REFERENCE_FILE.is_file():
    with np.load(REFERENCE_FILE, allow_pickle=False) as _e:
        _reference_required = {
            "slope_mean", "significant", "significant_fdr",
            "testable", "time_month_int", "sigma_grace",
        }
        _reference_missing = sorted(_reference_required - set(_e.files))
        if _reference_missing:
            raise SystemExit(
                f"{REFERENCE_FILE.name} is missing {_reference_missing}; "
                "rerun Stage 15/16"
            )
        ref_slope = np.asarray(_e["slope_mean"], dtype=float)
        _reference_sig = {
            "significant": np.asarray(_e["significant"], dtype=bool),
            "significant_fdr": np.asarray(
                _e["significant_fdr"], dtype=bool
            ),
        }
        _reference_testable = np.asarray(_e["testable"], dtype=bool)
        _reference_sigma_grace = np.asarray(
            _e["sigma_grace"], dtype=float
        )
        _reference_months = np.asarray(
            _e["time_month_int"], dtype=np.int64
        ).reshape(-1)
    if not np.array_equal(
        _reference_months,
        reference_months.astype("datetime64[M]").astype(np.int64),
    ):
        raise SystemExit(
            f"{REFERENCE_FILE.name} months differ from {BUDGET_FILE.name}"
        )
    if not np.allclose(slopes[0], ref_slope, atol=1e-9, equal_nan=True):
        raise SystemExit(
            "combination 0 does not reproduce the Stage-15 slope_mean")
    if not np.allclose(
        sigma_grace_per_combo[0], _reference_sigma_grace,
        atol=0.0, rtol=0.0, equal_nan=True,
    ):
        raise SystemExit(
            "combination 0 does not reproduce the Stage-16 sigma_grace"
        )
    if not np.array_equal(_reference_testable, testables[0]):
        raise SystemExit(
            "combination 0 does not reproduce the tested domain"
        )
    for _key, _mine in (("significant", sig_point[0]),
                        ("significant_fdr", sig_fdr[0])):
        _ref = _reference_sig[_key]
        _testable = _reference_testable
        if not np.array_equal(_ref[_testable], _mine[_testable]):
            raise SystemExit(
                f"combination 0 does not reproduce {_key}")
    print("combination 0 reproduces the Stage-15/16 trend statistics")
else:
    print(f"WARNING: {REFERENCE_FILE.name} absent - skipped the reference check")

# %% [4] Save ------------------------------------------------------------------
np.savez(
    OUT_FILE,
    schema_version=np.asarray(2, dtype=np.int64),
    combo_tags=np.asarray([row[3] for row in COMBINATION_REGISTRY]),
    combo_display_tags=np.asarray(display_tags),
    combo_obp_sources=np.asarray([row[0] for row in COMBINATION_REGISTRY]),
    combo_ssh_sources=np.asarray([row[1] for row in COMBINATION_REGISTRY]),
    combo_wind_sources=np.asarray([row[2] for row in COMBINATION_REGISTRY]),
    n_combos=np.asarray(len(COMBINATION_REGISTRY), dtype=np.int64),
    combo_index_formula=np.asarray("obp*4 + ssh*2 + wind"),
    slope_per_year=slopes.astype(np.float32),
    slope_interval_2sigma=intervals,
    slope_interval_2sigma_no_sate=intervals_no_sate,
    sigma_total=sigma_total,
    sigma_total_no_sate=sigma_total_no_sate,
    sigma_serial=sigma_serial,
    sigma_eps_per_combo=sigma_eps_per_combo,
    sigma_eps_definition=np.asarray(
        "sample SD (ddof=1) across exact fold/member OLS trend maps on the "
        "Stage-15 common month axis"
    ),
    member_prediction_source_per_combo=np.asarray(
        member_prediction_sources
    ),
    member_prediction_recomputed_per_combo=np.asarray(
        member_prediction_recomputed, dtype=np.bool_
    ),
    n_prediction_members_per_combo=np.asarray(
        member_counts, dtype=np.int64
    ),
    sigma_grace_per_combo=sigma_grace_per_combo,
    significant=sig_point,
    significant_fdr=sig_fdr,
    testable=testables,
    significant_no_sate=sig_point_no_sate,
    significant_fdr_no_sate=sig_fdr_no_sate,
    testable_no_sate=testables_no_sate,
    time_month_int=reference_months.astype(np.int64),
    months_trimmed=np.asarray(trimmed),
    trend_method=np.asarray(TREND_METHOD),
    block_months=np.asarray(TREND_BLOCK_MONTHS),
    n_boot=np.asarray(N_BOOT_TREND),
    seed=np.asarray(TREND_SEED),
    alpha_fdr=np.asarray(ALPHA_FDR),
    n_sigma=np.asarray(N_SIGMA),
    edge_months=np.asarray(EDGE_MONTHS, dtype=np.int64),
    sigma_sate_included=np.asarray(True),
    no_sate_definition=np.asarray(
        "conditional trend uncertainty excludes sigma_sate only"
    ),
    product_spread_include_gsfc=np.asarray(
        product_spread_include_gsfc, dtype=np.bool_
    ),
    product_spread_definition=np.asarray(product_spread_definition),
    source_budget_file=np.asarray(BUDGET_FILE.name),
    source_grace_file=np.asarray(
        grace_budgets_by_tag.get("", {}).get("file", "")
    ),
    grace_priced=np.asarray(
        "" in grace_budgets_by_tag
    ),
    source_grace_files=np.asarray(source_grace_files),
    measurement_noise_is_proxy_per_combo=np.asarray(
        measurement_noise_proxy_per_combo, dtype=np.bool_
    ),
    measurement_noise_definition_per_combo=np.asarray(
        measurement_noise_definition_per_combo
    ),
    grace_explicitly_omitted=np.asarray(_args.without_grace),
    grace_priced_per_combo=np.asarray(
        grace_priced_per_combo, dtype=np.bool_
    ),
    run_id=np.asarray(budget_run_id),
    training_source_run_id=np.asarray(
        _budget_provenance["training_source_run_id"]
    ),
    cmip_dataset_id=np.asarray(_budget_provenance["cmip_dataset_id"]),
    satellite_dataset_id=np.asarray(
        _budget_provenance["satellite_dataset_id"]
    ),
    training_experiment=np.asarray(
        _budget_provenance["training_experiment"]
    ),
    covariates=np.asarray(COVARIATES),
)
print("saved:", OUT_FILE)
