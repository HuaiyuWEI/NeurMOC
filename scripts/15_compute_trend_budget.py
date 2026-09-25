"""Stage 15: estimate monthly and trend uncertainty components.

Combine cross-model mapping errors, satellite-product differences, network
spread, and serial-sampling uncertainty using the configured estimators.
GRACE input-error propagation is calculated separately in Stage 16.
Run with --help for estimator and output options.
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import ACTIVE_RUN_ID
from neurmoc.evaluation import (
    BASELINE_EXPERIMENT,
    COVARIATES_ALL,
    DEFAULT_SIGMA_MAP_ESTIMATOR,
    DEFAULT_SIGMA_MAP_MONTHLY_CENTERING,
    PERF_ROOT,
    REALWORLD_ROOT,
    SIGMA_MAP_ESTIMATOR_CHOICES,
    SIGMA_MAP_MONTHLY_CENTERING_CHOICES,
    model_dir,
    sigma_map_monthly_centering_label,
    sigma_map_monthly_expected_ddof,
    sigma_map_monthly_retains_mean_shifts,
    sigma_map_transfer_cases,
)
from neurmoc.export import save_realworld_trend_npz
from neurmoc.io_utils import load_npz_or_mat
from neurmoc.moc_utils import (
    fill_down_columns,
    robust_trend,
    sliding_window_trends,
)
from neurmoc.satellite_products import (
    DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC,
    product_spread_definition,
    transfer_case_spec_json,
    uncertainty_product_registry,
)
from neurmoc.results import (
    branch_centered_mapping_spread,
    exact_common_month_indices,
    grand_centered_mapping_spread,
    load_real_world,
    trim_and_validate_member_predictions,
    validate_transfer_case_arrays,
)

# Editable default; the command-line option takes precedence.
PRODUCT_SPREAD_INCLUDE_GSFC = DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC

_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--step-months", type=int, default=None,
                  help="window-start stride within the held-out members "
                       "(default: the window length itself, i.e. "
                       "NON-OVERLAPPING within each member; "
                       "e.g. 12 for the dense sliding-window variant)")
_cli.add_argument("--max-start-year", type=float, default=2017.0,
                  help="latest calendar year a transfer window may START. "
                       "Default 2017: with the non-overlapping stride "
                       "this normally keeps only the FIRST observational-"
                       "length window of each member, beginning in early "
                       "2016 (the closest climate-state analog of "
                       "the observational era); 9999 = all three "
                       "non-overlapping windows to 2100")
_cli.add_argument("-x", "--covariates", default=None,
                  help="comma- or plus-separated input set selecting "
                       "the trained ablation network (default: all "
                       "three inputs, or NEURMOC_FIG_COVARIATES)")
_cli.add_argument("-E", "--experiment", default=None,
                  help="trained experiment folder (default: "
                       "NEURMOC_FIG_EXPERIMENT, then the profile baseline)")
_cli.add_argument("--mapping-error-estimator", "--sigma-map-estimator",
                  dest="sigma_map_estimator",
                  choices=SIGMA_MAP_ESTIMATOR_CHOICES,
                  default=DEFAULT_SIGMA_MAP_ESTIMATOR,
                  help="selects BOTH monthly- and trend-space maps; "
                       "mri_ssp245: five MRI SSP245 branch windows; "
                       "mri_ssp126_245_370_stacked: grand-centered SD of "
                       "15 correlated MRI scenario branches")
_cli.add_argument("--sigma-map-monthly-centering",
                  dest="sigma_map_monthly_centering",
                  choices=SIGMA_MAP_MONTHLY_CENTERING_CHOICES,
                  default=DEFAULT_SIGMA_MAP_MONTHLY_CENTERING,
                  help="MONTHLY sigma_map only; the trend-space map is "
                       "always grand-centered because each branch supplies "
                       "a single trend error. branch (default): remove each "
                       "branch's own record mean, matching the ZERO_BIAS "
                       "display and the offset-blind trend. grand: "
                       "one mean across the whole pool")
_cli.add_argument("--out-tag", default="",
                  help="suffix for sensitivity-test output so the default "
                       "budget is not overwritten")
_cli.add_argument(
    "--include-gsfc-in-product-spread",
    dest="product_spread_include_gsfc",
    action=argparse.BooleanOptionalAction,
    default=PRODUCT_SPREAD_INCLUDE_GSFC,
    help=(
        "include GSFC in the one shared monthly/trend product-spread "
        "registry (default from the scientific profile)"
    ),
)
_args = _cli.parse_known_args()[0]        # tolerate interactive environments
if _args.product_spread_include_gsfc and not _args.out_tag:
    # Keep the default JPL+CSR budget separate from the GSFC sensitivity test.
    _args.out_tag = "with_gsfc"
if (_args.out_tag.startswith("_") or "/" in _args.out_tag
        or "\\" in _args.out_tag):
    raise SystemExit(
        "--out-tag must be filename-safe without a leading underscore or "
        "path separators"
    )
EDGE_MONTHS = 12                 # per-member low-pass-filter edge trim
# L=0 denotes independent residual-month resampling (effective block length 1).
SERIAL_BLOCK_MONTHS = np.asarray([0, 6, 12, 24, 48, 72], dtype=np.int64)
SERIAL_ZERO_MONTH_DEFINITION = (
    "no_temporal_blocking; independent residual-month resampling; "
    "effective_block_months=1"
)
# None uses non-overlapping held-out windows.
STEP_MONTHS = _args.step_months
SIGMA_MAP_ESTIMATOR = _args.sigma_map_estimator
# Centering rule for monthly mapping errors.
SIGMA_MAP_MONTHLY_CENTERING = _args.sigma_map_monthly_centering
PRODUCT_SPREAD_INCLUDE_GSFC = bool(
    _args.product_spread_include_gsfc
)
PRODUCT_SPREAD_DEFINITION = product_spread_definition(
    PRODUCT_SPREAD_INCLUDE_GSFC
)
SIGMA_MAP_ESTIMATOR_CODES = {
    "mri_ssp245": 1,
    "mri_ssp126_245_370_stacked": 2,
}
SIGMA_MAP_ESTIMATOR_CODE = SIGMA_MAP_ESTIMATOR_CODES[SIGMA_MAP_ESTIMATOR]
# Held-out MRI cases used for mapping uncertainty: tag -> (members, year0).
TRANSFER_CASES = sigma_map_transfer_cases(SIGMA_MAP_ESTIMATOR)
# Save the case registry with the budget for reproducibility.
TRANSFER_CASE_SPEC_JSON = transfer_case_spec_json(TRANSFER_CASES)
# Limit transfer windows by start year; scenario branches share member lineages.
MAX_WINDOW_START_YEAR = _args.max_start_year
# Regions are diagnostic only; mapping uncertainty is estimated per cell.
SMOC_LAT_BOUNDARY = -34.0        # lat south of this -> Southern Ocean
ABYSSAL_SIGMA2 = 36.7            # sigma2 denser than this -> abyssal
# MRI-ESM2.0 scenario cases used for diagnostics.
SCENARIO_DIAGNOSTIC_CASES = {
    tag: count for tag, (count, _year0) in
    sigma_map_transfer_cases("mri_ssp126_245_370_stacked").items()
}
# Select input set from the CLI, environment, or all-input default.
COVARIATES = (_args.covariates
              or os.environ.get("NEURMOC_FIG_COVARIATES")
              or COVARIATES_ALL).replace(",", "+")
EXPERIMENT = (_args.experiment
              or os.environ.get("NEURMOC_FIG_EXPERIMENT")
              or BASELINE_EXPERIMENT)
NN_DIR = model_dir(root=REALWORLD_ROOT, experiment=EXPERIMENT,
                   covariates=COVARIATES)
SOURCE_NN_DIR = model_dir(root=PERF_ROOT, experiment=EXPERIMENT,
                          covariates=COVARIATES)

# Vary only products for inputs used by the selected network.
COMBINATION_REGISTRY = uncertainty_product_registry(
    COVARIATES, include_gsfc=PRODUCT_SPREAD_INCLUDE_GSFC
)
COMBO_OBP_SOURCES = np.asarray(
    [row[0] for row in COMBINATION_REGISTRY], dtype="U"
)
COMBO_SSH_SOURCES = np.asarray(
    [row[1] for row in COMBINATION_REGISTRY], dtype="U"
)
COMBO_WIND_SOURCES = np.asarray(
    [row[2] for row in COMBINATION_REGISTRY], dtype="U"
)
COMBO_TAGS = np.asarray([row[3] for row in COMBINATION_REGISTRY], dtype="U")
if not COMBINATION_REGISTRY or COMBINATION_REGISTRY[0][3] != "":
    raise RuntimeError("product registry must begin with the default reconstruction")
_tag = f"_{_args.out_tag}" if _args.out_tag else ""
OUT_FILE = NN_DIR / "RealWorld" / f"trend_error_budget{_tag}.npz"

# %% [1] One exact calendar window shared by every budget term -----------------
# Use the common month interval across all selected satellite products.
combo_reconstructions = []
combo_month_axes = []
_log = io.StringIO()
for _obp, _ssh, _wind, combo_tag in COMBINATION_REGISTRY:
    stem = f"Pred_RealWorld{combo_tag}"
    with redirect_stdout(_log):
        combo = load_real_world(
            NN_DIR, rmse_scenario=None, edge_months=EDGE_MONTHS,
            file_stem=stem,
        )
    if combo.time_month is None:
        raise RuntimeError(f"{stem} has no calendar month coordinate")
    combo_reconstructions.append(combo)
    combo_month_axes.append(np.asarray(combo.time_month).astype("datetime64[M]"))

common, combo_month_indices = exact_common_month_indices(
    combo_month_axes,
    labels=[f"Pred_RealWorld{tag or ' (default)'}" for tag in COMBO_TAGS],
)
rw_full = combo_reconstructions[0]
default_take = combo_month_indices[0]
for combo, combo_tag in zip(combo_reconstructions[1:], COMBO_TAGS[1:]):
    if combo.pred.shape[1:] != rw_full.pred.shape[1:]:
        raise RuntimeError(
            f"Pred_RealWorld{combo_tag} grid shape {combo.pred.shape[1:]} "
            f"!= default {rw_full.pred.shape[1:]}"
        )
    if (
        not np.array_equal(np.asarray(combo.lat), np.asarray(rw_full.lat))
        or not np.array_equal(
            np.asarray(combo.sigma2), np.asarray(rw_full.sigma2)
        )
    ):
        raise RuntimeError(f"Pred_RealWorld{combo_tag} grid coordinates differ")
    for provenance_key in (
        "run_id", "training_source_run_id", "cmip_dataset_id",
        "satellite_dataset_id",
        "training_experiment", "trained_on", "moc_convention",
    ):
        if getattr(combo, provenance_key) != getattr(rw_full, provenance_key):
            raise RuntimeError(
                f"Pred_RealWorld{combo_tag} {provenance_key} differs from "
                "the default reconstruction"
            )
if rw_full.training_source_run_id != ACTIVE_RUN_ID:
    raise RuntimeError(
        "Pred_RealWorld training source differs from the active profile: "
        f"{rw_full.training_source_run_id!r} != {ACTIVE_RUN_ID!r}"
    )

aligned = np.stack([
    combo.pred[take]
    for combo, take in zip(combo_reconstructions, combo_month_indices)
])
rw = replace(
    rw_full,
    pred=rw_full.pred[default_take],
    epistemic=rw_full.epistemic[default_take],
    total_uncertainty=rw_full.total_uncertainty[default_take],
    t_years=rw_full.t_years[default_take],
    time_month=common,
)
window = rw.pred.shape[0]
n_lev, n_lat = rw.pred.shape[1:]
if window < 3:
    raise RuntimeError(
        f"product combinations share only {window} months; at least 3 are required"
    )
if STEP_MONTHS is None:
    STEP_MONTHS = window            # non-overlapping (the default)
print(f"common product window: {window} months ({common[0]}..{common[-1]}); "
      f"stride {STEP_MONTHS} months"
      f"{' (non-overlapping)' if STEP_MONTHS >= window else ''}, "
      f"window starts <= {MAX_WINDOW_START_YEAR:g}; "
      f"sigma_map estimator {SIGMA_MAP_ESTIMATOR}; output {OUT_FILE.name}")

# %% [2] Mapping error in the held-out-model windows --------------------------
bt_all, bp_all = [], []          # pooled true / reconstructed window trends
monthly_error_all = []           # [branch window, month, lev, lat]
window_cases, window_members, window_start_years = [], [], []
used_cases = []
for tag, (n_rlz, year0) in TRANSFER_CASES.items():
    path = SOURCE_NN_DIR / f"Pred_{tag}"
    try:
        data = load_npz_or_mat(path, ["y", "y_pred"])
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Selected sigma_map estimator {SIGMA_MAP_ESTIMATOR!r} requires "
            f"{path}, but that Stage-10 evaluation is missing"
        ) from exc
    truth = np.asarray(data["y"])
    pred = np.asarray(data["y_pred"])
    nt_per = validate_transfer_case_arrays(
        truth, pred, n_rlz, (n_lev, n_lat), str(path)
    )
    n_win = n_tot = 0
    for m in range(n_rlz):
        sl = slice(m * nt_per + EDGE_MONTHS, (m + 1) * nt_per - EDGE_MONTHS)
        if sl.stop - sl.start < window:
            continue
        w_starts = np.arange(0, sl.stop - sl.start - window + 1, STEP_MONTHS)
        w_years = year0 + (EDGE_MONTHS + w_starts + 1) / 12.0
        keep_w = (np.ones(w_starts.size, bool)
                  if MAX_WINDOW_START_YEAR is None
                  else w_years <= MAX_WINDOW_START_YEAR)
        n_tot += w_starts.size
        if not keep_w.any():
            continue
        truth_member = truth[sl]
        pred_member = pred[sl]
        bt_case = sliding_window_trends(
            truth_member, window, STEP_MONTHS)[keep_w]
        bp_case = sliding_window_trends(
            pred_member, window, STEP_MONTHS)[keep_w]
        kept_starts = w_starts[keep_w]
        monthly_error_all.append(np.stack([
            pred_member[start:start + window]
            - truth_member[start:start + window]
            for start in kept_starts
        ]))
        bt_all.append(bt_case)
        bp_all.append(bp_case)
        n_kept = bt_case.shape[0]
        window_cases.extend([tag] * n_kept)
        window_members.extend([m + 1] * n_kept)
        window_start_years.extend(w_years[keep_w].tolist())
        n_win += n_kept
    if n_win == 0:
        raise SystemExit(
            f"Selected sigma_map case {tag} produced no windows at the "
            f"requested epoch cutoff {MAX_WINDOW_START_YEAR}"
        )
    used_cases.append(tag)
    print(f"  {tag}: {n_rlz} members x {nt_per} months -> {n_win} windows "
          f"kept of {n_tot} (starts <= {MAX_WINDOW_START_YEAR})")

if not bt_all:
    raise SystemExit(
        f"No transfer windows found in {SOURCE_NN_DIR}; evaluate at least one of "
        f"{list(TRANSFER_CASES)} before computing the budget")
bt = np.concatenate(bt_all)      # [N_win, lev, lat]
bp = np.concatenate(bp_all)
n_windows = bt.shape[0]
monthly_error = np.concatenate(monthly_error_all)
del bt_all, bp_all, monthly_error_all
if monthly_error.shape[:2] != (n_windows, window):
    raise RuntimeError(
        "monthly mapping-error pool is not aligned with the trend windows: "
        f"{monthly_error.shape[:2]} != {(n_windows, window)}"
    )

# %% [2b] Forcing-sensitivity DIAGNOSTIC (never enters the budget) -------------
# Estimate mapping errors separately by scenario for diagnostics only.
scen_tags, scen_n_members = [], []
scen_sigma, scen_bias, scen_truth = [], [], []
scen_sigma_monthly, scen_bias_monthly = [], []
for tag, n_rlz in SCENARIO_DIAGNOSTIC_CASES.items():
    try:
        d = load_npz_or_mat(SOURCE_NN_DIR / f"Pred_{tag}", ["y", "y_pred"])
    except FileNotFoundError:
        continue
    tr_s, pr_s = np.asarray(d["y"], float), np.asarray(d["y_pred"], float)
    nt_s = validate_transfer_case_arrays(
        tr_s, pr_s, n_rlz, (n_lev, n_lat),
        str(SOURCE_NN_DIR / f"Pred_{tag}"))
    if nt_s - 2 * EDGE_MONTHS < window:
        continue
    xs = np.arange(window) / 12.0
    a_s = (xs - xs.mean()) / ((xs - xs.mean()) ** 2).sum()
    db_s, bt_s, em_s = [], [], []
    for m in range(n_rlz):                       # first window per member
        s0 = m * nt_s + EDGE_MONTHS
        t_w = np.einsum("t,tij->ij", a_s, tr_s[s0:s0 + window])
        p_w = np.einsum("t,tij->ij", a_s, pr_s[s0:s0 + window])
        db_s.append(p_w - t_w)
        bt_s.append(t_w)
        em_s.append(pr_s[s0:s0 + window] - tr_s[s0:s0 + window])
    db_s, bt_s = np.stack(db_s), np.stack(bt_s)
    em_s = np.stack(em_s)
    scen_tags.append(tag)
    scen_n_members.append(db_s.shape[0])
    scen_sigma.append(db_s.std(axis=0, ddof=1))
    scen_bias.append(db_s.mean(axis=0))
    scen_truth.append(bt_s.mean(axis=0))
    # Use the budget's monthly centering and finite-data support rule.
    _scen_spread_fn = (
        branch_centered_mapping_spread
        if SIGMA_MAP_MONTHLY_CENTERING == "branch"
        else grand_centered_mapping_spread
    )
    _scen_spread, _scen_mean, _, _ = _scen_spread_fn(em_s, sample_ndim=2)
    scen_sigma_monthly.append(_scen_spread)
    scen_bias_monthly.append(_scen_mean)
if scen_tags:
    print("forcing-sensitivity diagnostic (scenario-specific maps; selected "
          "cases may also enter sigma_map), median over cells:")
    print(f"  {'scenario':<12} {'sigma_map':>10} {'|bias|':>9} {'true trend':>11}")
    for k, tag in enumerate(scen_tags):
        print(f"  {tag:<12} {np.nanmedian(scen_sigma[k]):>10.4f} "
              f"{np.nanmedian(np.abs(scen_bias[k])):>9.4f} "
              f"{np.nanmedian(scen_truth[k]):>11.4f}")

# Per-cell attenuation regression; unavailable target cells remain NaN.
ok = np.isfinite(bt) & np.isfinite(bp)
cnt = ok.sum(axis=0).astype(float)
with np.errstate(invalid="ignore", divide="ignore"):
    mt = np.nansum(np.where(ok, bt, 0), axis=0) / cnt
    mp = np.nansum(np.where(ok, bp, 0), axis=0) / cnt
    cov = np.nansum(np.where(ok, (bt - mt) * (bp - mp), 0), axis=0) / cnt
    var = np.nansum(np.where(ok, (bt - mt) ** 2, 0), axis=0) / cnt
    gamma = cov / var                      # attenuation - DIAGNOSTIC only
# Mapping uncertainty uses raw trend errors, not regression residuals.
db = np.where(ok, bp - bt, np.nan)
(sigma_map_cell, delta_beta_mean,
 trend_nobs, map_unpriced) = grand_centered_mapping_spread(db)
if not np.array_equal(trend_nobs, cnt.astype(np.int64)):
    raise RuntimeError("trend mapping-error support counts are inconsistent")
# Cells without complete held-out-model support are flagged and filled from
# the nearest priced density level above them for budget calculations.
gamma[map_unpriced] = np.nan

# Monthly and trend mapping errors use the same windows but may use different
# centering rules. Monthly branch centering removes each window's mean offset;
# the monthly values are serially correlated, not independent replicates.
monthly_expected_nobs = n_windows * window
_monthly_spread_fn = (
    branch_centered_mapping_spread
    if SIGMA_MAP_MONTHLY_CENTERING == "branch"
    else grand_centered_mapping_spread
)
(sigma_map_monthly_cell, delta_monthly_mean,
 monthly_nobs, monthly_map_unpriced) = _monthly_spread_fn(
    monthly_error, sample_ndim=2
)
# Degrees of freedom for the selected monthly centering rule.
monthly_ddof = sigma_map_monthly_expected_ddof(
    SIGMA_MAP_MONTHLY_CENTERING, n_windows)
_monthly_retains_shifts = sigma_map_monthly_retains_mean_shifts(
    SIGMA_MAP_MONTHLY_CENTERING)
if not np.array_equal(monthly_map_unpriced, map_unpriced):
    raise RuntimeError(
        "monthly and trend mapping-error support masks differ even though "
        "they were built from the same branch windows"
    )

# Mapping errors are estimated per cell; regions are used only for summaries.
smoc = rw.lat < SMOC_LAT_BOUNDARY
aby = rw.sigma2 > ABYSSAL_SIGMA2
region_id = 2 * aby[:, None].astype(int) + smoc[None, :].astype(int)
# Fill unpriced cells from the nearest priced density level above them.
_filled_field = fill_down_columns(sigma_map_cell)
sigma_map = np.nan_to_num(_filled_field, nan=0.0)
_filled = map_unpriced & np.isfinite(_filled_field)
_monthly_filled_field = fill_down_columns(sigma_map_monthly_cell)
sigma_map_monthly = np.nan_to_num(_monthly_filled_field, nan=0.0)
_monthly_filled = monthly_map_unpriced & np.isfinite(_monthly_filled_field)
print(f"sigma_map fill: {int(_filled.sum())} unpriced cells take the "
      f"deepest priced level above them "
      f"({int((map_unpriced & ~np.isfinite(_filled_field)).sum())} "
      "left at zero - no priced level in the column); filled median "
      f"{np.nanmedian(sigma_map[_filled]):.4f} Sv/yr")
print(
    "sigma_map_monthly fill: "
    f"{int(_monthly_filled.sum())} unpriced cells take the deepest priced "
    "level above them "
    f"({int((monthly_map_unpriced & ~np.isfinite(_monthly_filled_field)).sum())} "
    "left at zero); filled median "
    f"{np.nanmedian(sigma_map_monthly[_monthly_filled]):.3f} Sv"
)
print("sigma_map: per-cell std(dbeta); regional medians for reference:")
for rid, rname in [(0, "AMOC upper/mid"), (1, "SMOC upper/mid"),
                   (2, "AMOC abyssal"), (3, "SMOC abyssal")]:
    m = (region_id == rid) & ~map_unpriced
    print(f"  {rname:<16} median {np.nanmedian(sigma_map_cell[m]):.4f} "
          f"Sv/yr ({int(np.isfinite(sigma_map_cell[m]).sum())} cells)")

_lineage_counts = {count for count, _year0 in TRANSFER_CASES.values()}
if len(_lineage_counts) != 1:
    raise RuntimeError(
        "Selected sigma_map cases do not share one member-lineage count")
n_member_lineages = _lineage_counts.pop()
print(f"sigma_map: {n_windows} branch windows from {used_cases}; "
      f"{n_member_lineages} member lineages")
print(
    f"sigma_map_monthly: {SIGMA_MAP_MONTHLY_CENTERING}-centered sample SD of "
    f"{monthly_expected_nobs} serially correlated branch-month values "
    f"(ddof {monthly_ddof}); "
    f"priced-cell median {np.nanmedian(sigma_map_monthly_cell):.3f} Sv"
)
if SIGMA_MAP_MONTHLY_CENTERING == "branch":
    # Report the excluded between-branch spread. Plain statistics propagate
    # NaN at unpriced cells without empty-slice warnings.
    _branch_mean = monthly_error.mean(axis=1)
    _between = _branch_mean.std(axis=0, ddof=1)
    _between[monthly_map_unpriced] = np.nan
    print(
        "  excluded from sigma_map_monthly: between-branch spread of the "
        f"record-mean error, priced-cell median {np.nanmedian(_between):.3f}"
        f" Sv; grand signed bias median {np.nanmedian(delta_monthly_mean):+.3f}"
        " Sv"
    )
if len(TRANSFER_CASES) > 1:
    # Scenario-mean shifts remain only under grand centering.
    if _monthly_retains_shifts:
        print("  WARNING: scenario branches are correlated; stacked sample SD "
              "includes between-scenario mean-error shifts")
    sigma_map_dependence = (
        f"{n_member_lineages} MRI member lineages; sibling scenario branches "
        "are correlated"
    )
else:
    sigma_map_dependence = (
        f"{n_member_lineages} distinct MRI member lineages; no "
        "scenario-branch stacking"
    )
sigma_map_monthly_dependence = (
    f"{window} serially correlated months per branch; "
    f"{sigma_map_dependence}"
)
print(f"  attenuation gamma: median {np.nanmedian(gamma):.2f} "
      f"(IQR {np.nanpercentile(gamma, 25):.2f}.."
      f"{np.nanpercentile(gamma, 75):.2f})")
print("  per-cell spread:   median "
      f"{np.nanmedian(sigma_map_cell[~map_unpriced]):.4f} Sv/yr "
      "(= sigma_map over priced cells)")
print(f"  mean trend error:  median {np.nanmedian(np.abs(delta_beta_mean)):.4f}"
      " Sv/yr (|mean|; directional diagnostic)")

# %% [3] sigma_sate: trend spread across the product combinations --------------
t = np.asarray(rw.t_years, dtype=float)
xc = t - t.mean()
a = xc / (xc**2).sum()
trends = np.einsum("t,ctij->cij", a, aligned)   # [combination, lev, lat]
sigma_sate = trends.std(axis=0, ddof=1)
_valid_plane = np.isfinite(sigma_sate)
frac_unpriced = float((map_unpriced & _valid_plane).sum()
                      / _valid_plane.sum())
print(f"mapping term unpriced (deepest-level fill) at {frac_unpriced:.1%} "
      "of valid cells (no held-out MOC truth)")
#: Monthly product-choice spread is calculated separately from trend-space
#: spread and retained as [time, level, latitude]. The month axis allows
#: consumers to verify alignment.
sigma_sate_month = aligned.std(axis=0, ddof=1)          # [T, lev, lat]
sate_month_axis = common.astype("datetime64[M]").astype(np.int64)
_sate_ts = np.array([np.nanmedian(sigma_sate_month[i])
                     for i in range(sigma_sate_month.shape[0])])
print(f"sigma_sate: {aligned.shape[0]} combinations, "
      f"{common.size} common months ({common[0]}..{common[-1]})")
print(f"  trend spread:   median {np.nanmedian(sigma_sate):.4f} Sv/yr")
print(f"  monthly spread: median {np.nanmedian(sigma_sate_month):.3f} Sv "
      f"(time-dependent: {_sate_ts.min():.3f}..{_sate_ts.max():.3f} Sv "
      f"over the record, {_sate_ts.max() / max(_sate_ts.min(), 1e-9):.1f}x)")

# %% [4] sigma_eps: MEASURED across-member spread of the member trends ---------
# Estimate network uncertainty from the spread of member-specific trends.
_pred_raw = load_npz_or_mat(NN_DIR / "RealWorld" / "Pred_RealWorld")
if "pred_yz_members" not in _pred_raw:
    raise SystemExit("Pred_RealWorld has no pred_yz_members - rerun "
                     "14_reconstruct_real_world.py (baseline) first")
members_full = trim_and_validate_member_predictions(
    _pred_raw["pred_yz_members"], rw_full.pred, EDGE_MONTHS
)
members = members_full[:, default_take]
if not np.allclose(
    members.mean(axis=0), rw.pred, atol=1e-3, rtol=0.0, equal_nan=True
):
    raise RuntimeError(
        "calendar-aligned member mean does not match the default reconstruction"
    )
t_rw = np.asarray(rw.t_years, float)
xc_rw = t_rw - t_rw.mean()
a_rw = xc_rw / (xc_rw**2).sum()
member_trends = np.einsum("t,ktij->kij", a_rw, members)
sigma_eps = member_trends.std(axis=0, ddof=1)
print(f"sigma_eps: {members.shape[0]} members -> "
      f"measured median {np.nanmedian(sigma_eps):.4f} Sv/yr")

# %% [5] Effect on significance, and save --------------------------------------
#: Fixed bootstrap settings keep significance comparisons reproducible.
kw = dict(method="mbb", block_months=48, n_boot=1000, seed=0)
single = robust_trend(rw.pred, rw.t_years,
                      sigma_eps=sigma_eps, **kw)
full = robust_trend(rw.pred, rw.t_years,
                    sigma_map=sigma_map, sigma_sate=sigma_sate,
                    sigma_eps=sigma_eps, **kw)
# Exclude structural-zero cells from significance tests.
valid = np.isfinite(full.slope_pval) & (rw.pred.std(axis=0) > 0)
# Independent-month propagation is diagnostic; it omits temporal covariance.
_xc_rw = np.asarray(rw.t_years, float) - np.mean(rw.t_years)
_a_rw = _xc_rw / (_xc_rw**2).sum()
sigma_eps_propagated = np.sqrt(
    (_a_rw[:, None, None] ** 2 * np.asarray(rw.epistemic, float) ** 2).sum(axis=0)
)
# Include the Stage-16 GRACE term when its interval matches.
_grace_file = OUT_FILE.parent / f"grace_noise_budget{_tag}.npz"
sigma_grace = None
if _grace_file.is_file():
    with np.load(_grace_file, allow_pickle=False) as _gn:
        _sg = np.asarray(_gn["sigma_grace"], dtype=float)
        _grace_months = (
            np.asarray(_gn["time_month"]).astype("datetime64[M]").reshape(-1)
            if "time_month" in _gn.files else None
        )
    if _sg.shape != sigma_map.shape:
        print(f"  WARNING: {_grace_file.name} plane {_sg.shape} != "
              f"{sigma_map.shape}; GRACE term ignored here")
    elif _grace_months is None or not np.array_equal(_grace_months, common):
        print(f"  WARNING: {_grace_file.name} does not use the exact "
              f"{common[0]}..{common[-1]} common product interval; "
              "GRACE term ignored until Stage 16 is rerun")
    else:
        sigma_grace = _sg
full_grace = (None if sigma_grace is None else
              robust_trend(rw.pred, rw.t_years,
                           sigma_map=sigma_map, sigma_sate=sigma_sate,
                           sigma_eps=sigma_eps, sigma_grace=sigma_grace,
                           **kw))

serial_block_maps = []
for _block in SERIAL_BLOCK_MONTHS:
    if int(_block) == kw["block_months"]:
        serial_block_maps.append(full.sigma_serial)
    else:
        serial_block_maps.append(
            robust_trend(
                rw.pred, rw.t_years, method="mbb",
                block_months=int(_block), n_boot=kw["n_boot"],
                seed=kw["seed"],
            ).sigma_serial
        )
serial_block_maps = np.stack(serial_block_maps)
print(
    "MBB block sensitivity medians (Sv/yr): "
    + " | ".join(
        ("L=0 independent" if int(_block) == 0 else f"L={int(_block)}")
        + f" {np.nanmedian(_field[valid]):.4f}"
        for _block, _field in zip(SERIAL_BLOCK_MONTHS, serial_block_maps)
    )
)
print("sigma_serial medians (Sv/yr): "
      f"mbb {np.nanmedian(full.sigma_serial[valid]):.4f}")
print(f"  vs independence-propagated: median "
      f"{np.nanmedian(sigma_eps_propagated[valid]):.4f} Sv/yr")
frac_single = single.is_significant()[valid].mean()
frac_full = full.is_significant()[valid].mean()
print("budget medians (Sv/yr): "
      f"serial {np.nanmedian(full.sigma_serial[valid]):.4f} | "
      f"eps {np.nanmedian(full.sigma_eps):.4f} | "
      f"map {np.nanmedian(sigma_map[~map_unpriced]):.4f} (priced) | "
      f"sate {np.nanmedian(sigma_sate):.4f}")

# Significance fractions before and after adding the GRACE term.
frac_full_grace = (None if full_grace is None
                   else float(full_grace.is_significant()[valid].mean()))
print(f"significant fraction: {frac_single:.1%} (serial + ensemble) -> "
      f"{frac_full:.1%} (this stage's terms; sigma_grace added by stage 16)"
      + ("" if frac_full_grace is None else
         f" -> {frac_full_grace:.1%} (all five terms, using the existing "
         f"{_grace_file.name})"))

# Scenario-specific arrays are diagnostics, separate from the budget mapping term.
_scen = ({} if not scen_tags else dict(
    scen_tags=np.asarray(scen_tags, dtype="U"),
    scen_n_members=np.asarray(scen_n_members, dtype=np.int16),
    scen_sigma_map=np.stack(scen_sigma),
    scen_bias=np.stack(scen_bias),
    scen_true_trend=np.stack(scen_truth),
    scen_sigma_map_monthly=np.stack(scen_sigma_monthly),
    scen_bias_monthly=np.stack(scen_bias_monthly),
))

payload = dict(
    **_scen,
    mapping_error_estimator=np.str_(SIGMA_MAP_ESTIMATOR),
    sigma_map_estimator_code=np.int64(SIGMA_MAP_ESTIMATOR_CODE),
    sigma_map_estimator=np.str_(SIGMA_MAP_ESTIMATOR),
    sigma_map_centering=np.str_("grand_across_selected_branch_windows"),
    sigma_map_ddof=np.int64(1),
    sigma_map_dependence=np.str_(sigma_map_dependence),
    sigma_map_includes_between_scenario_mean_shifts=np.bool_(
        len(TRANSFER_CASES) > 1),
    sigma_map_n_branch_windows=np.int64(n_windows),
    sigma_map_n_member_lineages=np.int64(n_member_lineages),
    sigma_map_nobs=trend_nobs.astype(np.int16),
    sigma_map_expected_nobs=np.int64(n_windows),
    sigma_map=sigma_map, sigma_map_cell=sigma_map_cell,
    sigma_map_monthly_estimator=np.str_(SIGMA_MAP_ESTIMATOR),
    sigma_map_monthly_centering=np.str_(
        sigma_map_monthly_centering_label(SIGMA_MAP_MONTHLY_CENTERING)),
    sigma_map_monthly_ddof=np.int64(monthly_ddof),
    sigma_map_monthly_dependence=np.str_(sigma_map_monthly_dependence),
    # Grand centering retains between-branch mean shifts.
    sigma_map_monthly_includes_between_scenario_mean_shifts=np.bool_(
        _monthly_retains_shifts and len(TRANSFER_CASES) > 1),
    sigma_map_monthly_includes_between_member_mean_shifts=np.bool_(
        _monthly_retains_shifts),
    sigma_map_monthly_includes_between_branch_mean_shifts=np.bool_(
        _monthly_retains_shifts),
    sigma_map_monthly_n_branch_windows=np.int64(n_windows),
    sigma_map_monthly_months_per_branch=np.int64(window),
    sigma_map_monthly_nobs=monthly_nobs.astype(np.int32),
    sigma_map_monthly_expected_nobs=np.int64(monthly_expected_nobs),
    sigma_map_monthly=sigma_map_monthly,
    sigma_map_monthly_cell=sigma_map_monthly_cell,
    delta_monthly_mean=delta_monthly_mean,
    region_id=region_id, sigma_sate=sigma_sate,
    map_unpriced=map_unpriced,
    frac_map_unpriced=np.float64(frac_unpriced),
    monthly_map_unpriced=monthly_map_unpriced,
    frac_monthly_map_unpriced=np.float64(frac_unpriced),
    sigma_sate_month=sigma_sate_month,
    sate_month_axis=sate_month_axis,       # datetime64[M] as int64
    sigma_eps=sigma_eps,
    sigma_serial_mbb=full.sigma_serial,
    sigma_serial_block=serial_block_maps,
    serial_primary_block_months=np.int64(kw["block_months"]),
    serial_block_months=SERIAL_BLOCK_MONTHS,
    serial_zero_month_definition=np.str_(SERIAL_ZERO_MONTH_DEFINITION),
    serial_n_boot=np.int64(kw["n_boot"]),
    serial_seed=np.int64(kw["seed"]),
    lat=np.asarray(rw.lat), sigma2=np.asarray(rw.sigma2),
    t_years=np.asarray(rw.t_years),
    time_month_int=(
        np.empty(0, dtype=np.int64)
        if rw.time_month is None else
        np.asarray(rw.time_month).astype("datetime64[M]").astype(np.int64)
    ),
    run_id=np.str_(rw.run_id or "unknown"),
    training_source_run_id=np.str_(ACTIVE_RUN_ID),
    cmip_dataset_id=np.str_(rw.cmip_dataset_id or "unknown"),
    satellite_dataset_id=np.str_(rw.satellite_dataset_id or "unknown"),
    training_experiment=np.str_(EXPERIMENT),
    covariates=np.str_(COVARIATES),
    med_sate_month=np.float64(np.nanmedian(sigma_sate_month)), gamma=gamma,
    delta_beta_mean=delta_beta_mean, n_windows=np.int64(n_windows),
    transfer_true_trends=bt, transfer_pred_trends=bp,
    transfer_window_case=np.asarray(window_cases, dtype="U"),
    transfer_window_member=np.asarray(window_members, dtype=np.int16),
    transfer_window_start_year=np.asarray(window_start_years, dtype=float),
    window_months=np.int64(window), step_months=np.int64(STEP_MONTHS),
    edge_months=np.int64(EDGE_MONTHS),
    max_start_year=np.float64(MAX_WINDOW_START_YEAR),
    cases=np.array(used_cases), n_combos=np.int64(aligned.shape[0]),
    product_spread_include_gsfc=np.bool_(
        PRODUCT_SPREAD_INCLUDE_GSFC
    ),
    product_spread_definition=np.str_(PRODUCT_SPREAD_DEFINITION),
    combo_tags=COMBO_TAGS,
    combo_obp_sources=COMBO_OBP_SOURCES,
    combo_ssh_sources=COMBO_SSH_SOURCES,
    combo_wind_sources=COMBO_WIND_SOURCES,
    transfer_case_spec_json=np.str_(TRANSFER_CASE_SPEC_JSON),
    n_members=np.int64(members.shape[0]),
    sig_frac_single=np.float64(frac_single),
#: Fraction before adding the Stage-16 GRACE term.
    sig_frac_without_grace=np.float64(frac_full),
#: Fraction with all five terms when Stage 16 has run; NaN otherwise.
    sig_frac_with_grace=np.float64(
        np.nan if frac_full_grace is None else frac_full_grace),
    n_valid=np.int64(valid.sum()),
    med_serial=np.float64(np.nanmedian(full.sigma_serial[valid])),
    med_eps=np.float64(np.nanmedian(sigma_eps)),
    med_eps_propagated=np.float64(np.nanmedian(sigma_eps_propagated)),
)
try:
    np.savez(OUT_FILE, **payload)
except PermissionError as exc:
    raise SystemExit(
        f"Cannot overwrite {OUT_FILE}. Close any open NPZ file handles "
        "and rerun Stage 15."
    ) from exc
print("saved:", OUT_FILE)

# Trend statistics of the default reconstruction, used by the later stages.
save_realworld_trend_npz(
    rw,
    full,
    OUT_FILE.parent / f"real_world_trend_stats_without_grace{_tag}.npz",
    block_months=kw["block_months"],
    n_boot=kw["n_boot"],
    seed=kw["seed"],
    edge_months=EDGE_MONTHS,
    extra_fields={
        "training_source_run_id": np.asarray(ACTIVE_RUN_ID),
        "mapping_error_estimator": np.asarray(SIGMA_MAP_ESTIMATOR),
        "sigma_map_estimator_code": np.asarray(
            SIGMA_MAP_ESTIMATOR_CODE, dtype=np.int64
        ),
        "sigma_map_n_branch_windows": np.asarray(n_windows, dtype=np.int64),
        "sigma_map_n_member_lineages": np.asarray(
            n_member_lineages, dtype=np.int64
        ),
        "n_combos": np.asarray(aligned.shape[0], dtype=np.int64),
        "product_spread_include_gsfc": np.asarray(
            PRODUCT_SPREAD_INCLUDE_GSFC, dtype=np.bool_
        ),
        "product_spread_definition": np.asarray(
            PRODUCT_SPREAD_DEFINITION
        ),
        "combo_tags": COMBO_TAGS,
        "combo_obp_sources": COMBO_OBP_SOURCES,
        "combo_ssh_sources": COMBO_SSH_SOURCES,
        "combo_wind_sources": COMBO_WIND_SOURCES,
    },
)
