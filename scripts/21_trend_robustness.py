"""Stage 21: assess sensitivity of reconstructed trends.

Refit latitude-density trends after changing record endpoints, excluding
the GRACE/GRACE-FO gap, or changing the serial-error estimator. Other
uncertainty-budget terms remain fixed at their full-record values, so
short-window significance results require caution.
"""

import argparse
import os
import sys
from pathlib import Path

# This stage gains little from threaded MKL. Select sequential execution
# before importing NumPy; an explicit environment setting takes precedence.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurmoc.config import BASELINE_YEARS
from neurmoc.evaluation import (
    BASELINE_EXPERIMENT,
    COVARIATES_ALL,
    REALWORLD_ROOT,
    model_dir,
)
from neurmoc.moc_utils import find_nearest_index, robust_trend
from neurmoc.plotting.timeseries import GRACE_GAP_MONTHS
from neurmoc.results import load_real_world

# %% [0] Settings --------------------------------------------------------------
EDGE_MONTHS = 12
#: Same trend-estimator settings as Stage 15.
TREND_METHOD = "mbb"
TREND_BLOCK_MONTHS = 48
N_BOOT_TREND = 1000
TREND_SEED = 0
N_SIGMA = 2.0

#: start/end grid, in whole years trimmed from each end
TRIM_YEARS = (0, 1, 2, 3, 4)
#: buffer months added on EACH side of the GRACE gap before excision
GAP_BUFFERS = (0, 6, 12, 18, 24)
#: serial-correlation estimators. mbb0 is the independent-month reference.
MBB_BLOCKS = (0, 6, 12, 24, 48, 72)
# Short-record significance is diagnostic when other uncertainty terms remain fixed.
BUDGET_OK_FRACTION = 0.75

#: Color limit for the baseline trend row.
TREND_CLIM = 0.4
# Shared Sv/yr scale; None enables separate difference scales.
UNIFORM_CLIM = 0.4
# Difference scale when no uniform limit is specified.
DIFF_CLIM = None
AUTO_CLIM_PERCENTILE = 99.0
ROW_HEIGHT = 0.31          # figure height per section row, in DOUBLE_COL units

# Tests shown in each section figure.
FIGURE_ROWS = {
    "startend": ["start +0y, end -1y", "start +0y, end -2y",
                 "start +0y, end -4y", "start +1y, end -0y",
                 "start +2y, end -0y", "start +4y, end -0y"],
    # Separate gap-excision and short-segment sensitivity figures.
    "gracegap": ["excise gap +-0mo", "excise gap +-6mo",
                 "excise gap +-12mo", "excise gap +-24mo"],
    "gapsplit": ["pre-gap only", "post-gap only"],
    "serial": ["mbb L=0", "mbb L=12", "mbb L=24", "mbb L=72"],
}
# Maximum absolute slope change relative to baseline uncertainty by test family.
ENVELOPE_GROUPS = (
    ("start/end year grid", "startend", None),
    ("leave-one-year-out", "jackknife", None),
    ("GRACE gap excision", "gracegap", None),
)

_cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_cli.add_argument("-E", "--experiment", default=None)
_cli.add_argument("-x", "--covariates", default=None)
_cli.add_argument("--budget-tag", default=None,
                  help="stage-15 --out-tag without its leading underscore")
_cli.add_argument("--n-boot", type=int, default=N_BOOT_TREND,
                  help="bootstrap draws; lower only for a quick look")
_cli.add_argument("--no-jackknife", action="store_true",
                  help="skip the leave-one-year-out rows")
_cli.add_argument("--no-figure", action="store_true")
_args = _cli.parse_known_args()[0]

N_BOOT_TREND = int(_args.n_boot)
if N_BOOT_TREND < 1:
    raise SystemExit("--n-boot must be positive")
COVARIATES = (
    _args.covariates or os.environ.get("NEURMOC_FIG_COVARIATES") or COVARIATES_ALL
).replace(",", "+")
EXPERIMENT = (
    _args.experiment or os.environ.get("NEURMOC_FIG_EXPERIMENT") or BASELINE_EXPERIMENT
)
NN_DIR = model_dir(root=REALWORLD_ROOT, experiment=EXPERIMENT, covariates=COVARIATES)
OUT_DIR = NN_DIR / "RealWorld"
BUDGET_TAG = (
    _args.budget_tag
    if _args.budget_tag is not None
    else os.environ.get("NEURMOC_TREND_BUDGET_TAG", "")
).strip()
if BUDGET_TAG.startswith("_") or any(c in BUDGET_TAG for c in ("/", "\\")):
    raise SystemExit("BUDGET_TAG must be a filename-safe tag without '_' prefix")
_SUFFIX = f"_{BUDGET_TAG}" if BUDGET_TAG else ""
BUDGET_FILE = OUT_DIR / f"trend_error_budget{_SUFFIX}.npz"
GRACE_FILE = OUT_DIR / f"grace_noise_budget{_SUFFIX}.npz"
REFERENCE_FILE = OUT_DIR / f"real_world_trend_stats{_SUFFIX}.npz"
OUT_FILE = OUT_DIR / f"trend_robustness{_SUFFIX}.npz"
MD_FILE = OUT_DIR / f"trend_robustness{_SUFFIX}.md"

# %% [1] Reconstruction and the fixed budget terms -----------------------------
rw = load_real_world(NN_DIR, rmse_scenario=None, edge_months=EDGE_MONTHS)
n_t, n_lev, n_lat = rw.pred.shape
if rw.time_month is None:
    raise SystemExit(
        f"{NN_DIR} Pred_RealWorld has no time_month axis; the GRACE-gap rows "
        "need calendar months, not positions. Rerun stage 14."
    )
month_labels = np.asarray(rw.time_month, dtype="U7")
if rw.moc_baseline is None:
    raise SystemExit(
        f"{NN_DIR} Pred_RealWorld has no moc_baseline. It is the absolute "
        f"{BASELINE_YEARS[0]}-{BASELINE_YEARS[1]} MOC that turns the anomaly "
        "record into overturning strength, so it fixes the cell-core "
        "positions and the sign convention of every map below. The mean of "
        "the first 120 months of an already-demeaned record is NOT a "
        "substitute. Rerun stage 14."
    )

if not BUDGET_FILE.is_file():
    raise SystemExit(f"{BUDGET_FILE} missing - run 15_compute_trend_budget.py")
with np.load(BUDGET_FILE, allow_pickle=False) as _b:
    _need = {"sigma_map", "sigma_sate", "sigma_eps"}
    _missing = sorted(_need - set(_b.files))
    if _missing:
        raise SystemExit(f"{BUDGET_FILE} lacks {_missing} - rerun stage 15")
    sigma_map = np.asarray(_b["sigma_map"], dtype=float)
    sigma_sate = np.asarray(_b["sigma_sate"], dtype=float)
    sigma_eps = np.asarray(_b["sigma_eps"], dtype=float)

if GRACE_FILE.is_file():
    with np.load(GRACE_FILE, allow_pickle=False) as _g:
        sigma_grace = np.asarray(_g["sigma_grace"], dtype=float)
else:
    # A missing Stage-16 term is unavailable, not zero.
    print(f"WARNING {GRACE_FILE} missing - sigma_grace omitted from the "
          "budget; every significant fraction below is an UPPER bound")
    sigma_grace = np.zeros_like(sigma_sate)

for _name, _field in (("sigma_map", sigma_map), ("sigma_sate", sigma_sate),
                      ("sigma_eps", sigma_eps), ("sigma_grace", sigma_grace)):
    if _field.shape != (n_lev, n_lat):
        raise SystemExit(f"{_name} has shape {_field.shape}, expected "
                         f"{(n_lev, n_lat)}")
    if not np.isfinite(_field).all() or np.any(_field < 0):
        raise SystemExit(f"{_name} has non-finite or negative entries")

FIXED_SIGMA = dict(sigma_map=sigma_map, sigma_sate=sigma_sate,
                   sigma_eps=sigma_eps, sigma_grace=sigma_grace)


def fit(mask, *, method=TREND_METHOD, block_months=TREND_BLOCK_MONTHS):
    """robust_trend on the selected months, with the budget held fixed."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 24:
        raise ValueError(f"only {int(mask.sum())} months selected; "
                         "a trend needs at least two years")
    return robust_trend(
        rw.pred[mask], rw.t_years[mask], method=method,
        block_months=block_months, n_boot=N_BOOT_TREND, seed=TREND_SEED,
        **FIXED_SIGMA,
    )


# %% [2] Baseline validation ----------------------------------------------------
ALL_MONTHS = np.ones(n_t, dtype=bool)
base = fit(ALL_MONTHS)

if REFERENCE_FILE.is_file():
    with np.load(REFERENCE_FILE, allow_pickle=False) as _r:
        # Check direct estimator outputs exactly and quadrature totals within tolerance.
        for _key, _mine, _atol, _rtol in (
            ("slope_mean", base.slope_mean, 1e-12, 0.0),
            ("sigma_serial", base.sigma_serial, 1e-12, 0.0),
            ("sigma_total", base.sigma_total, 0.0, 1e-9),
        ):
            _ref = np.asarray(_r[_key], dtype=float)
            if _ref.shape != _mine.shape:
                raise SystemExit(f"{REFERENCE_FILE} {_key} shape {_ref.shape} "
                                 f"!= {_mine.shape}")
            _d = float(np.nanmax(np.abs(_ref - _mine)))
            _lim = _atol + _rtol * float(np.nanmax(np.abs(_ref)))
            if not np.isfinite(_d) or _d > _lim:
                raise SystemExit(
                    f"baseline does not reproduce {REFERENCE_FILE.name} "
                    f"{_key} (max |diff| = {_d:.3e} > {_lim:.3e}). The "
                    "estimator settings differ from the reference run; "
                    "resolve this before interpreting the sensitivity tests."
                )
        _settings = {
            "block_months": (int(_r["block_months"]), TREND_BLOCK_MONTHS),
            "n_boot": (int(_r["n_boot"]), N_BOOT_TREND),
            "seed": (int(_r["seed"]), TREND_SEED),
            "n_sigma": (float(_r["n_sigma"]), N_SIGMA),
        }
    _bad = {k: v for k, v in _settings.items() if v[0] != v[1]}
    if _bad:
        raise SystemExit(f"estimator settings differ from {REFERENCE_FILE.name}"
                         f": {_bad} (reference, here)")
    print(f"baseline reproduces {REFERENCE_FILE.name} exactly")
else:
    print(f"WARNING {REFERENCE_FILE} missing - baseline cannot be checked "
          "against the Stage-15 trend statistics")

# Use the same full-record, nonzero domain for every comparison.
FAMILY = np.isfinite(base.slope_pval) & (rw.pred.std(axis=0) > 0)
n_family = int(FAMILY.sum())
base_sig = base.is_significant(N_SIGMA) & FAMILY
base_slope = base.slope_mean
base_sigma = base.sigma_total

print(f"\nrecord     : {month_labels[0]} .. {month_labels[-1]} "
      f"({n_t} months, {(rw.t_years[-1] - rw.t_years[0]):.2f} yr)")
print(f"grid       : {n_lev} density x {n_lat} latitude, "
      f"{n_family} testable cells of {n_lev * n_lat}")
print(f"baseline   : {base_sig.sum() / n_family:.1%} significant "
      f"(+-{N_SIGMA:g} sigma, per point; FDR not applied to these tests)")

# %% [3] Diagnostic cells ------------------------------------------------------
CELL_SPECS = (
    ("AMOC 26.5N mid-depth", 26.5, "mid"),
    ("AMOC 45N mid-depth", 45.0, "mid"),
    ("SMOC 60S abyssal", -60.0, "abyssal"),
)
cells = []
for _label, _lat0, _which in CELL_SPECS:
    j = find_nearest_index(rw.lat, _lat0)
    k = int(rw.cores.mid_index[j] if _which == "mid"
            else rw.cores.abyssal_index[j])
    cells.append((_label, int(j), k, float(rw.lat[j])))
    print(f"  cell {_label:22s} lat {rw.lat[j]:+6.2f}  "
          f"sigma2 {rw.sigma2[k]:.2f}  baseline "
          f"{base_slope[k, j]:+.4f} +- {2 * base_sigma[k, j]:.4f} Sv/yr")

# %% [4] Month selections ------------------------------------------------------
gap_lo, gap_hi = GRACE_GAP_MONTHS
in_gap = (month_labels >= gap_lo) & (month_labels <= gap_hi)
if not in_gap.any():
    raise SystemExit(f"no month of the record falls in the GRACE gap "
                     f"{gap_lo}..{gap_hi}; check the time axis")
gap_idx = np.flatnonzero(in_gap)


def gap_excision(buffer_months: int) -> np.ndarray:
    """Keep every month outside the gap widened by `buffer_months` a side."""
    lo = max(0, gap_idx[0] - buffer_months)
    hi = min(n_t - 1, gap_idx[-1] + buffer_months)
    keep = np.ones(n_t, dtype=bool)
    keep[lo:hi + 1] = False
    return keep


# Use calendar month labels to assign years; decimal-year flooring misclassifies December.
years = np.array([int(s[:4]) for s in month_labels])
uniq_years = np.unique(years)

Test = []


def add(group, label, mask, note, method=TREND_METHOD,
        block=TREND_BLOCK_MONTHS, trim_start=-1, trim_end=-1):
    Test.append(dict(group=group, label=label, mask=np.asarray(mask, bool),
                     note=note, method=method, block=block,
                     trim_start=trim_start, trim_end=trim_end))


add("baseline", "full record", ALL_MONTHS,
    "reference settings; matches the Stage-15 trend statistics",
    trim_start=0, trim_end=0)

for a in TRIM_YEARS:
    for b in TRIM_YEARS:
        if a == 0 and b == 0:
            continue
        lo = rw.t_years[0] + a
        hi = rw.t_years[-1] - b
        m = (rw.t_years >= lo - 1e-9) & (rw.t_years <= hi + 1e-9)
        if m.sum() < 12 * 12:      # keep at least 12 years
            continue
        add("startend", f"start +{a}y, end -{b}y", m,
            f"{month_labels[m][0]}..{month_labels[m][-1]}",
            trim_start=a, trim_end=b)

if not _args.no_jackknife:
    for y in uniq_years:
        m = years != y
        if m.sum() < 12 * 12:
            continue
        add("jackknife", f"drop {y}", m, f"{int((~m).sum())} months removed")

for buf in GAP_BUFFERS:
    m = gap_excision(buf)
    add("gracegap", f"excise gap +-{buf}mo", m,
        f"{int((~m).sum())} months removed "
        f"({month_labels[~m][0]}..{month_labels[~m][-1]})")

pre = np.zeros(n_t, dtype=bool)
pre[:gap_idx[0]] = True
post = np.zeros(n_t, dtype=bool)
post[gap_idx[-1] + 1:] = True
for _lbl, _m in (("pre-gap only", pre), ("post-gap only", post)):
    if _m.sum() >= 24:
        add("gapsplit", _lbl, _m,
            f"contiguous {month_labels[_m][0]}..{month_labels[_m][-1]}")

for blk in MBB_BLOCKS:
    if blk == TREND_BLOCK_MONTHS:
        continue
    add("serial", f"mbb L={blk}", ALL_MONTHS,
        "independent months" if blk == 0 else f"{blk}-month circular blocks",
        block=blk)

print(f"\nrunning {len(Test)} tests "
      f"({TREND_METHOD}, {N_BOOT_TREND} draws, seed {TREND_SEED}) ...")

# %% [5] Run - every test keeps its full map -----------------------------------
n_test = len(Test)
slope_maps = np.full((n_test, n_lev, n_lat), np.nan)
sigma_maps = np.full((n_test, n_lev, n_lat), np.nan)
sig_maps = np.zeros((n_test, n_lev, n_lat), dtype=bool)
rows = []
for i, t in enumerate(Test):
    tr = fit(t["mask"], method=t["method"], block_months=t["block"])
    slope_maps[i] = tr.slope_mean
    sigma_maps[i] = tr.sigma_total
    sig_maps[i] = tr.is_significant(N_SIGMA) & FAMILY

    d = tr.slope_mean - base_slope
    with np.errstate(invalid="ignore", divide="ignore"):
        d_over_sigma = np.abs(d) / base_sigma
    f = FAMILY
    x, y = base_slope[f], tr.slope_mean[f]
    ok = np.isfinite(x) & np.isfinite(y)
    corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan
    bs = base_sig
    retained = float(sig_maps[i][bs].mean()) if bs.any() else np.nan
    flipped = (float(np.mean(np.sign(tr.slope_mean[bs])
                             != np.sign(base_slope[bs])))
               if bs.any() else np.nan)
    # Compare the refitted serial term with its full-record value.
    serial_ratio = float(np.nanmedian(tr.sigma_serial[f])
                         / np.nanmedian(base.sigma_serial[f]))
    rows.append(dict(
        group=t["group"], label=t["label"], note=t["note"],
        method=t["method"], block=int(t["block"]),
        trim_start=int(t["trim_start"]), trim_end=int(t["trim_end"]),
        sigma_serial_ratio=serial_ratio,
        budget_ok=bool(t["mask"].sum() >= BUDGET_OK_FRACTION * n_t),
        n_months=int(t["mask"].sum()),
        t_start=float(rw.t_years[t["mask"]][0]),
        t_end=float(rw.t_years[t["mask"]][-1]),
        sig_frac=float(sig_maps[i].sum() / n_family),
        retained_frac=retained, signflip_frac=flipped, slope_corr=corr,
        med_abs_dslope=float(np.nanmedian(np.abs(d[f]))),
        max_abs_dslope=float(np.nanmax(np.abs(d[f]))),
        med_dslope_over_sigma=float(np.nanmedian(d_over_sigma[f])),
        max_dslope_over_sigma=float(np.nanmax(d_over_sigma[f])),
        frac_dslope_gt_1sigma=float(np.nanmean(d_over_sigma[f] > 1.0)),
        frac_dslope_gt_2sigma=float(np.nanmean(d_over_sigma[f] > 2.0)),
        cell_slope=[float(tr.slope_mean[k, j]) for _, j, k, _ in cells],
        cell_sigma=[float(tr.sigma_total[k, j]) for _, j, k, _ in cells],
    ))
    if (i + 1) % 10 == 0 or i + 1 == n_test:
        print(f"  {i + 1}/{n_test}")

LABEL_INDEX = {r["label"]: i for i, r in enumerate(rows)}
#: difference maps, NaN outside the reconstructed domain
diff_maps = slope_maps - base_slope[None]
diff_maps[:, ~FAMILY] = np.nan

# Serial estimators change uncertainty but not the OLS slope.
for i, r in enumerate(rows):
    if r["group"] == "serial":
        _dmax = float(np.nanmax(np.abs(diff_maps[i])))
        if _dmax > 1e-12:
            raise SystemExit(
                f"serial estimator {r['label']} changed the OLS slope by "
                f"{_dmax:.3e} Sv/yr; it must not. The estimator has drifted.")

# %% [6] Console table ---------------------------------------------------------
GROUP_TITLE = {
    "baseline": "BASELINE",
    "startend": "START / END YEAR (budget held fixed - see the caveat)",
    "jackknife": "LEAVE-ONE-YEAR-OUT",
    "gracegap": "GRACE/GRACE-FO GAP EXCISION",
    "gapsplit": "GAP SPLIT (contiguous segments)",
    "serial": "SERIAL-CORRELATION ESTIMATOR (slope unchanged by construction)",
}
HEAD = (f"{'test':26s} {'n':>4s} {'sig%':>6s} {'kept%':>6s} {'flip%':>6s} "
        f"{'r':>6s} {'med':>6s} {'max':>6s} {'>1sig':>6s} {'sr':>5s} "
        + "".join(f"{lbl.split()[0] + ' ' + lbl.split()[1]:>18s}"
                  for lbl, _, _, _ in cells))
for grp in ("baseline", "startend", "jackknife", "gracegap", "gapsplit",
            "serial"):
    sel = [r for r in rows if r["group"] == grp]
    if not sel:
        continue
    print(f"\n{GROUP_TITLE[grp]}")
    print("   med/max are |d slope| / sigma_total over all "
          f"{n_family} reconstructed cells")
    print(HEAD)
    print("-" * len(HEAD))
    for r in sel:
        cellstr = "".join(f"{s:+9.4f}+-{2 * u:<7.4f}"
                          for s, u in zip(r["cell_slope"], r["cell_sigma"]))
        flag = " " if r["budget_ok"] else "!"
        print(f"{r['label']:25s}{flag} {r['n_months']:4d} "
              f"{100 * r['sig_frac']:6.1f} {100 * r['retained_frac']:6.1f} "
              f"{100 * r['signflip_frac']:6.1f} {r['slope_corr']:6.3f} "
              f"{r['med_dslope_over_sigma']:6.2f} "
              f"{r['max_dslope_over_sigma']:6.2f} "
              f"{100 * r['frac_dslope_gt_1sigma']:6.1f} "
              f"{r['sigma_serial_ratio']:5.2f} {cellstr}")

_short = [r for r in rows if not r["budget_ok"]]
if _short:
    print(f"\n! {len(_short)} row(s) keep < {BUDGET_OK_FRACTION:.0%} of the "
          "record. Interpret their significance with caution: four budget "
          "terms remain fixed at full-record values. Slope differences "
          "remain valid.")
    for r in _short:
        print(f"    {r['label']:24s} {r['n_months']:4d} months "
              f"({r['n_months'] / n_t:.0%}), sigma_serial x"
              f"{r['sigma_serial_ratio']:.2f}, diagnostic sig "
              f"{100 * r['sig_frac']:.1f}%")

_var = [r for r in rows
        if r["group"] in ("startend", "jackknife", "gracegap", "gapsplit")]
_ok = [r for r in _var if r["budget_ok"]]
_worst = max(_ok, key=lambda r: r["med_dslope_over_sigma"])
print(f"\nacross {len(_ok)} record-definition tests with sufficient coverage "
      f"({len(_var) - len(_ok)} short rows excluded): median |d slope| stays "
      f"below {_worst['med_dslope_over_sigma']:.2f} sigma_total "
      f"(worst: {_worst['label']}); slope-map correlation stays above "
      f"{min(r['slope_corr'] for r in _ok):.3f}; at most "
      f"{100 * max(r['signflip_frac'] for r in _ok):.1f}% of the "
      f"{int(base_sig.sum())} baseline-significant cells change sign; the "
      f"per-point significant fraction ranges "
      f"{100 * min(r['sig_frac'] for r in _ok):.1f}-"
      f"{100 * max(r['sig_frac'] for r in _ok):.1f}% against "
      f"{100 * base_sig.sum() / n_family:.1f}% for the full record")

# Report both per-test medians and per-cell worst cases.
print("\nper-cell worst case over each family (max over tests, then the "
      "distribution over cells):")
for _title, _grp, _ in ENVELOPE_GROUPS:
    _idx = [i for i, r in enumerate(rows) if r["group"] == _grp]
    if not _idx:
        continue
    with np.errstate(invalid="ignore", divide="ignore"):
        _env = np.nanmax(np.abs(diff_maps[_idx]) / base_sigma[None], axis=0)
    _e = _env[FAMILY]
    print(f"  {_title:22s} ({len(_idx):2d} tests) median "
          f"{np.nanmedian(_e):.2f}, p95 {np.nanpercentile(_e, 95):.2f}, "
          f"max {np.nanmax(_e):.2f} sigma_total; "
          f"{100 * np.nanmean(_e > 1.0):.0f}% of cells exceed 1 sigma")

_ser = [r for r in rows if r["group"] == "serial"]
if _ser:
    print(f"\nthe bootstrap block length leaves every slope untouched and "
          f"moves only sigma_total: the per-point significant fraction runs "
          f"{100 * min(r['sig_frac'] for r in _ser):.1f}-"
          f"{100 * max(r['sig_frac'] for r in _ser):.1f}%. The serial term is "
          "stable for blocks of 24 months or longer; 48 months is used.")

# %% [7] Save maps --------------------------------------------------------------
def _col(name, dtype=float):
    return np.array([r[name] for r in rows], dtype=dtype)


np.savez_compressed(
    OUT_FILE,
    schema_version=np.int64(2),
    group=np.array([r["group"] for r in rows]),
    label=np.array([r["label"] for r in rows]),
    note=np.array([r["note"] for r in rows]),
    method=np.array([r["method"] for r in rows]),
    block_months=_col("block", np.int64),
    trim_start=_col("trim_start", np.int64), trim_end=_col("trim_end", np.int64),
    n_months=_col("n_months", np.int64),
    t_start=_col("t_start"), t_end=_col("t_end"),
    sigma_serial_ratio=_col("sigma_serial_ratio"),
    budget_ok=_col("budget_ok", bool),
    budget_ok_fraction=np.float64(BUDGET_OK_FRACTION),
    # [n_test, n_lev, n_lat] sensitivity maps.
    slope_map=slope_maps, sigma_total_map=sigma_maps,
    significant_map=sig_maps, diff_map=diff_maps,
    sig_frac=_col("sig_frac"), retained_frac=_col("retained_frac"),
    signflip_frac=_col("signflip_frac"), slope_corr=_col("slope_corr"),
    med_abs_dslope=_col("med_abs_dslope"),
    max_abs_dslope=_col("max_abs_dslope"),
    med_dslope_over_sigma=_col("med_dslope_over_sigma"),
    max_dslope_over_sigma=_col("max_dslope_over_sigma"),
    frac_dslope_gt_1sigma=_col("frac_dslope_gt_1sigma"),
    frac_dslope_gt_2sigma=_col("frac_dslope_gt_2sigma"),
    cell_slope=np.array([r["cell_slope"] for r in rows], dtype=float),
    cell_sigma=np.array([r["cell_sigma"] for r in rows], dtype=float),
    cell_label=np.array([c[0] for c in cells]),
    cell_lat=np.array([c[3] for c in cells], dtype=float),
    cell_lev_index=np.array([c[2] for c in cells], dtype=np.int64),
    baseline_slope=base_slope, baseline_sigma_total=base_sigma,
    baseline_sig=base_sig, family=FAMILY, n_family=np.int64(n_family),
    moc_baseline=rw.moc_baseline,
    lat=rw.lat, sigma2=rw.sigma2, t_years=rw.t_years,
    month_labels=month_labels,
    grace_gap_months=np.array(GRACE_GAP_MONTHS),
    n_sigma=np.float64(N_SIGMA), n_boot=np.int64(N_BOOT_TREND),
    seed=np.int64(TREND_SEED), edge_months=np.int64(EDGE_MONTHS),
    fdr_note=np.array("false-discovery-rate control is deliberately NOT "
                      "applied to these sensitivity tests; significant_map "
                      "is the per-point +-n_sigma rule."),
    budget_held_fixed=np.array(
        "sigma_map/sigma_sate/sigma_eps/sigma_grace are full-record values "
        "reused in every sub-period row; only sigma_serial is refitted, so "
        "short-window sigma_total is understated and its significance "
        "overstated. The slope and diff maps are unaffected."),
    run_id=np.array(str(rw.run_id)),
    training_source_run_id=np.array(str(rw.training_source_run_id)),
    satellite_dataset_id=np.array(str(rw.satellite_dataset_id)),
    covariates=np.array(COVARIATES), training_experiment=np.array(EXPERIMENT),
)
print(f"\nsaved: {OUT_FILE}")

# %% [8] Markdown summary ------------------------------------------------------
def md_table(sel):
    out = ["| test | months | period | \u00b12\u03c3 sig. | kept | sign flip "
           "| med \\|\u0394\u03b2\\|/\u03c3 | max \\|\u0394\u03b2\\|/\u03c3 | slope *r* |",
           "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sel:
        mark = "" if r["budget_ok"] else " \u26a0"
        sig = ("diagnostic only" if not r["budget_ok"]
               else f"{100 * r['sig_frac']:.1f}%")
        out.append(
            f"| {r['label']}{mark} | {r['n_months']} | "
            f"{r['t_start']:.2f}\u2013{r['t_end']:.2f} | {sig} | "
            f"{100 * r['retained_frac']:.1f}% | "
            f"{100 * r['signflip_frac']:.1f}% | "
            f"{r['med_dslope_over_sigma']:.2f} | "
            f"{r['max_dslope_over_sigma']:.2f} | {r['slope_corr']:.3f} |")
    return "\n".join(out)


_md = [
    "# Trend robustness tests",
    "",
    f"Record {month_labels[0]}\u2013{month_labels[-1]} ({n_t} months). "
    f"Estimator: `robust_trend`, {TREND_METHOD}, {TREND_BLOCK_MONTHS}-month "
    f"blocks, {N_BOOT_TREND} draws, seed {TREND_SEED}. Every test is "
    f"evaluated at all {n_family} reconstructed density\u2013latitude cells; the "
    "difference maps are the figures. Baseline: "
    f"{100 * base_sig.sum() / n_family:.1f}% of cells significant at "
    f"\u00b1{N_SIGMA:g}\u03c3 per point. False-discovery-rate control is not applied "
    "to these tests; it answers a different question from the local "
    "sensitivity analysis.",
    "",
    "**Caveat carried in every sub-period row.** \u03c3_map, \u03c3_sate, \u03c3_eps and "
    "\u03c3_grace were measured on the full record and are held fixed; only "
    "\u03c3_serial is refitted. Short windows therefore have an understated "
    "\u03c3_total and an overstated significant fraction. Rows marked \u26a0 keep "
    f"less than {BUDGET_OK_FRACTION:.0%} of the record and their "
    "significance is withheld rather than printed. The `|\u0394\u03b2|/\u03c3` and "
    "slope-correlation columns, and every difference map, are free of this "
    "caveat.",
    "",
]
for grp in ("startend", "jackknife", "gracegap", "gapsplit", "serial"):
    sel = [r for r in rows if r["group"] == grp]
    if sel:
        _md += [f"## {GROUP_TITLE[grp]}", "", md_table(sel), ""]
_md += [
    "## Not covered here",
    "",
    "- **Filter cutoff.** `LPF_OBS` is applied to the network *inputs* in "
    "stage 12, so a different cutoff is a stage-12/14 rerun, not a refit.",
    "- **GRACE gap-fill treatment.** Also fixed in stage 12 (linear "
    "interpolation). The excision rows above bound how much it can matter.",
    "",
]
MD_FILE.write_text("\n".join(_md), encoding="utf-8")
print(f"saved: {MD_FILE}")

# %% [9] Section figures -------------------------------------------------------
if not _args.no_figure:
    import matplotlib.pyplot as plt

    from neurmoc.plotting import (
        CMAP_AMPLITUDE,
        CMAP_DIVERGING,
        DOUBLE_COL,
        apply_style,
        save_figure,
        section_row,
    )

    apply_style()
    sign_ref = rw.moc_baseline
    SEC = dict(negative_mask=sign_ref < 0,
               core_curves_smoc=(rw.cores.mid_sigma2, rw.cores.abyssal_sigma2),
               core_curves_amoc=(rw.cores.mid_sigma2,))

    def section_stack(panels, out_name, shared_from=1):
        """Stacked SMOC|AMOC rows; rows >= shared_from share one colorbar.

        `panels` are (field, title, cmap, vmin, vmax, unit). Row 0 keeps its
        own scale - it is the baseline trend, not a difference - exactly as
        the sea-ice sensitivity figure separates the two.
        """
        n_rows = len(panels)
        fig = plt.figure(figsize=(DOUBLE_COL, ROW_HEIGHT * DOUBLE_COL * n_rows))
        gs = fig.add_gridspec(n_rows, 1, hspace=0.62,
                              left=0.08, right=0.88, bottom=0.05, top=0.94)
        shared_axes, shared_mesh, shared_unit = [], None, ""
        for row, (field, title, cmap, vmin, vmax, unit) in enumerate(panels):
            _, (ax_s, ax_a), mesh = section_row(
                field, rw.lat, rw.sigma2, cmap=cmap, vmin=vmin, vmax=vmax,
                fig=fig, subplot_spec=gs[row], add_colorbar=False, **SEC)
            ax_s.set_title(f"{chr(ord('A') + row)}   {title}", loc="left",
                           fontsize=plt.rcParams["font.size"])
            if row < shared_from:
                cbar = fig.colorbar(mesh, ax=[ax_s, ax_a], pad=0.015,
                                    fraction=0.03, aspect=22)
                cbar.outline.set_visible(False)
                cbar.ax.set_title(unit, fontsize=plt.rcParams["font.size"],
                                  pad=6)
            else:
                shared_axes += [ax_s, ax_a]
                shared_mesh, shared_unit = mesh, unit
        if shared_mesh is not None:
            cbar = fig.colorbar(shared_mesh, ax=shared_axes, pad=0.015,
                                fraction=0.03, aspect=22 * (n_rows - shared_from))
            cbar.outline.set_visible(False)
            cbar.ax.set_title(shared_unit, fontsize=plt.rcParams["font.size"],
                              pad=6)
        save_figure(fig, OUT_DIR / f"{out_name}{_SUFFIX}",
                    formats=("png", "pdf"))
        print(f"saved: {out_name}{_SUFFIX}.png / .pdf")

    def diff_limit(indices):
        """One symmetric scale for the difference rows actually drawn."""
        if UNIFORM_CLIM:
            return float(UNIFORM_CLIM)
        if DIFF_CLIM:
            return float(DIFF_CLIM)
        pool = np.abs(diff_maps[list(indices)][:, FAMILY])
        lim = float(np.nanpercentile(pool, AUTO_CLIM_PERCENTILE))
        return lim if np.isfinite(lim) and lim > 0 else TREND_CLIM / 10

    base_clim = float(UNIFORM_CLIM) if UNIFORM_CLIM else TREND_CLIM
    base_row = (np.where(FAMILY, base_slope, np.nan),
                "Reconstruction trend (baseline)", CMAP_DIVERGING,
                -base_clim, base_clim, "Sv yr$^{-1}$")

    # --- one figure per family of record-definition tests --------------------
    for name in ("startend", "gracegap", "gapsplit"):
        labels = [lb for lb in FIGURE_ROWS[name] if lb in LABEL_INDEX]
        missing = [lb for lb in FIGURE_ROWS[name] if lb not in LABEL_INDEX]
        if missing:
            print(f"  {name}: skipping unavailable rows {missing}")
        if not labels:
            continue
        idx = [LABEL_INDEX[lb] for lb in labels]
        lim = diff_limit(idx)
        panels = [base_row]
        for lb, i in zip(labels, idx):
            r = rows[i]
            mark = "" if r["budget_ok"] else "  [budget held fixed: "\
                                             "significance diagnostic only]"
            panels.append((diff_maps[i],
                           f"{lb}   minus baseline   "
                           f"({r['n_months']} months){mark}",
                           CMAP_DIVERGING, -lim, lim,
                           r"$\Delta$ trend (Sv yr$^{-1}$)"))
        section_stack(panels, f"trend_robustness_{name}")
        print(f"  {name}: difference scale +-{lim:.4g} Sv/yr")

    # --- worst case over each family, relative to baseline uncertainty ------
    envs = []
    for title, grp, _ in ENVELOPE_GROUPS:
        idx = [i for i, r in enumerate(rows) if r["group"] == grp]
        if not idx:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = (np.abs(diff_maps[idx]) if UNIFORM_CLIM
                      else np.abs(diff_maps[idx]) / base_sigma[None])
            env = np.nanmax(scaled, axis=0)
        env[~FAMILY] = np.nan
        envs.append((title, len(idx), env))
    if envs and UNIFORM_CLIM:
        # Sv/yr, so the shared trend scale applies. Non-negative by
        # construction, hence [0, lim] rather than a symmetric limit.
        env_panels = [base_row] + [
            (env, f"worst |$\\Delta\\beta$| over {title} ({n} tests; "
                  f"median {np.nanmedian(env):.3f}, max "
                  f"{np.nanmax(env):.3f} Sv yr$^{{-1}}$)",
             CMAP_AMPLITUDE, 0.0, float(UNIFORM_CLIM),
             r"$|\Delta$ trend$|$ (Sv yr$^{-1}$)")
            for title, n, env in envs]
        section_stack(env_panels, "trend_robustness_envelope")
        for title, _, env in envs:
            print(f"  envelope {title}: max {np.nanmax(env):.4g}, "
                  f"scale 0..{float(UNIFORM_CLIM):.3g} Sv/yr")
    elif envs:
        # Use per-row scales for test families with different magnitudes.
        env_panels = [base_row]
        for title, n, env in envs:
            lim = max(0.05, float(np.nanpercentile(env[FAMILY],
                                                   AUTO_CLIM_PERCENTILE)))
            env_panels.append((
                env, f"worst |$\\Delta\\beta$| over {title} ({n} tests; "
                     f"median {np.nanmedian(env):.2f}, max "
                     f"{np.nanmax(env):.2f} $\\sigma_{{\\rm total}}$)",
                CMAP_AMPLITUDE, 0.0, lim,
                r"$|\Delta\beta| / \sigma_{\rm total}$"))
            print(f"  envelope {title}: scale 0..{lim:.3g} sigma_total")
        # Each row has its own colorbar.
        section_stack(env_panels, "trend_robustness_envelope",
                      shared_from=len(env_panels))

    # --- serial estimators: slopes identical, so map sigma_total ------------
    labels = [lb for lb in FIGURE_ROWS["serial"] if lb in LABEL_INDEX]
    if labels:
        idx = [LABEL_INDEX[lb] for lb in labels]
        with np.errstate(invalid="ignore", divide="ignore"):
            # Use Sv/yr under a uniform limit; otherwise show the ratio.
            fields = [np.where(FAMILY,
                               sigma_maps[i] - base_sigma if UNIFORM_CLIM
                               else sigma_maps[i] / base_sigma, np.nan)
                      for i in idx]
        base_sigma_top = (float(UNIFORM_CLIM) if UNIFORM_CLIM
                          else float(np.nanpercentile(base_sigma[FAMILY],
                                                      AUTO_CLIM_PERCENTILE)))
        panels = [(np.where(FAMILY, base_sigma, np.nan),
                   f"$\\sigma_{{\\rm total}}$, baseline "
                   f"(mbb L={TREND_BLOCK_MONTHS})",
                   CMAP_AMPLITUDE, 0.0, base_sigma_top, "Sv yr$^{-1}$")]
        if UNIFORM_CLIM:
            lo, hi = -float(UNIFORM_CLIM), float(UNIFORM_CLIM)
            unit = r"$\Delta\sigma_{\rm total}$ (Sv yr$^{-1}$)"
            fmt = ("{lb}   $\\sigma_{{\\rm total}}$ minus baseline "
                   "(median {m:+.3f} Sv yr$^{{-1}}$; slope identical)")
        else:
            spread = float(np.nanpercentile(
                np.abs(np.concatenate([f[FAMILY] for f in fields]) - 1.0),
                AUTO_CLIM_PERCENTILE))
            lo, hi = 1.0 - spread, 1.0 + spread
            unit = r"$\sigma_{\rm total}$ ratio"
            fmt = ("{lb}   $\\sigma_{{\\rm total}}$ ratio to baseline "
                   "(median {m:.2f}; slope identical)")
        for lb, field in zip(labels, fields):
            panels.append((field,
                           fmt.format(lb=lb, m=np.nanmedian(field)),
                           CMAP_DIVERGING, lo, hi, unit))
        section_stack(panels, "trend_robustness_serial")
        print(f"  serial: scale {lo:.3g}..{hi:.3g}, largest drawn "
              f"{np.nanmax([np.nanmax(np.abs(f)) for f in fields]):.4g}")

print("Done.")
