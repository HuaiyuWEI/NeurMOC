"""Loaders for evaluation products consumed by the figure scripts.

These read .mat/.npz evaluation and real-world reconstruction products and
return arrays in a single convention:
`[time, n_lev, n_lat]` with `sigma2 = rho2 - 1000`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BASELINE_YEARS, N_LATS, N_LEVS, N_TEST_REALIZATIONS
from .filtering import detrend_by_realization, std_by_realization
from .io_utils import load_npz_or_mat, require_file
from .moc_utils import (
    CellCores,
    extract_at_cores,
    fill_down_columns,
    locate_cell_cores,
    pointwise_r,
    pointwise_r2,
    pointwise_rmse,
)
from .timeaxis import decimal_year, normalize_month_axis


def _to_tzy(flat_or_grid: np.ndarray) -> np.ndarray:
    """Normalize `[T, n_lev*n_lat]` or `[T, n_lev, n_lat]` to `[T, n_lev, n_lat]`."""
    arr = np.asarray(flat_or_grid)
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0], N_LEVS, N_LATS)
    return arr


@dataclass
class ScenarioEval:
    """Stage-10 evaluation of one network on one CMIP scenario.

    Truth and prediction come from clean-input evaluation (`Pred_<tag>`).
    Observational uncertainty is quantified from alternative-product
    reconstructions rather than this evaluation."""

    truth: np.ndarray          # [T, lev, lat] diagnosed MOC (Sv)
    pred: np.ndarray           # [T, lev, lat] reconstruction
    lat: np.ndarray
    sigma2: np.ndarray         # rho2 - 1000
    uncertainty: np.ndarray | None = None  # optional input-noise band
    n_realizations: int = N_TEST_REALIZATIONS

    @property
    def rmse(self) -> np.ndarray:            # [lev, lat]
        return pointwise_rmse(self.truth, self.pred)

    @property
    def truth_std(self) -> np.ndarray:       # [lev, lat]
        """Mean per-member std; pooling members would mix their offsets into variability."""
        return std_by_realization(self.truth, self.n_realizations)

    @property
    def r2(self) -> np.ndarray:              # [lev, lat] squared correlation
        return pointwise_r2(self.truth, self.pred)

    @property
    def r(self) -> np.ndarray:               # [lev, lat] signed correlation
        return pointwise_r(self.truth, self.pred)

    def r_detrended(self) -> np.ndarray:
        """Signed correlation after removing each realization's trend."""
        truth_dt = detrend_by_realization(self.truth, self.n_realizations)
        pred_dt = detrend_by_realization(self.pred, self.n_realizations)
        return pointwise_r(truth_dt, pred_dt)

    def r2_detrended(self) -> np.ndarray:
        """Squared correlation after removing each realization's linear trend."""
        truth_dt = detrend_by_realization(self.truth, self.n_realizations)
        pred_dt = detrend_by_realization(self.pred, self.n_realizations)
        return pointwise_r2(truth_dt, pred_dt)

    @property
    def samples_per_realization(self) -> int:
        return self.truth.shape[0] // self.n_realizations

    def realization_slice(self, realization: int) -> slice:
        """0-based realization index -> time slice of that block."""
        n = self.samples_per_realization
        return slice(realization * n, (realization + 1) * n)


def load_scenario_eval(nn_dir: Path | str, cmip_tag: str,
                       n_realizations: int = N_TEST_REALIZATIONS) -> ScenarioEval:
    """Read the stage-10 evaluation `Pred_<tag>` + `TestR2_<tag>` grids."""
    nn_dir = Path(nn_dir)
    data = load_npz_or_mat(nn_dir / f"Pred_{cmip_tag}", ["y", "y_pred"])
    grid = load_npz_or_mat(nn_dir / f"TestR2_{cmip_tag}", ["rho2", "lat_psi"])
    rho2 = np.asarray(grid["rho2"]).squeeze()
    sigma2 = rho2 - 1000.0 if rho2[0] > 1000 else rho2
    return ScenarioEval(
        truth=_to_tzy(data["y"]),
        pred=_to_tzy(data["y_pred"]),
        lat=np.asarray(grid["lat_psi"]).squeeze(),
        sigma2=sigma2,
        n_realizations=n_realizations,
    )


def load_reference_moc(nn_dir: Path | str, cmip_tag: str) -> np.ndarray:
    """Diagnosed MOC `[T, lev, lat]` from a stage-10 `Pred_<tag>` output.

    Used to define the time-mean MOC for cell cores and the negative-cell mask.
    """
    data = load_npz_or_mat(Path(nn_dir) / f"Pred_{cmip_tag}", ["y"])
    return _to_tzy(data["y"])


@dataclass
class RealWorldReconstruction:
    """Satellite-based reconstruction plus its uncertainty decomposition."""

    pred: np.ndarray           # [T, lev, lat] member-mean reconstruction (Sv)
    epistemic: np.ndarray      # [T, lev, lat] member spread
    total_uncertainty: np.ndarray  # [T, lev, lat] assembled monthly 1-sigma
    t_years: np.ndarray        # decimal years
    lat: np.ndarray
    sigma2: np.ndarray
    cores: CellCores
    #: [lev, lat] reference mean for anomaly products, if available.
    moc_baseline: np.ndarray | None = None
    time_month: np.ndarray | None = None  # normalized YYYY-MM labels when available
    wind_convention: str | None = None
    wind_baseline_years: tuple[int, int] | None = None
    moc_convention: str | None = None
    moc_baseline_years: tuple[int, int] | None = None
    input_covariates: tuple[str, ...] = ()
    input_sources: tuple[str, ...] = ()
    input_source_files: tuple[str, ...] = ()
    input_baseline_specs: tuple[str, ...] = ()
    trained_on: str | None = None
    training_experiment: str | None = None
    ensemble_num_folds: int | None = None
    ensemble_members_per_fold: int | None = None
    run_id: str | None = None
    training_source_run_id: str | None = None
    cmip_dataset_id: str | None = None
    satellite_dataset_id: str | None = None

    def strength(self, which: str = "mid") -> tuple[np.ndarray, np.ndarray]:
        """Strength and current uncertainty at a cell core, `[T, n_lat]`."""
        idx = self.cores.mid_index if which == "mid" else self.cores.abyssal_index
        return (extract_at_cores(self.pred, idx),
                extract_at_cores(self.total_uncertainty, idx))


def _scenario_rmse(nn_dir: Path, scenario: str,
                   variant: str = "debiased") -> np.ndarray:
    """Cross-model RMSE, with optional removal of the mean transfer bias.

    Saved per-member estimates are preferred; older products use a pooled
    fallback. The `debiased` variant matches mean-centered reconstructions.
    """
    key = {"debiased": "rmse_debiased_yz", "full": "rmse_yz"}.get(variant)
    if key is None:
        raise ValueError(f"unknown scenario-RMSE variant {variant!r} "
                         "(use 'debiased' or 'full')")
    try:
        grid = load_npz_or_mat(Path(nn_dir) / f"TestR2_{scenario}")
        if key in grid:
            # Extend the deepest available error within each latitude column.
            return fill_down_columns(np.asarray(grid[key]))
    except FileNotFoundError:
        pass
    data = load_npz_or_mat(Path(nn_dir) / f"Pred_{scenario}", ["y", "y_pred"])
    err = _to_tzy(data["y_pred"]) - _to_tzy(data["y"])
    if variant == "debiased":
        err = err - np.nanmean(err, axis=0)      # pooled-bias fallback
    return fill_down_columns(np.sqrt(np.nanmean(err**2, axis=0)))


def training_moc_baseline(lat: np.ndarray, sigma2: np.ndarray,
                          nn_dir: Path | str | None = None):
    """Absolute 2004-2009 mean MOC state [lev, lat] of the training model.

    Prefer ``moc_baseline.npz`` saved with the trained model; fall back to
    the historical-simulation mean computed in Stage 06. Return None when neither
    is available, as for absolute-target runs."""
    from .config import (
        SCIENTIFIC_CONFIG,
        cmip_interim_dir,
    )

    def _validated(baseline: np.ndarray):
        ref = np.asarray(baseline).T                       # [lat,lev]->[lev,lat]
        if ref.shape != (sigma2.size, lat.size):
            return None
        return ref

    if nn_dir is not None:
        saved = Path(nn_dir) / "moc_baseline.npz"
        if saved.is_file():
            with np.load(saved) as fh:
                ref = _validated(fh["MOC_baseline_mean"])
            if ref is not None:
                return ref

    inputs = SCIENTIFIC_CONFIG.get("training_input_datasets",
                                   ["ACCESS_historical"])
    historical = next((e for e in inputs if str(e).endswith("_historical")),
                      str(inputs[0]))
    cache = cmip_interim_dir(historical) / "MOC_TimeMean_2004_2009.npz"
    if not cache.is_file():
        return None
    with np.load(cache) as fh:
        return _validated(fh["MOC_2004_2009_mean"])


def load_training_moc_baseline(
    nn_dir: Path | str,
    lat: np.ndarray,
    rho2: np.ndarray,
    expected_period: tuple[int, int] = (2004, 2009),
) -> np.ndarray:
    """Load the training model's 2004-2009 mean MOC saved with the networks.

    The networks reconstruct anomalies relative to this reference state; it
    is used to locate the cell cores and to set the sign convention of the
    abyssal cells, not as an estimate of the real-ocean mean. The file stores
    ``MOC_baseline_mean`` as [lat, lev]; it is returned as [lev, lat].
    """
    path = require_file(Path(nn_dir) / "moc_baseline.npz", "Training MOC baseline")
    with np.load(path) as data:
        period = np.asarray(data["moc_baseline_period"]).astype(int).reshape(-1)
        baseline = np.asarray(data["MOC_baseline_mean"]).T
    if not np.array_equal(period, np.asarray(expected_period, dtype=int)):
        raise RuntimeError(f"{path}: baseline period is {period.tolist()}, "
                           f"expected {list(expected_period)}")
    expected_shape = (np.asarray(rho2).size, np.asarray(lat).size)
    if baseline.shape != expected_shape:
        raise RuntimeError(f"{path}: baseline shape {baseline.shape}, "
                           f"expected [lev, lat] = {expected_shape}")
    return baseline


def grand_centered_mapping_spread(
    errors: np.ndarray,
    *,
    sample_ndim: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample SD about one pooled mean over the first `sample_ndim` axes.

    Returns `(spread, mean_error, nobs, unpriced)`. Incomplete cells are
    unpriced and have NaN spread and mean error.
    """
    values = np.asarray(errors, dtype=float)
    if sample_ndim < 1 or sample_ndim >= values.ndim:
        raise ValueError(
            f"sample_ndim must be in [1, {values.ndim - 1}], got "
            f"{sample_ndim}"
        )
    sample_axes = tuple(range(sample_ndim))
    expected_nobs = int(np.prod(values.shape[:sample_ndim]))
    if expected_nobs < 2:
        raise ValueError("mapping-error spread requires at least two values")

    finite = np.isfinite(values)
    nobs = finite.sum(axis=sample_axes).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_error = np.where(finite, values, 0.0).sum(axis=sample_axes) / nobs
        expanded_mean = mean_error[(None,) * sample_ndim]
        ss = np.where(finite, (values - expanded_mean) ** 2, 0.0).sum(
            axis=sample_axes
        )
        spread = np.sqrt(ss / np.maximum(nobs - 1, 1))
    unpriced = nobs != expected_nobs
    spread = np.asarray(spread, dtype=float)
    mean_error = np.asarray(mean_error, dtype=float)
    spread[unpriced] = np.nan
    mean_error[unpriced] = np.nan
    return spread, mean_error, nobs, unpriced


def branch_centered_mapping_spread(
    errors: np.ndarray,
    *,
    sample_ndim: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pooled spread after removing each branch's mean.

    Axis 0 indexes branches; subsequent sample axes index months. The spread
    uses `ddof=n_branches`, while `mean_error` is the pooled signed bias.
    Returns `(spread, mean_error, nobs, unpriced)`; incomplete cells are NaN.
    """
    values = np.asarray(errors, dtype=float)
    if sample_ndim < 2 or sample_ndim >= values.ndim:
        raise ValueError(
            f"sample_ndim must be in [2, {values.ndim - 1}] for branch "
            f"centering, got {sample_ndim}"
        )
    n_branches = int(values.shape[0])
    within_axes = tuple(range(1, sample_ndim))
    sample_axes = tuple(range(sample_ndim))
    expected_nobs = int(np.prod(values.shape[:sample_ndim]))
    if expected_nobs - n_branches < 1:
        raise ValueError(
            "branch-centered spread needs more pooled values than branches"
        )

    finite = np.isfinite(values)
    nobs = finite.sum(axis=sample_axes).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        branch_nobs = finite.sum(axis=within_axes, keepdims=True)
        branch_mean = (
            np.where(finite, values, 0.0).sum(axis=within_axes, keepdims=True)
            / branch_nobs
        )
        residual = values - branch_mean
        ss = np.where(finite, residual**2, 0.0).sum(axis=sample_axes)
        spread = np.sqrt(ss / np.maximum(nobs - n_branches, 1))
        mean_error = np.where(finite, values, 0.0).sum(axis=sample_axes) / nobs
    unpriced = nobs != expected_nobs
    spread = np.asarray(spread, dtype=float)
    mean_error = np.asarray(mean_error, dtype=float)
    spread[unpriced] = np.nan
    mean_error[unpriced] = np.nan
    return spread, mean_error, nobs, unpriced


def validate_transfer_case_arrays(
    truth: np.ndarray,
    pred: np.ndarray,
    n_realizations: int,
    spatial_shape: tuple[int, int],
    label: str,
) -> int:
    """Validate one held-out transfer case and return months per member."""
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    if truth.shape != pred.shape:
        raise RuntimeError(
            f"{label}: y shape {truth.shape} != y_pred shape {pred.shape}"
        )
    if truth.ndim != 3 or truth.shape[1:] != tuple(spatial_shape):
        raise RuntimeError(
            f"{label}: expected [time, lev, lat] with spatial shape "
            f"{tuple(spatial_shape)}, got {truth.shape}"
        )
    if n_realizations < 1:
        raise RuntimeError(f"{label}: n_realizations must be positive")
    if truth.shape[0] == 0:
        raise RuntimeError(f"{label}: transfer case contains no monthly samples")
    if truth.shape[0] % n_realizations:
        raise RuntimeError(
            f"{label}: {truth.shape[0]} months cannot be divided into "
            f"{n_realizations} realization blocks"
        )
    return truth.shape[0] // n_realizations


def trim_and_validate_member_predictions(
    members: np.ndarray,
    ensemble_mean: np.ndarray,
    edge_months: int,
    atol: float = 1e-3,
) -> np.ndarray:
    """Trim stage-14 member fields and verify their saved ensemble mean."""
    members = np.asarray(members, dtype=float)
    ensemble_mean = np.asarray(ensemble_mean)
    if members.ndim != 4:
        raise RuntimeError(
            "Pred_RealWorld pred_yz_members must be [member, time, lev, lat], "
            f"got {members.shape}"
        )
    if members.shape[0] < 2:
        raise RuntimeError(
            "At least two reconstruction members are required to estimate sigma_eps"
        )
    if edge_months < 0:
        raise RuntimeError("edge_months must be non-negative")
    if edge_months and members.shape[1] <= 2 * edge_months:
        raise RuntimeError(
            f"Pred_RealWorld has only {members.shape[1]} months; cannot trim "
            f"{edge_months} months from each edge"
        )
    trimmed = (
        members[:, edge_months:-edge_months] if edge_months else members
    )
    if trimmed.shape[1:] != ensemble_mean.shape:
        raise RuntimeError(
            "Trimmed pred_yz_members has shape "
            f"{trimmed.shape[1:]}; expected [time, lev, lat] "
            f"{ensemble_mean.shape}"
        )
    if not np.allclose(
        trimmed.mean(axis=0), ensemble_mean,
        atol=atol, rtol=0.0, equal_nan=True,
    ):
        raise RuntimeError(
            "Pred_RealWorld member mean does not match the saved ensemble mean"
        )
    return trimmed


def exact_common_month_indices(
    month_axes,
    labels: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return the exact common monthly axis and row indices for each input.

    Every input axis must itself be contiguous and unique.  The returned
    indices select the same ordered calendar months from every input, rather
    than merely truncating all arrays to a shared length.  This distinction is
    important when one satellite product starts later or ends earlier than
    the others.
    """
    axes = list(month_axes)
    if not axes:
        raise RuntimeError("at least one monthly coordinate is required")
    if labels is None:
        labels = tuple(f"monthly coordinate {i}" for i in range(len(axes)))
    if len(labels) != len(axes):
        raise RuntimeError(
            f"{len(labels)} labels supplied for {len(axes)} monthly coordinates"
        )

    normalized = tuple(
        normalize_month_axis(axis, np.asarray(axis).size, str(label))
        for axis, label in zip(axes, labels)
    )
    empty = [str(label) for axis, label in zip(normalized, labels) if not axis.size]
    if empty:
        raise RuntimeError(
            "monthly coordinate is empty: " + ", ".join(empty)
        )

    start = max(axis[0] for axis in normalized)
    stop = min(axis[-1] for axis in normalized)
    if stop < start:
        ranges = ", ".join(
            f"{label}={axis[0]}..{axis[-1]}"
            for axis, label in zip(normalized, labels)
        )
        raise RuntimeError(f"monthly coordinates do not overlap: {ranges}")
    common = np.arange(start, stop + np.timedelta64(1, "M"),
                       dtype="datetime64[M]")

    indices = []
    for axis, label in zip(normalized, labels):
        first = int(np.searchsorted(axis, common[0]))
        take = np.arange(first, first + common.size, dtype=np.int64)
        if take[-1] >= axis.size or not np.array_equal(axis[take], common):
            raise RuntimeError(
                f"{label}: failed to select the exact common monthly coordinate"
            )
        indices.append(take)
    return common, tuple(indices)


def load_real_world(
    nn_dir: Path | str,
    rmse_scenario: str | None = "MRI_SSP245",
    edge_months: int = 12,
    t0_years: float = 2002 + 4 / 12,
    rmse_variant: str = "debiased",
    file_stem: str = "Pred_RealWorld",
) -> RealWorldReconstruction:
    """Load a reconstruction, trim filter edges, and optionally add RMSE.

    With `rmse_scenario=None`, `total_uncertainty` contains network spread
    only. Other monthly uncertainty components must be added separately.
    Quadrature with scenario RMSE assumes independent error components.
    """
    nn_dir = Path(nn_dir)
    if edge_months < 0:
        raise ValueError("edge_months must be non-negative")
    data = load_npz_or_mat(nn_dir / "RealWorld" / file_stem)
    edge_slice = slice(edge_months, -edge_months) if edge_months else slice(None)
    pred = np.asarray(data["pred_yz"])[edge_slice]
    epistemic = np.asarray(data["pred_yz_std"])[edge_slice]
    lat = np.asarray(data["lat"]).squeeze()
    rho2 = np.asarray(data["rho2"]).squeeze()
    sigma2 = rho2 - 1000.0 if rho2[0] > 1000 else rho2

    if rmse_scenario is None:
        # Caller must assemble the full uncertainty envelope before display.
        total = epistemic.copy()
    else:
        scenario = _scenario_rmse(nn_dir, rmse_scenario, rmse_variant)
        total = np.sqrt(scenario[None, :, :] ** 2 + epistemic**2)

    full_n_t = np.asarray(data["pred_yz"]).shape[0]
    n_t = pred.shape[0]
    time_month_full = None
    if "time_month" in data:
        time_month_full = normalize_month_axis(
            data["time_month"], full_n_t,
            f"{nn_dir / 'RealWorld' / file_stem} time_month",
        )

    if "t_year" in data:
        t_years_full = np.asarray(data["t_year"], dtype=float).reshape(-1)
        if t_years_full.size != full_n_t:
            raise ValueError("Pred_RealWorld t_year length does not match pred_yz")
        if not np.isfinite(t_years_full).all():
            raise ValueError("Pred_RealWorld t_year contains non-finite values")
        if time_month_full is not None:
            expected_t = decimal_year(time_month_full)
            if not np.allclose(t_years_full, expected_t, rtol=0.0, atol=1e-10):
                first = int(np.flatnonzero(
                    ~np.isclose(t_years_full, expected_t, rtol=0.0, atol=1e-10)
                )[0])
                raise ValueError(
                    "Pred_RealWorld t_year disagrees with time_month at "
                    f"row {first}: {t_years_full[first]} vs {expected_t[first]}"
                )
        t_years = t_years_full[edge_slice]
    elif time_month_full is not None:
        t_years = decimal_year(time_month_full)[edge_slice]
    else:
        # Without saved dates, the untrimmed series starts at t0 (April 2002).
        t_years = t0_years + (edge_months + np.arange(n_t)) / 12
    time_month = (
        time_month_full[edge_slice] if time_month_full is not None else None
    )

    def text_scalar(key: str) -> str | None:
        if key not in data:
            return None
        value = np.asarray(data[key]).squeeze()
        if value.ndim == 0:
            item = value.item()
            return item.decode() if isinstance(item, bytes) else str(item)
        return "".join(value.astype(str).ravel()).strip()

    def text_values(key: str) -> tuple[str, ...]:
        """Decode a saved string vector without concatenating its entries."""
        if key not in data:
            return ()
        value = np.asarray(data[key]).squeeze()
        if value.ndim == 0:
            item = value.item()
            text = item.decode() if isinstance(item, bytes) else str(item)
            return (text.strip(),) if text.strip() else ()
        values = []
        for item in value.reshape(-1):
            if isinstance(item, np.ndarray):
                text = "".join(item.astype(str).reshape(-1))
            elif isinstance(item, bytes):
                text = item.decode()
            else:
                text = str(item)
            if text.strip():
                values.append(text.strip())
        return tuple(values)

    def positive_int_scalar(key: str) -> int | None:
        if key not in data:
            return None
        value = np.asarray(data[key]).squeeze()
        if value.ndim != 0:
            raise ValueError(f"Pred_RealWorld {key} must be a scalar")
        result = int(value.item())
        if result < 1:
            raise ValueError(f"Pred_RealWorld {key} must be positive, got {result}")
        return result

    wind_baseline_years = None
    if "wind_baseline_years" in data:
        years = np.asarray(data["wind_baseline_years"]).astype(int).reshape(-1)
        if years.size == 2:
            wind_baseline_years = (int(years[0]), int(years[1]))

    moc_baseline_years = None
    if "moc_baseline_years" in data:
        years = np.asarray(data["moc_baseline_years"]).astype(int).reshape(-1)
        if years.size == 2:
            moc_baseline_years = (int(years[0]), int(years[1]))

    moc_convention = text_scalar("moc_convention")

    # Locate cores on the absolute mean state. Use the saved anomaly baseline
    # where available, then the trained-model or Stage-06 baseline; otherwise
    # use the reconstruction mean for absolute-target products.
    core_ref = None
    if "moc_baseline" in data:
        saved_baseline = np.asarray(data["moc_baseline"]).squeeze()
        if saved_baseline.shape != (sigma2.size, lat.size):
            raise ValueError(
                "Pred_RealWorld moc_baseline has shape "
                f"{saved_baseline.shape}; expected {(sigma2.size, lat.size)}")
        core_ref = saved_baseline
    if moc_convention == "anomaly_2004_2009":
        if core_ref is None:
            raise ValueError(
                "Pred_RealWorld anomaly is missing its [lev, lat] "
                "moc_baseline; rerun 14_reconstruct_real_world.py"
            )
        if moc_baseline_years != tuple(BASELINE_YEARS):
            raise ValueError(
                "Pred_RealWorld anomaly baseline period is "
                f"{moc_baseline_years}, expected {tuple(BASELINE_YEARS)}"
            )
    if core_ref is None:
        core_ref = training_moc_baseline(lat, sigma2, nn_dir=nn_dir)
    cores = locate_cell_cores(
        core_ref if core_ref is not None else pred.mean(axis=0), lat, sigma2)
    return RealWorldReconstruction(
        pred=pred, epistemic=epistemic, total_uncertainty=total,
        t_years=t_years, lat=lat, sigma2=sigma2, cores=cores,
        moc_baseline=core_ref,
        time_month=time_month,
        wind_convention=text_scalar("wind_convention"),
        wind_baseline_years=wind_baseline_years,
        moc_convention=moc_convention,
        moc_baseline_years=moc_baseline_years,
        input_covariates=text_values("input_covariates"),
        input_sources=text_values("input_sources"),
        input_source_files=text_values("input_source_files"),
        input_baseline_specs=text_values("input_baseline_specs"),
        trained_on=text_scalar("trained_on"),
        training_experiment=text_scalar("training_experiment"),
        ensemble_num_folds=positive_int_scalar("ensemble_num_folds"),
        ensemble_members_per_fold=positive_int_scalar(
            "ensemble_members_per_fold"
        ),
        run_id=text_scalar("run_id"),
        training_source_run_id=text_scalar("training_source_run_id"),
        cmip_dataset_id=text_scalar("cmip_dataset_id"),
        satellite_dataset_id=text_scalar("satellite_dataset_id"),
    )
