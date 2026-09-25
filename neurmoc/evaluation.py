"""Shared paths, baseline settings, and Stage-10 evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RUN_ROOT, SCIENTIFIC_CONFIG
from .config import FIGURE_DIR as FIGURE_DIR
from .filtering import std_by_realization
from .io_utils import load_npz_or_mat
from .moc_utils import (
    CellCores,
    extract_at_cores,
    find_nearest_index,
    locate_cell_cores,
)
from .naming import training_config_from_scientific
from .results import load_reference_moc, load_scenario_eval

# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------
FROZEN_TRAINING = training_config_from_scientific(SCIENTIFIC_CONFIG)
BASELINE_EXPERIMENT = FROZEN_TRAINING.experiment_name()
COVARIATES_ALL = FROZEN_TRAINING.covariates.input_var

# Held-out MRI cases used to estimate mapping error.
# The stacked option reuses five member lineages across three scenarios.
SIGMA_MAP_ESTIMATORS = {
    "mri_ssp245": {"MRI_SSP245": (5, 2015.0)},
    "mri_ssp126_245_370_stacked": {
        "MRI_SSP126": (5, 2015.0),
        "MRI_SSP245": (5, 2015.0),
        "MRI_SSP370": (5, 2015.0),
    },
}
SIGMA_MAP_ESTIMATOR_CHOICES = tuple(SIGMA_MAP_ESTIMATORS)
DEFAULT_SIGMA_MAP_ESTIMATOR = "mri_ssp126_245_370_stacked"

# Monthly mapping spread can remove each branch mean or one pooled mean.
# Trend errors are pooled across branches.
SIGMA_MAP_MONTHLY_CENTERINGS = {
    "branch": "per_branch_record_mean_across_selected_branch_window_months",
    "grand": "grand_across_selected_branch_window_months",
}
SIGMA_MAP_MONTHLY_CENTERING_CHOICES = tuple(SIGMA_MAP_MONTHLY_CENTERINGS)
DEFAULT_SIGMA_MAP_MONTHLY_CENTERING = "branch"


def sigma_map_monthly_centering_label(
    centering: str = DEFAULT_SIGMA_MAP_MONTHLY_CENTERING,
) -> str:
    """Return the saved provenance label for a monthly centering mode."""
    try:
        return SIGMA_MAP_MONTHLY_CENTERINGS[centering]
    except KeyError as exc:
        raise ValueError(
            f"unknown monthly sigma_map centering {centering!r}; choose "
            f"from {SIGMA_MAP_MONTHLY_CENTERING_CHOICES}"
        ) from exc


def sigma_map_monthly_expected_ddof(centering: str,
                                    n_branch_windows: int) -> int:
    """Means removed by a monthly centering mode: 1 grand, or one per branch."""
    sigma_map_monthly_centering_label(centering)   # validate the mode
    return 1 if centering == "grand" else int(n_branch_windows)


def sigma_map_monthly_retains_mean_shifts(centering: str) -> bool:
    """Whether centering retains between-branch mean shifts."""
    sigma_map_monthly_centering_label(centering)   # validate the mode
    return centering == "grand"


def sigma_map_transfer_cases(
    estimator: str = DEFAULT_SIGMA_MAP_ESTIMATOR,
) -> dict[str, tuple[int, float]]:
    """Return the MRI-ESM2.0 cases used by a mapping-error estimator."""
    try:
        return dict(SIGMA_MAP_ESTIMATORS[estimator])
    except KeyError as exc:
        raise ValueError(
            f"unknown sigma_map estimator {estimator!r}; choose from "
            f"{SIGMA_MAP_ESTIMATOR_CHOICES}"
        ) from exc


# Alias for the default transfer-case registry.
TREND_BUDGET_TRANSFER_CASES = sigma_map_transfer_cases()

TRAINING_DATASET = str(
    SCIENTIFIC_CONFIG.get("training_dataset", "ACCESS_hist+SSP585")
)
# Trained networks and their model-test evaluations.
PERF_ROOT = RUN_ROOT / TRAINING_DATASET
# Save observational products under the active analysis run.
REALWORLD_ROOT = RUN_ROOT / TRAINING_DATASET

def model_dir(root: Path = PERF_ROOT, experiment: str = BASELINE_EXPERIMENT,
              covariates: str = COVARIATES_ALL) -> Path:
    return root / experiment / covariates


@dataclass
class PerformanceCase:
    """Truth, predictions, and reference circulation for one test case."""

    mc: "object"                  # ScenarioEval (stage-10 truth/prediction)
    cores: CellCores              # from the reference diagnosed MOC
    moc_mean: np.ndarray          # [lev, lat] reference time-mean MOC
    moc_std: np.ndarray           # [lev, lat]
    t_years: np.ndarray           # decimal years of the test period

    @property
    def lat(self):
        return self.mc.lat

    @property
    def sigma2(self):
        return self.mc.sigma2

    def negative_mask(self, require_variability: bool = False) -> np.ndarray:
        """Cells with negative time-mean MOC (optionally only where variable)."""
        mask = self.moc_mean < 0
        if require_variability:
            mask &= self.moc_std > 0.5
        return mask

    def strengths(self, which: str = "mid"):
        """Truth, prediction, and optional uncertainty at a cell core."""
        idx = self.cores.mid_index if which == "mid" else self.cores.abyssal_index
        return (extract_at_cores(self.mc.truth, idx),
                extract_at_cores(self.mc.pred, idx),
                extract_at_cores(self.mc.uncertainty, idx)
                if self.mc.uncertainty is not None else None)

    def lat_index(self, lat_value: float) -> int:
        return find_nearest_index(self.lat, lat_value)


def load_performance_case(cmip_tag: str, nn_dir: Path | None = None,
                          year0: float = 2015.0,
                          reference_dir: Path | None = None,
                          n_realizations: int = 5) -> PerformanceCase:
    """Assemble the stage-10 skill data + reference MOC of one scenario.

    `reference_dir` defaults to `nn_dir`; pass the all-covariates model dir
    when evaluating reduced-input models (whose folders hold no Pred file).
    `n_realizations` must match the evaluated case for correct member slicing.
    """
    nn_dir = nn_dir or model_dir()
    mc = load_scenario_eval(nn_dir, cmip_tag, n_realizations=n_realizations)
    reference = load_reference_moc(reference_dir or nn_dir, cmip_tag)
    moc_mean = reference.mean(axis=0)
    # Recover the absolute mean state from the stored baseline when model
    # targets are anomalies; absolute targets already contain that mean.
    try:
        grid = load_npz_or_mat(Path(reference_dir or nn_dir)
                               / f"TestR2_{cmip_tag}")
        if "moc_baseline" in grid:
            moc_mean = np.asarray(grid["moc_baseline"]) + moc_mean
    except FileNotFoundError:
        pass
    cores = locate_cell_cores(moc_mean, mc.lat, mc.sigma2)
    n = mc.samples_per_realization
    t_years = year0 + (1 + np.arange(n)) / 12
    # Compute variability within members, excluding between-member offsets.
    return PerformanceCase(mc=mc, cores=cores, moc_mean=moc_mean,
                           moc_std=std_by_realization(reference,
                                                      n_realizations),
                           t_years=t_years)
