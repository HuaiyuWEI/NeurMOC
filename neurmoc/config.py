"""Data locations and shared settings.

Machine-specific paths are read from ``configs/paths.local.json`` (see
``configs/paths.example.json``) or from ``NEURMOC_<KEY>`` environment
variables. Scientific settings are read from ``configs/neurmoc_v1.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = Path(os.environ.get(
    "NEURMOC_PATHS_FILE", PROJECT_ROOT / "configs" / "paths.local.json"))


def _load_local_paths() -> dict[str, object]:
    if not PATHS_FILE.exists():
        return {}
    with PATHS_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


_LOCAL_PATHS = _load_local_paths()


def _configured_path(key: str, default: str | Path) -> Path:
    """Read a path from the environment, the local paths file, or the default."""
    value = os.environ.get(f"NEURMOC_{key.upper()}", _LOCAL_PATHS.get(key, default))
    return Path(value).expanduser()


def _configured_bool(key: str, default: bool) -> bool:
    """Read a boolean from the environment, the local paths file, or the default."""
    value = os.environ.get(f"NEURMOC_{key.upper()}", _LOCAL_PATHS.get(key, default))
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


SCIENTIFIC_CONFIG_FILE = _configured_path(
    "scientific_config_file", PROJECT_ROOT / "configs" / "neurmoc_v1.json")
with SCIENTIFIC_CONFIG_FILE.open(encoding="utf-8") as _fh:
    SCIENTIFIC_CONFIG = json.load(_fh)

# ---------------------------------------------------------------------------
# Data folders
# ---------------------------------------------------------------------------
DATA_ROOT = _configured_path("data_root", PROJECT_ROOT / "data")
RAW_ROOT = DATA_ROOT / "raw"
REFERENCE_ROOT = DATA_ROOT / "reference"
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"
RESULTS_ROOT = DATA_ROOT / "results"
CACHE_ROOT = DATA_ROOT / "cache"

ACTIVE_RUN_ID = str(SCIENTIFIC_CONFIG.get("run_id", "neurmoc_v1"))
CMIP_DATASET_ID = str(SCIENTIFIC_CONFIG["dataset_id"])
SATELLITE_DATASET_ID = str(SCIENTIFIC_CONFIG["satellite_dataset_id"])
INSITU_DATASET_ID = str(SCIENTIFIC_CONFIG["insitu_dataset_id"])
#: File and variable suffix for the area-weighted mascon products.
MASCON_VERSION = str(SCIENTIFIC_CONFIG["mascon_version"])

#: MOC target convention: "anomaly_2004_2009" subtracts the January
#: 2004-December 2009 ensemble mean of each dataset; "absolute" keeps the full MOC.
MOC_CONVENTION = str(SCIENTIFIC_CONFIG["moc_convention"])

#: Ocean fields use cell-area weights; other fields use cosine-latitude weights.
MASCON_WEIGHTING = str(SCIENTIFIC_CONFIG["mascon_weighting"])


def mascon_var(prefix: str) -> str:
    """Versioned mascon variable name, e.g. mascon_var('obp') -> 'obp_mascon_V7'."""
    return f"{prefix}_mascon_{MASCON_VERSION}"


# Raw input data. The file names record the product versions used.
CMIP_RAW_ROOT = RAW_ROOT / "cmip6"
GRACE_MASCON_NC = (
    RAW_ROOT
    / "satellite"
    / "OBP"
    / "grace-jpl"
    / "GRCTellus.JPL.200204_202605.GLO.RL06.3M.MSCNv04CRI.nc"
)
CSR_MASCON_NC = (
    RAW_ROOT
    / "satellite"
    / "OBP"
    / "grace-csr"
    / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
)
#: NASA GSFC RL06 v2.0 mascons with ICE6G-D GIA removed and GAD restored
#: over ocean pixels (Loomis et al., 2019, doi:10.1007/s00190-019-01252-y).
#: The 0.5-degree grid and 2004-2009 reference match JPL. GSFC is used for the
#: RAPID comparison only and is excluded from the satellite-product spread.
GSFC_MASCON_NC = (
    RAW_ROOT
    / "satellite"
    / "OBP"
    / "grace-gsfc"
    / "gsfc.glb_.200204_202603_rl06v2.0_obp-ice6gd_halfdegree.nc"
)
DUACS_DAILY_DIR = RAW_ROOT / "satellite" / "SSH" / "duacs" / "daily"
#: NASA-SSH v1.1: weekly 0.5-degree sea-surface-height anomalies to ~71.2 degrees S,
#: used as the alternative to DUACS.
NASASSH_GRID_DIR = RAW_ROOT / "satellite" / "SSH" / "nasa-ssh" / "v11"
CCMP_RAW_DIR = RAW_ROOT / "satellite" / "Wind" / "ccmp"
ERA5_DIR = RAW_ROOT / "satellite" / "Wind" / "era5"
#: Monthly 10-m winds; the most recent months are preliminary ERA5T data.
ERA5_WIND_NC = ERA5_DIR / "ERA5_Wind_2002_2026.nc"
RAPID_RAW_DIR = RAW_ROOT / "insitu" / "rapid"
OSNAP_RAW_DIR = RAW_ROOT / "insitu" / "osnap"
ECCO_V4R3_RAW_DIR = RAW_ROOT / "ecco" / "v4r3"

# Overturning streamfunction computed from the ECCO v4r3 state estimate.
ECCO_V4R3_OVERTURNING_DIR = ECCO_V4R3_RAW_DIR / "myproducts_monthly"

# Derived data.
CMIP_INTERIM_ROOT = INTERIM_ROOT / ACTIVE_RUN_ID / "cmip6"
WIND_CALIBRATION_DIR = REFERENCE_ROOT / "wind_calibration" / ACTIVE_RUN_ID
CMIP_PROCESSED_ROOT = PROCESSED_ROOT / "cmip6" / CMIP_DATASET_ID
CMIP_ROOT = CMIP_PROCESSED_ROOT
OBS_MASCON_ROOT = PROCESSED_ROOT / "observations" / SATELLITE_DATASET_ID
INSITU_ROOT = PROCESSED_ROOT / "observations" / INSITU_DATASET_ID
RAPID_DIR = INSITU_ROOT / "rapid"
OSNAP_DIR = INSITU_ROOT / "osnap"
ECCO_V4R3_OBS_DIR = INSITU_ROOT / "ecco_v4r3"

#: Trained networks, evaluations, and real-world reconstructions of this run.
RUN_ROOT = RESULTS_ROOT / ACTIVE_RUN_ID
FIGURE_DIR = RUN_ROOT / "figures"
BASINMASK_DIR = REFERENCE_ROOT / "grids"
CM4_GR_DIR = CMIP_RAW_ROOT / "GFDL_PIcontrol" / "CM4"
DUACS_CACHE_DIR = CACHE_ROOT / "duacs"

# Batch runs can close figures with ``NEURMOC_KEEP_FIGURES_OPEN=0``.
KEEP_FIGURES_OPEN = _configured_bool("keep_figures_open", True)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
#: Target latitude-density grid of the interpolated MOC.
N_LEVS = 18
N_LATS = 140

#: GRACE anomalies are relative to the Jan 2004 - Dec 2009 mean.
BASELINE_YEARS = tuple(SCIENTIFIC_CONFIG.get("baseline_years", (2004, 2009)))

#: Realizations used for training and for in-model testing.
_TRAIN_RANGE = SCIENTIFIC_CONFIG.get("training_realizations", (1, 35))
_TEST_RANGE = SCIENTIFIC_CONFIG.get("external_test_realizations", (36, 40))
TRAIN_REALIZATIONS = range(int(_TRAIN_RANGE[0]), int(_TRAIN_RANGE[1]) + 1)
TEST_REALIZATIONS = range(int(_TEST_RANGE[0]), int(_TEST_RANGE[1]) + 1)
N_TEST_REALIZATIONS = len(TEST_REALIZATIONS)

#: Boussinesq reference density (kg m^-3) used to convert mass to volume transport.
RHO_CONST = 1035.0


# ---------------------------------------------------------------------------
# Low-pass filter settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterConfig:
    """Butterworth low-pass filter settings for monthly time series."""

    cutoff_months: int = 24     # cut-off period
    order: int = 5
    sampling_rate: float = 1.0  # samples per month
    padding_months: int | None = None  # reflect padding; default = 2 * cutoff

    @property
    def cutoff_freq(self) -> float:
        return 1.0 / self.cutoff_months

    @property
    def padding_length(self) -> int:
        if self.padding_months is None:
            return 2 * self.cutoff_months
        return self.padding_months


#: 2-year filter applied to all training inputs/targets before training.
LPF_TRAIN = FilterConfig(
    cutoff_months=int(SCIENTIFIC_CONFIG.get("training_lowpass_months", 24)),
    order=int(SCIENTIFIC_CONFIG.get("training_lowpass_order", 5)),
)

#: 10-year filter applied *after* training when assessing decadal skill.
LPF_DECADAL = FilterConfig(cutoff_months=120)

#: 2-year filter for the (shorter) observational records, with reduced padding.
LPF_OBS = FilterConfig(cutoff_months=24, padding_months=24)


def cmip_raw_dir(cmip_name: str) -> Path:
    """Original CMIP6 files for one experiment."""
    return CMIP_RAW_ROOT / cmip_name


def cmip_interim_dir(cmip_name: str) -> Path:
    """Mascon-averaged inputs and regridded MOC (scripts 03-05) for one experiment."""
    return CMIP_INTERIM_ROOT / cmip_name


def experiment_dir(cmip_name: str) -> Path:
    """Model-ready data (scripts 06-07) for one experiment."""
    return CMIP_PROCESSED_ROOT / cmip_name


def results_dir(cmip_name: str, lpf_tag: str = "_LPF2Year") -> Path:
    """Folder holding the trained networks for one training dataset.

    The experiment folder name below it already contains the filter tag, so
    `lpf_tag` does not change the path.
    """
    return RUN_ROOT / cmip_name
