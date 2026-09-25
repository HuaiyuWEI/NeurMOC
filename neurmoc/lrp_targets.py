"""LRP target selection and output naming.

A target is one latitude-density cell of the reconstructed MOC: either the
mid-depth or abyssal cell core at a given latitude, or an explicit density
level. The helpers here locate that cell on the trained output grid and build
the output directory names used by the LRP scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import BASINMASK_DIR
from .moc_utils import find_nearest_index, locate_cell_cores

REFERENCE_GRID = BASINMASK_DIR / "ACCESS_target_MOC_grid.npz"

OBP_SOURCE_CHOICES = {"GRACE", "GRACE_CSR"}
SSH_SOURCE_CHOICES = {"DUACS", "NASASSH"}
WIND_SOURCE_CHOICES = {"CCMP", "ERA5"}


@dataclass(frozen=True)
class TargetRequest:
    """A requested LRP target."""

    latitude: float
    core: str = "mid"
    sigma2: float | None = None
    sign: float = 1.0


@dataclass(frozen=True)
class ResolvedTarget:
    """The target cell on the trained MOC grid."""

    latitude_index: int
    level_index: int
    flat_grid_index: int
    valid_output_index: int
    latitude: float
    sigma2: float
    mode: str
    baseline_value: float


def _require_finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


def resolve_target(
    lat: np.ndarray,
    sigma2: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray | None,
    target_latitude: float,
    target_core: str,
    target_sigma2: float | None,
) -> ResolvedTarget:
    """Locate a target cell: a cell core at a latitude, or an explicit level."""
    target_latitude = _require_finite(target_latitude, "target latitude")
    lat = np.asarray(lat, dtype=float).reshape(-1)
    sigma2 = np.asarray(sigma2, dtype=float).reshape(-1)
    mask = np.asarray(mask).reshape(-1).astype(bool)
    lat_index = find_nearest_index(lat, target_latitude)

    if target_sigma2 is None:
        if baseline is None:
            raise RuntimeError(
                "Core-based LRP targets require the training MOC baseline; "
                "use an explicit sigma2 otherwise"
            )
        cores = locate_cell_cores(np.asarray(baseline), lat, sigma2)
        if target_core == "mid":
            level_index = int(cores.mid_index[lat_index])
            core_sigma2 = cores.mid_sigma2[lat_index]
        elif target_core == "abyssal":
            level_index = int(cores.abyssal_index[lat_index])
            core_sigma2 = cores.abyssal_sigma2[lat_index]
        else:
            raise ValueError(
                f"target core must be 'mid' or 'abyssal', got {target_core!r}"
            )
        if not np.isfinite(core_sigma2):
            raise ValueError(
                f"No valid {target_core} core is defined at {lat[lat_index]:g} degrees"
            )
        mode = f"{target_core}_core"
    else:
        target_sigma2 = _require_finite(target_sigma2, "target sigma2")
        level_index = find_nearest_index(sigma2, target_sigma2)
        mode = "explicit_cell"

    n_lat = lat.size
    if mask.shape != (sigma2.size * n_lat,):
        raise ValueError(
            f"Psi mask has shape {mask.shape}; expected {(sigma2.size * n_lat,)}"
        )
    flat_grid_index = level_index * n_lat + lat_index
    valid_grid_indices = np.flatnonzero(mask)
    matches = np.flatnonzero(valid_grid_indices == flat_grid_index)
    if matches.size != 1:
        raise ValueError(
            f"Requested target (sigma2={sigma2[level_index]:g}, "
            f"lat={lat[lat_index]:g}) is outside the trained Psi mask"
        )
    baseline_value = (
        float(np.asarray(baseline)[level_index, lat_index])
        if baseline is not None
        else np.nan
    )
    return ResolvedTarget(
        latitude_index=lat_index,
        level_index=level_index,
        flat_grid_index=flat_grid_index,
        valid_output_index=int(matches[0]),
        latitude=float(lat[lat_index]),
        sigma2=float(sigma2[level_index]),
        mode=mode,
        baseline_value=baseline_value,
    )


def number_slug(value: float, *, signed: bool = False) -> str:
    """Format a coordinate for a directory name, e.g. -45.5 -> 'm45p5'."""
    value = _require_finite(value, "slug value")
    text = f"{value:+.4f}" if signed else f"{value:.4f}"
    text = text.rstrip("0").rstrip(".")
    return text.replace("+", "p").replace("-", "m").replace(".", "p")


def target_slug(target: ResolvedTarget, sign: float) -> str:
    """Return the output directory name for one target."""
    sign = _require_finite(sign, "target sign")
    if sign not in (-1.0, 1.0):
        raise ValueError(f"target sign must be +1 or -1, got {sign!r}")
    slug = (
        f"{target.mode}_lat{number_slug(target.latitude, signed=True)}_"
        f"sigma2_{number_slug(target.sigma2)}"
    )
    return slug + ("_negPsi" if sign < 0 else "_Psi")


def epsilon_slug(epsilon: float) -> str:
    """Return the directory name for an LRP epsilon value (0 = LRP-0)."""
    epsilon = _require_finite(epsilon, "LRP epsilon")
    if epsilon < 0:
        raise ValueError(f"LRP epsilon must be >= 0, got {epsilon!r}")
    if epsilon == 0:
        return "lrp0_z_rule"
    mantissa, exponent = format(epsilon, ".11e").lower().split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    exponent_value = int(exponent)
    exponent_text = f"{exponent_value:+03d}".replace("+", "")
    text = f"{mantissa}e{exponent_text}"
    return f"epsilon_{text}"


def source_suffix(obp: str, ssh: str, wind: str) -> str:
    """Return the file suffix for a satellite-product combination."""
    if obp not in OBP_SOURCE_CHOICES:
        raise ValueError(f"unsupported OBP source {obp!r}")
    if ssh not in SSH_SOURCE_CHOICES:
        raise ValueError(f"unsupported SSH source {ssh!r}")
    if wind not in WIND_SOURCE_CHOICES:
        raise ValueError(f"unsupported wind source {wind!r}")
    suffix = ""
    if obp != "GRACE":
        suffix += f"_obp{obp.removeprefix('GRACE_')}"
    if ssh != "DUACS":
        suffix += f"_ssh{ssh}"
    if wind == "ERA5":
        suffix += "_ERA5wind"
    return suffix


def input_source_names(obp: str, ssh: str, wind: str) -> list[str]:
    source_suffix(obp, ssh, wind)  # validate all three values
    return [f"obp_{obp}", f"ssh_{ssh}", f"uas_{wind}"]
