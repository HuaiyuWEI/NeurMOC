"""Out-of-sample model-test cases used by the cross-model evaluation and LRP."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import N_TEST_REALIZATIONS, SCIENTIFIC_CONFIG, RUN_ROOT


def safe_case_component(value: str, label: str) -> str:
    """Check that a case or dataset tag is a single path component."""
    value = str(value).strip()
    if not value:
        raise ValueError(f"{label} must be non-empty")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ValueError(
            f"{label} must be one dataset name, not a path: {value!r}"
        )
    return value


def evaluation_case_registry() -> dict[str, tuple[str, str, int]]:
    """Return {case name: (output tag, realization tag, number of realizations)}."""
    registered = {
        "MRI_SSP245": ("MRI_SSP245", "_r1_r5", 5),
        "MRI_SSP126": ("MRI_SSP126", "_r1_r5", 5),
        "MRI_SSP370": ("MRI_SSP370", "_r1_r5", 5),
    }
    external = safe_case_component(
        str(SCIENTIFIC_CONFIG.get("external_test_dataset", "ACCESS_SSP245")),
        "external_test_dataset",
    )
    external_spec = ("SSP245", "_r36_r40", int(N_TEST_REALIZATIONS))
    return {external: external_spec, **registered}


def registered_case_specs() -> tuple[tuple[str, str, str, int], ...]:
    """Return the registered cases as (name, output tag, realization tag, count)."""
    return tuple(
        (name, out_tag, realization_tag, n_realizations)
        for name, (out_tag, realization_tag, n_realizations)
        in evaluation_case_registry().items()
    )


def resolve_case_spec(spec: str) -> tuple[str, str, str, int]:
    """Resolve a registered case name or an explicit NAME:OUT:RLZ:N case."""
    raw = str(spec).strip()
    if not raw:
        raise ValueError("case specification must be non-empty")
    parts = raw.split(":")
    registry = evaluation_case_registry()
    if len(parts) == 1:
        name = safe_case_component(parts[0], "case name")
        if name not in registry:
            raise ValueError(
                f"unknown case {name!r}; use the explicit "
                "NAME:OUT_TAG:RLZ_TAG:N_REALIZATIONS form"
            )
        out_tag, realization_tag, n_realizations = registry[name]
        return name, out_tag, realization_tag, n_realizations
    if len(parts) != 4:
        raise ValueError(
            "an explicit case must contain four colon-separated fields: "
            "NAME:OUT_TAG:RLZ_TAG:N_REALIZATIONS"
        )
    name = safe_case_component(parts[0], "case name")
    out_tag = safe_case_component(parts[1], "case output tag")
    realization_tag = safe_case_component(parts[2], "realization tag")
    n_realizations = int(parts[3])
    if n_realizations <= 0:
        raise ValueError("case realization count must be positive")
    return name, out_tag, realization_tag, n_realizations


def stage06_lpf_key(lpf_months: int) -> str:
    """Return the model-ready array suffix for the training low-pass filter."""
    months = int(lpf_months)
    if months == 24:
        return "_LPF_ALL"
    if months == 0:
        return "_ALL"
    raise ValueError(f"model-test data are prepared for lpf_months 0 or 24, not {months}")


def contiguous_realization_groups(
    realization_index: np.ndarray,
    expected_count: int,
) -> list[np.ndarray]:
    """Return the sample indices of each realization in a concatenated record."""
    labels = np.asarray(realization_index)
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], labels.size]
    if len(starts) != int(expected_count):
        raise ValueError(
            f"realization_index contains {len(starts)} contiguous blocks; "
            f"expected {expected_count}"
        )
    return [np.arange(start, stop, dtype=np.int64) for start, stop in zip(starts, stops)]


def training_dataset_root(trained_on: str) -> Path:
    """Return the model-ready data folder of the training dataset."""
    return RUN_ROOT / safe_case_component(trained_on, "training dataset name")
