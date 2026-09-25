"""Satellite-product combinations used for the product-spread uncertainty.

Each network input has a primary product and one alternative:
JPL or CSR ocean bottom pressure, DUACS or NASA-SSH sea surface height, and
CCMP or ERA5 wind. The satellite-product uncertainty is the spread of the
reconstructions obtained with all combinations of these products.
"""

from __future__ import annotations

import json
from itertools import product

from .config import SCIENTIFIC_CONFIG, mascon_var
from .evaluation import TREND_BUDGET_TRANSFER_CASES

OBP, SSH, UAS = mascon_var("obp"), mascon_var("ssh"), mascon_var("uas")

# (source name, file suffix); the first entry on each axis is the primary product.
OBP_SOURCES = ((None, ""), ("GRACE_CSR", "_obpCSR"),
               ("GRACE_GSFC", "_obpGSFC"))
SSH_SOURCES = ((None, ""), ("NASASSH", "_sshNASASSH"))
WIND_SOURCES = ((None, ""), ("ERA5", "_ERA5wind"))

# GSFC is used only for the RAPID comparison, not for the product spread.
DEFAULT_PRODUCT_SPREAD_INCLUDE_GSFC = bool(
    SCIENTIFIC_CONFIG.get("product_spread_include_gsfc", False)
)


def uncertainty_combination_steps(
    covariates: str,
    include_gsfc: bool = False,
) -> list[tuple[object, object, object, str]]:
    """Return (obp, ssh, wind, suffix) for every product combination."""
    used = set(covariates.split("+"))
    obp_axis = OBP_SOURCES if include_gsfc else OBP_SOURCES[:2]
    axes = (
        obp_axis if OBP in used else obp_axis[:1],
        SSH_SOURCES if SSH in used else SSH_SOURCES[:1],
        WIND_SOURCES if UAS in used else WIND_SOURCES[:1],
    )
    return [
        (obp, ssh, wind, f"{obp_tag}{ssh_tag}{wind_tag}")
        for (obp, obp_tag), (ssh, ssh_tag), (wind, wind_tag)
        in product(*axes)
    ]


def uncertainty_product_registry(
    covariates: str,
    include_gsfc: bool = False,
) -> list[tuple[str, str, str, str]]:
    """Return the ordered product combinations with explicit source names."""
    used = set(covariates.split("+"))
    return [
        (
            (obp or "GRACE") if OBP in used else "",
            (ssh or "DUACS") if SSH in used else "",
            (wind or "CCMP") if UAS in used else "",
            tag,
        )
        for obp, ssh, wind, tag in uncertainty_combination_steps(
            covariates, include_gsfc=include_gsfc
        )
    ]


def product_spread_definition(include_gsfc: bool = False) -> str:
    """Describe the satellite-product spread for output metadata."""
    gsfc = "included" if include_gsfc else "excluded"
    return (
        "sample standard deviation (ddof=1) across the satellite-product "
        f"combinations, for monthly values and fitted trends; GSFC OBP {gsfc}"
    )


def transfer_case_spec_json(
    transfer_cases: dict[str, tuple[int, float]] | None = None,
) -> str:
    """Describe the MRI-ESM2.0 cases used for the mapping uncertainty as JSON."""
    if transfer_cases is None:
        transfer_cases = TREND_BUDGET_TRANSFER_CASES
    return json.dumps(
        {
            tag: {
                "n_realizations": int(n_realizations),
                "year0": float(year0),
            }
            for tag, (n_realizations, year0) in
            transfer_cases.items()
        },
        sort_keys=True,
        separators=(",", ":"),
    )
