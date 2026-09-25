"""Shared figure style and color palettes."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

from ..config import KEEP_FIGURES_OPEN

MM = 1 / 25.4
SINGLE_COL = 89 * MM      # 3.50 in
ONE_HALF_COL = 136 * MM   # 5.35 in
DOUBLE_COL = 183 * MM     # 7.20 in

#: Okabe-Ito colorblind-safe palette + project-specific line colors.
COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    # semantic aliases used across the MOC figures
    "prediction": "#1A1A1A",     # NeurMOC / DBNN reconstruction
    "truth": "#D55E00",          # model truth / RAPID observations
    "ecco": "#0072B2",           # ECCO state estimate
    "trend_up": "#C23B22",       # significant positive trend
    "trend_down": "#1F77B4",     # significant negative trend
    "background": "#E6E6E6",     # NaN background of section plots
    "gap": "#8CE3F0",            # GRACE/GRACE-FO gap shading
}

#: pmkmp(40, 'Swtth') from Matteo Niccoli's "Perceptually improved
#: colormaps" MATLAB toolbox. Values extracted verbatim from pmkmp.m.
_SWTTH40 = [
    (1.000000, 0.539500, 1.000000),
    (1.000000, 0.485193, 1.000000),
    (0.998221, 0.429666, 1.000000),
    (0.892241, 0.372213, 1.000000),
    (0.731950, 0.311708, 1.000000),
    (0.572564, 0.246664, 1.000000),
    (0.409713, 0.174234, 1.000000),
    (0.218738, 0.075207, 1.000000),
    (0.001151, 0.082579, 0.988104),
    (0.166429, 0.172120, 1.000000),
    (0.220143, 0.321656, 1.000000),
    (0.220049, 0.452047, 1.000000),
    (0.196953, 0.572594, 1.000000),
    (0.161800, 0.686400, 1.000000),
    (0.114007, 0.796058, 1.000000),
    (0.029810, 0.905445, 0.999709),
    (0.000000, 0.971884, 0.946791),
    (0.000000, 0.916890, 0.774783),
    (0.000000, 0.851947, 0.609542),
    (0.000000, 0.784745, 0.464818),
    (0.000000, 0.716320, 0.355839),
    (0.000000, 0.647148, 0.295087),
    (0.000000, 0.574254, 0.261614),
    (0.000000, 0.490965, 0.197429),
    (0.000000, 0.463886, 0.166350),
    (0.000000, 0.541209, 0.220397),
    (0.000000, 0.607400, 0.220200),
    (0.015831, 0.653229, 0.051078),
    (0.277903, 0.717620, 0.000000),
    (0.467576, 0.784059, 0.000000),
    (0.645601, 0.845009, 0.000000),
    (0.822898, 0.900915, 0.000000),
    (0.926499, 0.884938, 0.000000),
    (0.930413, 0.761048, 0.000000),
    (0.930399, 0.638175, 0.000000),
    (0.923497, 0.509649, 0.000000),
    (0.905389, 0.371518, 0.000000),
    (0.870759, 0.211972, 0.000000),
    (0.820664, 0.000000, 0.058458),
    (0.760600, 0.000000, 0.176900),
]

#: Default colormaps (perceptually uniform / colorblind-considerate).
CMAP_R2 = mpl.colors.ListedColormap(_SWTTH40, name="swtth40")  # skill maps (0..1)
CMAP_AMPLITUDE = "Spectral_r"  # RMSE / STD / uncertainty (0..max)
CMAP_DIVERGING = "RdBu_r"    # trends, mean MOC (symmetric about 0)
#: Correlation values remain signed in calculations and saved products, but
#: correlation-map colors intentionally emphasize nonnegative skill.
CORRELATION_DISPLAY_LIMITS = (0.0, 1.0)


def apply_style(font_size: float = 7.0) -> None:
    """Set global rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update({
        # fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "legend.fontsize": font_size - 0.5,
        "mathtext.fontset": "dejavusans",
        # axes
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        # ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        # lines
        "lines.linewidth": 1.0,
        "axes.prop_cycle": mpl.cycler(color=[
            COLORS["black"], COLORS["vermilion"], COLORS["blue"],
            COLORS["green"], COLORS["purple"], COLORS["orange"],
        ]),
        # legend
        "legend.frameon": False,
        # hatching (significance masks)
        "hatch.linewidth": 0.4,
        # export: keep text editable in Illustrator/Inkscape
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 130,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save_figure(fig, path_stem, formats=("png",), dpi: int = 450) -> list[Path]:
    """Save a figure and apply the configured interactive/batch close policy.

    Figures remain open by default for interactive Spyder inspection. Set
    ``keep_figures_open`` to false in ``configs/paths.local.json`` (or set
    ``NEURMOC_KEEP_FIGURES_OPEN=0``) for batch runs that should release each
    figure after it is written.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = stem.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, format=fmt)
        written.append(target)
    print("saved:", ", ".join(str(p) for p in written))
    finish_figure(fig)
    return written


def finish_figure(fig) -> None:
    """Close ``fig`` only when the centralized policy requests it."""
    if not KEEP_FIGURES_OPEN:
        import matplotlib.pyplot as plt

        plt.close(fig)


#: Subtle white backing for in-panel text so labels stay legible over data.
TEXT_BBOX = dict(facecolor="white", alpha=0.7, edgecolor="none",
                 boxstyle="square,pad=0.15")


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.05,
                fontsize: float | None = None) -> None:
    """Bold panel letter ('a', 'b', ...) at the top-left corner (Nature style)."""
    ax.text(x, y, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold",
            fontsize=fontsize or (mpl.rcParams["font.size"] + 1))
