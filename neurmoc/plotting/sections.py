"""Latitude-density plots for Southern Ocean and Atlantic MOC sections."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from .style import COLORS

SMOC_XLIM = (-74.0, -34.0)
AMOC_XLIM = (-34.0, 64.0)


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell boundaries of a 1-D center grid, matching pcolormesh's
    shading="nearest" convention (midpoints, end cells extrapolated)."""
    c = np.asarray(centers, dtype=float)
    edges = np.empty(c.size + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges


def section_ylim(sigma2: np.ndarray) -> tuple[float, float]:
    """Density limits excluding the deepest level from section plots."""
    dsig = np.median(np.diff(sigma2))
    return sigma2[0] - dsig / 2, sigma2[-1] - dsig / 2


def format_lat_axis(ax, lo: float, hi: float, tick_step: float = 5.0,
                    label_step: float = 10.0) -> None:
    """Degree ticks every `tick_step`, S/N labels every `label_step`."""
    start = np.ceil(lo / tick_step) * tick_step
    ticks = np.arange(start, hi + 0.1, tick_step)
    labels = []
    for v in ticks:
        if abs(v) % label_step == 0:
            if v == 0:
                labels.append("0\N{DEGREE SIGN}")
            elif v < 0:
                labels.append(f"{abs(v):.0f}\N{DEGREE SIGN}S")
            else:
                labels.append(f"{v:.0f}\N{DEGREE SIGN}N")
        else:
            labels.append("")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def draw_section(
    ax,
    field_zy: np.ndarray,
    lat: np.ndarray,
    sigma2: np.ndarray,
    *,
    cmap,
    vmin: float,
    vmax: float,
    xlim: tuple[float, float],
    negative_mask: np.ndarray | None = None,
    core_curves: tuple = (),
    hatch_mask: np.ndarray | None = None,
    hatch_pattern: str = "////",
    hatch_color: str = "black",
    dot_size: float = 1.2,
    lat_label_step: float = 10.0,
):
    """Draw one latitude-density panel onto `ax`; returns the mappable.

    `field_zy` is `[n_lev, n_lat]`; NaNs show the gray background.
    `negative_mask` (same shape) marks cells drawn with small white dots.
    `core_curves` are sigma2(lat) curves overlaid as dashed black lines.
    `hatch_mask` (same shape, boolean) is hatched (e.g. insignificant trend).
    """
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    mesh = ax.pcolormesh(
        lat, sigma2, np.ma.masked_invalid(field_zy),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="nearest", rasterized=True,
    )

    if negative_mask is not None:
        zz, yy = np.nonzero(negative_mask)
        ax.scatter(np.asarray(lat)[yy], np.asarray(sigma2)[zz], s=dot_size,
                   c="white", linewidths=0, zorder=3)

    if hatch_mask is not None:
        # Hatch masked data cells using their exact grid boundaries.
        shown = np.asarray(hatch_mask, dtype=bool) & np.isfinite(field_zy)
        xe, ye = _cell_edges(lat), _cell_edges(sigma2)
        verts = [
            ((xe[j], ye[k]), (xe[j + 1], ye[k]),
             (xe[j + 1], ye[k + 1]), (xe[j], ye[k + 1]))
            for k, j in zip(*np.nonzero(shown))
        ]
        if verts:
            # Hide cell outlines while retaining hatch strokes.
            ax.add_collection(PolyCollection(
                verts, facecolors="none", edgecolors=hatch_color,
                linewidths=0.0, hatch=hatch_pattern, zorder=2))

    for curve in core_curves:
        ax.plot(lat, curve, "--", color="black", lw=0.8, zorder=4)

    ax.set_xlim(xlim)
    ax.set_ylim(*section_ylim(sigma2))
    ax.invert_yaxis()
    ax.tick_params(which="both", top=False, right=False)
    ax.minorticks_on()
    format_lat_axis(ax, *xlim, label_step=lat_label_step)
    return mesh


def section_row(
    field_zy: np.ndarray,
    lat: np.ndarray,
    sigma2: np.ndarray,
    *,
    cmap,
    vmin: float,
    vmax: float,
    cbar_label: str = "",
    negative_mask: np.ndarray | None = None,
    core_curves_smoc: tuple = (),
    core_curves_amoc: tuple = (),
    hatch_mask: np.ndarray | None = None,
    hatch_pattern: str = "////",
    hatch_color: str = "black",
    axes: tuple | None = None,
    fig=None,
    subplot_spec=None,
    figsize: tuple[float, float] = (7.2, 2.4),
    add_colorbar: bool = True,
    show_ylabel: bool = True,
    panel_names: tuple[str, str] = ("SMOC", "AMOC"),
    yticks=np.arange(35.0, 37.6, 0.5),
    lat_label_step: float = 10.0,
):
    """SMOC | AMOC panel pair with a shared colorbar.

    Provide either nothing (a new figure is created), or `fig` +
    `subplot_spec` (a `GridSpec` slot, to stack several rows in one figure),
    or pre-made `axes`.
    Returns `(fig, (ax_smoc, ax_amoc), mappable)`.
    """
    width_ratios = (SMOC_XLIM[1] - SMOC_XLIM[0], AMOC_XLIM[1] - AMOC_XLIM[0])

    if axes is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if subplot_spec is None:
            gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=0.05,
                                  left=0.09, right=0.88, bottom=0.16, top=0.92)
        else:
            gs = subplot_spec.subgridspec(1, 2, width_ratios=width_ratios, wspace=0.05)
        ax_s = fig.add_subplot(gs[0])
        ax_a = fig.add_subplot(gs[1])
    else:
        ax_s, ax_a = axes
        fig = ax_s.figure

    common = dict(cmap=cmap, vmin=vmin, vmax=vmax, lat_label_step=lat_label_step,
                  negative_mask=negative_mask, hatch_mask=hatch_mask,
                  hatch_pattern=hatch_pattern, hatch_color=hatch_color)
    mesh = draw_section(ax_s, field_zy, lat, sigma2, xlim=SMOC_XLIM,
                        core_curves=core_curves_smoc, **common)
    draw_section(ax_a, field_zy, lat, sigma2, xlim=AMOC_XLIM,
                 core_curves=core_curves_amoc, **common)

    ax_s.set_yticks(yticks)
    ax_a.set_yticks(yticks)
    ax_a.set_yticklabels([])
    # set_yticks expands the view limits to cover out-of-range ticks;
    # Apply the shared density limit.
    lo, hi = section_ylim(sigma2)
    for ax in (ax_s, ax_a):
        ax.set_ylim(hi, lo)
    if show_ylabel:
        ax_s.set_ylabel(r"Density $\sigma_2$ (kg m$^{-3}$)")

    for ax, name in zip((ax_s, ax_a), panel_names):
        if name:
            ax.text(0.03, 0.96, name, transform=ax.transAxes,
                    fontweight="bold", va="top", color="0.1")

    if add_colorbar:
        cbar = fig.colorbar(mesh, ax=(ax_s, ax_a), pad=0.015, fraction=0.03, aspect=25)
        cbar.ax.tick_params(direction="out")
        cbar.outline.set_visible(False)
        if cbar_label:
            cbar.ax.set_title(cbar_label, fontsize=plt.rcParams["font.size"], pad=6)

    return fig, (ax_s, ax_a), mesh
