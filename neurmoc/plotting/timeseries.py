"""MOC time-series panels: uncertainty band, GRACE gap, trend annotation."""

from __future__ import annotations

import numpy as np

from .style import COLORS, TEXT_BBOX

#: Missing calendar months and their plotting edges. Monthly points use the
#: project's ``year + month_number / 12`` convention, so the half-month edges
#: shade exactly July 2017 through May 2018 without covering valid June 2018.
GRACE_GAP_MONTHS = ("2017-07", "2018-05")
GRACE_GAP = (2017 + 6.5 / 12, 2018 + 5.5 / 12)


def shade_gap(ax, gap: tuple[float, float] = GRACE_GAP, alpha: float = 0.25) -> None:
    """Shade an observational gap over the full panel height."""
    ax.axvspan(gap[0], gap[1], color=COLORS["gap"], alpha=alpha, lw=0, zorder=0)


def emptier_side(
    t: np.ndarray,
    series,
    *,
    y_band: tuple[float, float],
    x_frac: float,
    prefer: str = "right",
) -> str:
    """Choose the less occupied side of an annotation band.

    `y_band` and `x_frac` are normalized panel coordinates. Ties use `prefer`.
    """
    if prefer not in {"left", "right"}:
        raise ValueError(f"prefer must be 'left' or 'right', got {prefer!r}")
    stacked = [np.asarray(s, dtype=float).reshape(-1) for s in series]
    t = np.asarray(t, dtype=float).reshape(-1)
    if not stacked or any(s.size != t.size for s in stacked):
        return prefer

    y = np.concatenate(stacked)
    x = np.tile(t, len(stacked))
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return prefer
    x, y = x[finite], y[finite]

    x_span = np.ptp(x)
    y_span = np.ptp(y)
    if x_span <= 0 or y_span <= 0:
        return prefer
    xn = (x - x.min()) / x_span
    yn = (y - y.min()) / y_span

    in_band = (yn >= y_band[0]) & (yn <= y_band[1])
    left = int((in_band & (xn <= x_frac)).sum())
    right = int((in_band & (xn >= 1.0 - x_frac)).sum())
    if left == right:
        return prefer
    return "left" if left < right else "right"


def moc_timeseries(
    ax,
    t: np.ndarray,
    y: np.ndarray,
    uncertainty: np.ndarray | None = None,
    *,
    color: str = COLORS["prediction"],
    band_alpha: float = 0.18,
    lw: float = 1.3,
    label: str | None = None,
    zorder: int = 3,
):
    """Line with an optional +/- uncertainty band; returns the line handle."""
    if uncertainty is not None:
        ax.fill_between(t, y - uncertainty, y + uncertainty,
                        color=color, alpha=band_alpha, lw=0, zorder=zorder - 1)
    (line,) = ax.plot(t, y, "-", color=color, lw=lw, label=label, zorder=zorder)
    return line


def style_timeseries_axis(ax, xlim=None, ylim=None, xlabel: str = "Year",
                          ylabel: str = r"$\Psi$ (Sv)") -> None:
    """Shared cosmetics of every MOC time-series panel."""
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.minorticks_on()
    ax.grid(axis="y", alpha=0.15, lw=0.5)
    ax.tick_params(which="both", top=False, right=False)


def trend_annotation(
    ax,
    t: np.ndarray,
    slope: float,
    intercept: float,
    significant: bool,
    ci: tuple[float, float] | None = None,
    *,
    sign: float = 1.0,
    text_xy: tuple[float, float] = (0.98, 0.93),
    ha: str = "right",
    lw: float = 1.0,
):
    """Add a trend label and, when significant, a dashed trend line.

    `ci` is the slope +/- 2-sigma interval. `sign=-1` flips the displayed
    slope for panels plotted as negative overturning strength.
    """
    slope_plot = sign * slope
    color = COLORS["trend_up"] if slope_plot > 0 else COLORS["trend_down"]
    if significant:
        line = sign * (np.asarray(t) * slope + intercept)
        ax.plot(t, line, "--", color=color, lw=lw, zorder=4)
    if ci is None:
        if not significant:
            ax.text(*text_xy, "Trend not significant", ha=ha, va="top",
                    transform=ax.transAxes, color="0.35", bbox=TEXT_BBOX)
        return
    unc = abs(float(ci[1]) - float(ci[0])) / 2.0        # = 2 sigma_total
    label = (rf"Trend = {slope_plot:+.3f} $\pm$ {unc:.3f} "
             rf"Sv yr$^{{-1}}$ (2$\sigma$" + ("" if significant else ", n.s.")
             + ")")
    # Right alignment avoids overlap with a left-side panel title.
    ax.text(*text_xy, label, ha=ha, va="top", transform=ax.transAxes,
            color=color if significant else "0.35", bbox=TEXT_BBOX)


def latitude_title(ax, text: str, xy: tuple[float, float] = (0.02, 0.94),
                   ha: str = "left") -> None:
    """Bold in-panel annotation such as 'AMOC at 26.5 N'.

    Pass ``xy=(0.98, 0.94), ha="right"`` to move it to the upper right when
    the panel's own curves crowd the upper left.
    """
    ax.text(*xy, text, transform=ax.transAxes, fontweight="bold",
            va="top", ha=ha, color="0.1", bbox=TEXT_BBOX)


def lat_label(lat: float) -> str:
    """'26.5 degrees N' / '20.5 degrees S' style label."""
    hemi = "N" if lat >= 0 else "S"
    return f"{abs(lat):.1f}\N{DEGREE SIGN}{hemi}"
