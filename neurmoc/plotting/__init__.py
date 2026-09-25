"""Publication-quality plotting toolkit (Nature-journal conventions)."""

from .sections import draw_section, format_lat_axis, section_row  # noqa: F401
from .style import (  # noqa: F401
    CMAP_AMPLITUDE,
    CMAP_DIVERGING,
    CMAP_R2,
    COLORS,
    CORRELATION_DISPLAY_LIMITS,
    DOUBLE_COL,
    ONE_HALF_COL,
    SINGLE_COL,
    apply_style,
    finish_figure,
    save_figure,
)
from .timeseries import (  # noqa: F401
    GRACE_GAP,
    GRACE_GAP_MONTHS,
    moc_timeseries,
    shade_gap,
    trend_annotation,
)
