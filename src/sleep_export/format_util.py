"""Display formatters shared across web and PDF rendering."""

from __future__ import annotations

_MIN_PER_HOUR: int = 60


def format_minutes_hm(minutes: float) -> str:
    """Format a count of minutes as 'XXh YYm' (or 'YY min' below 60 min).

    Used everywhere we display a *duration* or *period* (mean total sleep,
    dominant period, SDs). Rates like midpoint drift (min/day) keep their
    decimal form because the precision matters at sub-minute resolution.

    Examples:
        format_minutes_hm(432)   -> '7h 12m'
        format_minutes_hm(1471)  -> '24h 31m'
        format_minutes_hm(45.6)  -> '46 min'
        format_minutes_hm(-30)   -> '-30 min'
    """
    sign = "-" if minutes < 0 else ""
    abs_min = abs(minutes)
    if abs_min < _MIN_PER_HOUR:
        return f"{sign}{abs_min:.0f} min"
    hours = int(abs_min // _MIN_PER_HOUR)
    mins = round(abs_min - hours * _MIN_PER_HOUR)
    if mins == _MIN_PER_HOUR:  # carry on rounding edge
        hours += 1
        mins = 0
    return f"{sign}{hours}h {mins:02d}m"
