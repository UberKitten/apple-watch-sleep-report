"""Plotly figure dicts for the web UI.

Returns plain dictionaries that the frontend feeds directly to
`Plotly.newPlot(...)`. No plotly Python dep needed; the schema is
hand-crafted.
"""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from sleep_export import analysis
from sleep_export.format_util import format_minutes_hm

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from numpy.typing import NDArray

    from sleep_export.models import NightSummary, SleepRecord

# Discrete colour stops for the actogram (0=awake, 1=in bed, 2=asleep).
ACTOGRAM_COLORSCALE: list[list[float | str]] = [
    [0.0, "rgba(255,255,255,0)"],
    [0.499, "rgba(255,255,255,0)"],
    [0.5, "rgba(147,197,253,0.55)"],  # tailwind blue-300 @ 55%
    [0.999, "rgba(147,197,253,0.55)"],
    [1.0, "rgba(30,58,138,0.95)"],  # tailwind blue-900
]


def _label_dates(num_days: int, anchor: date) -> list[str]:
    return [(anchor + timedelta(days=i)).isoformat() for i in range(num_days)]


def _hour_ticks_48h() -> tuple[list[int], list[str]]:
    tickvals = [h * 60 for h in range(0, 49, 6)]  # every 6 hours
    ticktext = [f"{(h % 24):02d}:00" for h in range(0, 49, 6)]
    return tickvals, ticktext


def actogram(records: Sequence[SleepRecord]) -> dict[str, Any]:
    """Double-plotted actogram as a Plotly heatmap dict.

    Each row N contains day N's 1440 minutes in cols [0, 1440) and
    day N+1's 1440 minutes in cols [1440, 2880). The visual effect is
    each calendar day appearing twice, on adjacent rows — the standard
    chronobiology presentation that makes free-running drift jump out
    visually as a diagonal stripe.
    """
    grid, anchor = analysis.build_state_grid(records)
    if grid.shape[0] == 0:
        return {"data": [], "layout": {"title": "Actogram (no data)"}}
    num_days = grid.shape[0]
    # Pad with a zero row at the end so we can index grid[i+1] safely.
    padded = np.vstack([grid, np.zeros((1, grid.shape[1]), dtype=np.int8)])
    double = np.concatenate([padded[:-1], padded[1:]], axis=1)  # (num_days, 2880)
    z = double.tolist()
    y_labels = _label_dates(num_days, anchor)
    tickvals, ticktext = _hour_ticks_48h()
    return {
        "data": [
            {
                "type": "heatmap",
                "z": z,
                "y": y_labels,
                "colorscale": ACTOGRAM_COLORSCALE,
                "zmin": 0,
                "zmax": 2,
                "showscale": False,
                "hovertemplate": "%{y} %{x}<extra></extra>",
            }
        ],
        "layout": {
            "title": {"text": "Double-plotted actogram"},
            "xaxis": {
                "title": {"text": "Time of day (48-hour double plot)"},
                "tickvals": tickvals,
                "ticktext": ticktext,
                "range": [0, 2880],
            },
            "yaxis": {
                "title": {"text": "Date"},
                "autorange": "reversed",  # newest at bottom (UCSD convention)
                "type": "category",  # treat date strings as discrete labels, not epoch dates
                "tickmode": "auto",
                "nticks": 30,
            },
            "height": max(400, min(900, 320 + num_days * 1)),
            "margin": {"l": 100, "r": 30, "t": 60, "b": 40},
            "shapes": [
                {
                    "type": "line",
                    "x0": 1440, "x1": 1440,
                    "y0": -0.5, "y1": num_days - 0.5,
                    "yref": "y",
                    "xref": "x",
                    "line": {"color": "rgba(0,0,0,0.3)", "width": 1, "dash": "dot"},
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# midpoint drift


def midpoint_drift(nights: Sequence[NightSummary]) -> dict[str, Any]:
    """Scatter of nightly sleep midpoint vs date, with regression overlay."""
    if len(nights) < 2:  # noqa: PLR2004
        return {"data": [], "layout": {"title": "Midpoint drift (need ≥2 nights)"}}
    dates = [n.sleep_date.isoformat() for n in nights]
    midpoint_hours = [n.midpoint_minutes_from_midnight / 60.0 for n in nights]
    slope = analysis.midpoint_drift_min_per_day(nights)
    r2 = _r_squared(nights)
    ordinals = np.array([n.sleep_date.toordinal() for n in nights], dtype=np.float64)
    midpoints = np.array(midpoint_hours, dtype=np.float64)
    if np.var(ordinals) > 0:
        coeffs = np.polyfit(ordinals, midpoints, 1)
        line_y = (coeffs[0] * ordinals + coeffs[1]).tolist()
    else:
        line_y = [float(midpoints.mean())] * len(midpoints)
    annotation_color = "#d97706" if abs(slope) > 30 else "#1e40af"  # noqa: PLR2004
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": dates,
                "y": midpoint_hours,
                "name": "Nightly midpoint",
                "marker": {"size": 6, "color": "#1d4ed8"},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": dates,
                "y": line_y,
                "name": f"Regression: {slope:+.1f} min/day",
                "line": {"color": annotation_color, "width": 2},
            },
        ],
        "layout": {
            "title": {"text": f"Midpoint drift: {slope:+.1f} min/day (R²={r2:.2f})"},
            "xaxis": {"title": {"text": "Date"}},
            "yaxis": {"title": {"text": "Midpoint of sleep (hours past midnight)"}},
            "showlegend": True,
        },
    }


def periodogram(signal: NDArray[np.int8]) -> dict[str, Any]:
    """Sokolove-Bushell chi-square periodogram as a Plotly figure dict.

    x-axis: candidate period in hours (22 to 28).
    y-axis: chi-square Q_p statistic.
    Peak marked, 24h reference line drawn, 99.9% W-H threshold overlaid.
    """
    periods_min, q_values = analysis.chi_square_periodogram(signal)
    if periods_min.size == 0:
        return {"data": [], "layout": {"title": "Periodogram (signal too short)"}}
    periods_h = (periods_min / 60.0).tolist()
    q_list = q_values.tolist()
    peak_idx = int(q_values.argmax())
    peak_period_min = float(periods_min[peak_idx])
    peak_period_h = peak_period_min / 60.0
    peak_period_label = format_minutes_hm(peak_period_min)
    peak_q = float(q_values[peak_idx])
    threshold_at_peak = analysis.chi_square_threshold(peak_period_min)
    is_significant = peak_q > threshold_at_peak
    y_max = max(float(q_values.max()), 1.0)
    sig_text = (
        f"significant (Q={peak_q:.0f} > {threshold_at_peak:.0f})"
        if is_significant
        else f"below 99.9% threshold ({threshold_at_peak:.0f}); drift smears alignment"
    )
    traces: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": periods_h,
            "y": q_list,
            "name": "Q_p",
            "line": {"color": "#1d4ed8", "width": 2},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "x": [peak_period_h],
            "y": [peak_q],
            "name": "Peak",
            "marker": {"color": "#dc2626", "size": 10, "symbol": "circle"},
            "showlegend": False,
        },
    ]
    # Only draw the threshold line if it's roughly in scale; otherwise it
    # squashes the Q curve to invisibility on drifty data.
    if threshold_at_peak < y_max * 2:
        threshold = [analysis.chi_square_threshold(p) for p in periods_min.tolist()]
        traces.insert(1, {
            "type": "scatter",
            "mode": "lines",
            "x": periods_h,
            "y": threshold,
            "name": "99.9% threshold",
            "line": {"color": "#9ca3af", "dash": "dot", "width": 1},
        })
    return {
        "data": traces,
        "layout": {
            "title": {
                "text": f"Chi-square periodogram — peak at {peak_period_label} ({sig_text})",
            },
            "xaxis": {
                "title": {"text": "Candidate period (hours)"},
                "dtick": 0.5,
            },
            "yaxis": {"title": {"text": "Chi-square Q_p"}, "range": [0, y_max * 1.15]},
            "shapes": [
                {
                    "type": "line",
                    "x0": 24.0, "x1": 24.0,
                    "y0": 0, "y1": y_max * 1.10,
                    "yref": "y", "xref": "x",
                    "line": {"color": "#9ca3af", "width": 1, "dash": "dash"},
                }
            ],
            "annotations": [
                {
                    "x": 24.0, "y": y_max * 1.10,
                    "xref": "x", "yref": "y",
                    "text": "24h reference",
                    "showarrow": False,
                    "yshift": 10,
                    "font": {"color": "#6b7280", "size": 11},
                }
            ],
            "showlegend": True,
        },
    }


def _r_squared(nights: Sequence[NightSummary]) -> float:
    if len(nights) < 2:  # noqa: PLR2004
        return 0.0
    x = np.array([n.sleep_date.toordinal() for n in nights], dtype=np.float64)
    y = np.array([n.midpoint_minutes_from_midnight for n in nights], dtype=np.float64)
    if np.var(x) == 0 or np.var(y) == 0:
        return 0.0
    coeffs = np.polyfit(x, y, 1)
    y_pred = coeffs[0] * x + coeffs[1]
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# polar bedtime / waketime


def polar_bedwake(nights: Sequence[NightSummary]) -> dict[str, Any]:
    """Two side-by-side polar histograms: bedtime (cool) + waketime (warm)."""
    if not nights:
        return {"data": [], "layout": {"title": "Polar (no data)"}}
    bed_hours = [
        ((n.night_start_local.hour + n.night_start_local.minute / 60.0) % 24) for n in nights
    ]
    wake_hours = [
        ((n.night_end_local.hour + n.night_end_local.minute / 60.0) % 24) for n in nights
    ]
    bed_counts, _ = np.histogram(bed_hours, bins=np.arange(25))
    wake_counts, _ = np.histogram(wake_hours, bins=np.arange(25))
    theta = [h * 15 for h in range(24)]  # 360/24 = 15 deg/hour
    return {
        "data": [
            {
                "type": "barpolar",
                "r": bed_counts.tolist(),
                "theta": theta,
                "name": "Bedtime",
                "marker": {"color": "#1d4ed8"},  # cool
            },
            {
                "type": "barpolar",
                "r": wake_counts.tolist(),
                "theta": theta,
                "name": "Waketime",
                "marker": {"color": "#ea580c"},  # warm
            },
        ],
        "layout": {
            "title": {"text": "Bedtime / waketime distribution"},
            "polar": {
                "angularaxis": {
                    "tickmode": "array",
                    "tickvals": [h * 15 for h in range(0, 24, 3)],
                    "ticktext": [f"{h:02d}" for h in range(0, 24, 3)],
                    "direction": "clockwise",
                    "rotation": 90,  # 00:00 at the top
                },
            },
            "showlegend": True,
        },
    }


# ---------------------------------------------------------------------------
# weekly heatmap


def weekly_heatmap(records: Sequence[SleepRecord]) -> dict[str, Any]:
    """Day-of-week by hour-of-day fraction-asleep heatmap.

    Reveals "anchor sleep" patterns vs scattered schedules. Mondays
    are y=0; the y-axis labels use abbreviations.
    """
    grid, anchor = analysis.build_state_grid(records)
    if grid.shape[0] == 0:
        return {"data": [], "layout": {"title": "Weekly heatmap (no data)"}}
    asleep = (grid == 2).astype(np.float64)  # noqa: PLR2004 — '2' = asleep marker
    # (n_days, 1440) -> (n_days, 24, 60) -> mean across minutes -> (n_days, 24)
    hourly = asleep.reshape(grid.shape[0], 24, 60).mean(axis=2)
    dows = np.array(
        [(anchor + timedelta(days=i)).weekday() for i in range(grid.shape[0])]
    )
    by_dow = np.zeros((7, 24), dtype=np.float64)
    counts = np.zeros(7, dtype=np.float64)
    for i in range(grid.shape[0]):
        by_dow[dows[i]] += hourly[i]
        counts[dows[i]] += 1
    counts_safe = np.where(counts == 0, 1, counts)
    by_dow /= counts_safe[:, None]
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "data": [
            {
                "type": "heatmap",
                "z": by_dow.tolist(),
                "x": list(range(24)),
                "y": labels,
                "colorscale": "Blues",
                "zmin": 0,
                "zmax": 1,
                "colorbar": {"title": {"text": "P(asleep)"}, "tickformat": ".0%"},
            }
        ],
        "layout": {
            "title": {"text": "Weekly sleep pattern"},
            "xaxis": {"title": {"text": "Hour of day"}, "dtick": 2},
            "yaxis": {"title": {"text": "Day of week"}, "autorange": "reversed"},
        },
    }
