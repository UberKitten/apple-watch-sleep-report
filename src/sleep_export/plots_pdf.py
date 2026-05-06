"""Matplotlib renderers for the PDF report.

Each function returns a `matplotlib.figure.Figure` ready to be passed to
PdfPages.savefig(). Visual style is matched as closely as possible to
plots_web.py so the PDF and the on-screen view tell the same story.
"""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend

from datetime import timedelta
from typing import TYPE_CHECKING

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np

from sleep_export import analysis
from sleep_export.format_util import format_minutes_hm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from sleep_export.models import Analysis, NightSummary, SleepRecord


# Matches ACTOGRAM_COLORSCALE in plots_web.py:
ACTOGRAM_CMAP = LinearSegmentedColormap.from_list(
    "actogram",
    [
        (0.0, (1, 1, 1, 0)),  # awake -> transparent
        (0.5, (147 / 255, 197 / 255, 253 / 255, 0.55)),  # in-bed
        (1.0, (30 / 255, 58 / 255, 138 / 255, 0.95)),  # asleep
    ],
    N=256,
)


def actogram(records: Sequence[SleepRecord]) -> Figure:
    """Double-plotted actogram (matplotlib).

    Same data layout as plots_web.actogram: each row pairs day N (left
    half) with day N+1 (right half). Y axis newest-at-bottom.
    """
    grid, anchor = analysis.build_state_grid(records)
    if grid.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        _ = ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig
    num_days = grid.shape[0]
    padded = np.vstack([grid, np.zeros((1, grid.shape[1]), dtype=np.int8)])
    double = np.concatenate([padded[:-1], padded[1:]], axis=1).astype(np.float64) / 2.0
    height = max(4.0, min(20.0, num_days * 0.18))
    fig, ax = plt.subplots(figsize=(11, height))
    _ = ax.imshow(
        double,
        cmap=ACTOGRAM_CMAP,
        norm=Normalize(0, 1),
        aspect="auto",
        interpolation="none",
        extent=(0.0, 2880.0, num_days - 0.5, -0.5),
    )
    _ = ax.axvline(1440, color="#666", lw=0.5, ls=":")
    tickvals = list(range(0, 49, 6))
    ax.set_xticks([h * 60 for h in tickvals])
    ax.set_xticklabels([f"{(h % 24):02d}" for h in tickvals])
    ax.set_xlabel("Time of day (48h double plot)")
    ax.set_ylabel("Date")
    # Show every Nth y-tick so labels don't overlap
    step = max(1, num_days // 30)
    yticks = list(range(0, num_days, step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([(anchor + timedelta(days=i)).isoformat() for i in yticks])
    ax.set_title("Double-Plotted Actogram")
    fig.tight_layout()
    return fig


def midpoint_drift(nights: Sequence[NightSummary]) -> Figure:
    """Midpoint vs date scatter + regression line (matplotlib)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(nights) < 2:  # noqa: PLR2004
        _ = ax.text(0.5, 0.5, "Need ≥2 nights", ha="center", va="center")
        ax.set_axis_off()
        return fig
    # Use numeric date ordinals as the x-axis; matplotlib's date locator
    # would let us label them, but for type safety we just label by year.
    ordinals = np.array([n.sleep_date.toordinal() for n in nights], dtype=np.float64)
    midpoint_hours = np.array(
        [n.midpoint_minutes_from_midnight / 60.0 for n in nights], dtype=np.float64
    )
    slope = analysis.midpoint_drift_min_per_day(nights)
    if np.var(ordinals) > 0:
        coeffs = np.polyfit(ordinals, midpoint_hours, 1)
        line_y = coeffs[0] * ordinals + coeffs[1]
    else:
        line_y = np.full_like(midpoint_hours, midpoint_hours.mean())
    is_concerning = abs(slope) > 30  # noqa: PLR2004
    line_color = "#d97706" if is_concerning else "#1e40af"
    _ = ax.scatter(ordinals, midpoint_hours, s=18, color="#1d4ed8", zorder=2)
    _ = ax.plot(ordinals, line_y, color=line_color, lw=2, zorder=3,
                label=f"Slope: {slope:+.1f} min/day")
    # Re-label the x-axis with date strings at sparse tick positions
    n = len(nights)
    tick_idx = list(range(0, n, max(1, n // 6)))
    ax.set_xticks([ordinals[i] for i in tick_idx])
    ax.set_xticklabels([nights[i].sleep_date.isoformat() for i in tick_idx], rotation=30)
    ax.set_xlabel("Date")
    ax.set_ylabel("Midpoint of sleep (hours past midnight, unwrapped)")
    ax.set_title(f"Midpoint Drift — {slope:+.1f} min/day")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def periodogram(signal: NDArray[np.int8]) -> Figure:
    """Sokolove-Bushell chi-square periodogram (matplotlib)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    periods_min, q_values = analysis.chi_square_periodogram(signal)
    if periods_min.size == 0:
        _ = ax.text(0.5, 0.5, "Signal too short for periodogram",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    periods_h = periods_min / 60.0
    peak_idx = int(np.argmax(q_values))
    peak_period_h = float(periods_h[peak_idx])
    peak_q = float(q_values[peak_idx])
    threshold_at_peak = analysis.chi_square_threshold(periods_min[peak_idx])
    is_significant = peak_q > threshold_at_peak
    peak_period_label = format_minutes_hm(periods_min[peak_idx])
    sig_label = (
        f"significant (Q={peak_q:.0f} > {threshold_at_peak:.0f})"
        if is_significant
        else f"below 99.9% threshold ({threshold_at_peak:.0f}) — drift smears alignment"
    )
    _ = ax.plot(periods_h, q_values, color="#1d4ed8", lw=2, label="Q_p")
    _ = ax.axvline(24.0, color="#9ca3af", ls="--", lw=1, alpha=0.7,
                   label="24h reference")
    # Only draw the threshold line if it's within ~2x of the data range; on
    # drifty signals it's typically 10x higher and just squashes the Q curve.
    y_max = max(float(q_values.max()), 1.0)
    if threshold_at_peak < y_max * 2:
        threshold = np.array(
            [analysis.chi_square_threshold(p) for p in periods_min], dtype=np.float64
        )
        _ = ax.plot(periods_h, threshold, color="#9ca3af", ls=":", lw=1,
                    label="99.9% threshold")
    _ = ax.scatter([peak_period_h], [peak_q], color="#dc2626", s=80, zorder=5)
    _ = ax.annotate(
        f"Peak: {peak_period_label}",
        xy=(peak_period_h, peak_q),
        xytext=(peak_period_h + 0.3, peak_q),
        fontsize=10, color="#dc2626",
        arrowprops={"arrowstyle": "->", "color": "#dc2626"},
    )
    ax.set_xlabel("Candidate period (hours)")
    ax.set_ylabel("Chi-square Q_p")
    ax.set_title(
        f"Chi-square Periodogram (Sokolove-Bushell) — peak {peak_period_label}"
    )
    ax.text(0.99, 0.97, sig_label, transform=ax.transAxes,
            fontsize=9, color="#6b7280", ha="right", va="top")
    ax.set_ylim(0, y_max * 1.15)
    ax.legend(loc="upper left")
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    return fig


def polar_bedwake(nights: Sequence[NightSummary]) -> Figure:
    """Two side-by-side polar histograms: bedtime and waketime."""
    fig, (ax_bed, ax_wake) = plt.subplots(
        1, 2, figsize=(11, 5), subplot_kw={"projection": "polar"}
    )
    if not nights:
        for ax in (ax_bed, ax_wake):
            _ = ax.text(0, 0, "no data", ha="center")
        return fig
    bed_hours = np.array(
        [(n.night_start_local.hour + n.night_start_local.minute / 60.0) % 24 for n in nights]
    )
    wake_hours = np.array(
        [(n.night_end_local.hour + n.night_end_local.minute / 60.0) % 24 for n in nights]
    )
    bed_counts, _ = np.histogram(bed_hours, bins=np.arange(25))
    wake_counts, _ = np.histogram(wake_hours, bins=np.arange(25))
    theta = np.arange(24) * (2 * np.pi / 24)
    width = 2 * np.pi / 24
    for ax, counts, color, title in (
        (ax_bed, bed_counts, "#1d4ed8", "Bedtime"),
        (ax_wake, wake_counts, "#ea580c", "Waketime"),
    ):
        _ = ax.bar(theta, counts, width=width, color=color, alpha=0.75, edgecolor="white")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
        ax.set_title(title)
    fig.suptitle("Bedtime / Waketime Distribution")
    fig.tight_layout()
    return fig


def weekly_heatmap(records: Sequence[SleepRecord]) -> Figure:
    """Day-of-week by hour-of-day heatmap (matplotlib)."""
    grid, anchor = analysis.build_state_grid(records)
    fig, ax = plt.subplots(figsize=(10, 4))
    if grid.shape[0] == 0:
        _ = ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig
    asleep = (grid == 2).astype(np.float64)  # noqa: PLR2004 — '2' = asleep marker
    hourly = asleep.reshape(grid.shape[0], 24, 60).mean(axis=2)
    dows = np.array([(anchor + timedelta(days=i)).weekday() for i in range(grid.shape[0])])
    by_dow = np.zeros((7, 24), dtype=np.float64)
    counts = np.zeros(7, dtype=np.float64)
    for i in range(grid.shape[0]):
        by_dow[dows[i]] += hourly[i]
        counts[dows[i]] += 1
    counts_safe = np.where(counts == 0, 1, counts)
    by_dow /= counts_safe[:, None]
    im = ax.imshow(by_dow, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="upper")
    _ = fig.colorbar(im, ax=ax, label="P(asleep)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_title("Weekly Sleep Pattern")
    fig.tight_layout()
    return fig


def stats_summary(nights: Sequence[NightSummary], a: Analysis) -> Figure:
    """A figure containing a single summary-stats table (no chart).

    Wraps an `Analysis` object so we don't take many positional args.
    """
    fig, ax = plt.subplots(figsize=(8.27, 5.5))  # roughly A4 portrait width
    ax.set_axis_off()
    coverage = (
        f"{a.n_complete_nights} of {len(nights)} nights"
        + (f" ({a.n_complete_nights / len(nights) * 100:.0f}%)" if nights else "")
    )
    period_label = (
        f"{format_minutes_hm(a.dominant_period_min)}    (Q={a.dominant_period_q:.0f}; 24h 00m = entrained)"
        if a.dominant_period_min > 0
        else "n/a (not enough data)"
    )
    rows = [
        ("Nights with sufficient data", coverage),
        ("Mean total sleep", format_minutes_hm(a.mean_duration_min)),
        ("Dominant period (chi-square periodogram)", period_label),
        ("Midpoint drift (linear fit)", f"{a.midpoint_drift_min_per_day:+.2f} min/day"),
        ("Interdaily Stability (IS)", f"{a.is_value:.3f}    (1.0 = perfectly regular)"),
        ("Intradaily Variability (IV)", f"{a.iv_value:.3f}    (lower = less fragmented)"),
        ("Sleep Regularity Index (SRI)", f"{a.sri_value:+.1f}    (100 = identical day-to-day)"),
        ("Bedtime SD (circular, 24h clock)", format_minutes_hm(a.bedtime_sd_min)),
        ("Waketime SD (circular, 24h clock)", format_minutes_hm(a.waketime_sd_min)),
        ("Duration SD", format_minutes_hm(a.duration_sd_min)),
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.45, 0.5])
    table.auto_set_font_size(False)  # noqa: FBT003 — matplotlib API
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#cccccc")
    ax.set_title("Summary Statistics", pad=10, fontsize=14)
    fig.tight_layout()
    return fig


def nightly_table(nights: Sequence[NightSummary]) -> list[Figure]:
    """Render the nightly raw values as paginated A4 figures.

    Returns one Figure per page so the caller can `pdf.savefig(fig)`
    each one in sequence.
    """
    rows_per_page = 30
    pages: list[Figure] = []
    if not nights:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        _ = ax.text(0.5, 0.5, "No nights to display", ha="center", va="center")
        ax.set_axis_off()
        pages.append(fig)
        return pages
    headers = ["Date", "Bedtime", "Waketime", "Asleep", "In bed"]
    for start in range(0, len(nights), rows_per_page):
        page = nights[start : start + rows_per_page]
        rows = [
            [
                n.sleep_date.isoformat(),
                n.night_start_local.strftime("%H:%M"),
                n.night_end_local.strftime("%H:%M"),
                format_minutes_hm(n.total_sleep_min),
                format_minutes_hm(n.in_bed_min),
            ]
            for n in page
        ]
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.set_axis_off()
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
            colWidths=[0.20, 0.15, 0.15, 0.20, 0.20],
        )
        table.auto_set_font_size(False)  # noqa: FBT003 — matplotlib API
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        page_num = start // rows_per_page + 1
        page_total = (len(nights) - 1) // rows_per_page + 1
        ax.set_title(f"Nightly raw values (page {page_num} of {page_total})", pad=10)
        pages.append(fig)
    return pages
