"""
tools/chart_tools.py
--------------------
Matplotlib chart generation → base64 PNG.

Tools provided:
  - generate_scatter_chart : Tracks in valence × energy space, colored by popularity
  - generate_radar_chart   : Audio feature comparison for 2-3 tracks
  - generate_bar_chart     : Horizontal bar chart for feature distributions / rankings
  - generate_trend_line    : Line chart for how a feature changes over time / rank
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import matplotlib
matplotlib.use("Agg")  # MUST be before pyplot import — no display needed
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 0,   # suppress axes title (shown in HTML header instead)
})
import matplotlib.patheffects as pe
import numpy as np

# ── App theme (muted colorful palette) ────────────────────────
_BG      = "#13131A"
_INDIGO  = "#8B7CF8"   # primary accent
_TEAL    = "#5BB8C8"   # secondary
_CORAL   = "#E07B6A"   # warm
_GOLD    = "#C9A856"   # highlight
_ROSE    = "#C478B0"   # category
_WHITE   = "#E2E0F0"
_GREY    = "#7A7898"
_DARK    = "#2C2C44"
# Keep _GREEN as alias for scatter colormap (uses YlGn — unchanged)
_GREEN   = _INDIGO     # used as primary color in single-series charts
_SAVE_KW = dict(format="png", dpi=200, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
# Multi-series palette for radar / line comparisons
_PALETTE = [_INDIGO, _TEAL, _CORAL, _GOLD, _ROSE]


def _encode(buf: io.BytesIO) -> str:
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _spotify_fig(figsize=(9, 6)):
    """Return a pre-styled (fig, ax) with Spotify dark background."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_DARK)
    ax.tick_params(colors=_GREY, labelsize=9)
    ax.xaxis.label.set_color(_GREY)
    ax.yaxis.label.set_color(_GREY)
    ax.title.set_color(_WHITE)
    return fig, ax


# ──────────────────────────────────────────────────────────────
# PUBLIC TOOL FUNCTIONS
# ──────────────────────────────────────────────────────────────

def generate_scatter_chart(
    tracks_with_features: list[dict],
    x_feature: str = "duration",
    y_feature: str = "gain",
    color_by: str = "popularity",
    title: str = "Music Feature Map",
) -> dict[str, Any]:
    """
    Scatter plot of tracks in feature space.
    Color encodes popularity (light=low, dark=high).

    Returns:
    {
        "chart_b64": str,
        "chart_type": "scatter",
        "title": str,
        "x_label": str,
        "y_label": str,
    }
    """
    try:
        if not tracks_with_features:
            return {"error": "No track data for scatter chart"}
        xs = [t.get(x_feature, 0) or 0 for t in tracks_with_features]
        ys = [t.get(y_feature, 0) or 0 for t in tracks_with_features]
        cs = [t.get(color_by, 50) or 50 for t in tracks_with_features]
        names = [f"{t.get('name', '')} – {t.get('artist', '')}" for t in tracks_with_features]

        # Validate: need at least some variance (not all-zero)
        if len(set(xs)) < 2 and len(set(ys)) < 2:
            return {"error": f"Scatter chart skipped: all {x_feature}/{y_feature} values identical ({xs[0] if xs else 0})"}

        fig, ax = _spotify_fig(figsize=(8, 5))

        sc = ax.scatter(
            xs, ys,
            c=cs, cmap="cool",
            s=60, alpha=0.85,
            linewidths=0.4, edgecolors=_DARK,
        )

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_by.capitalize(), color=_GREY, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=_GREY, labelsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_GREY)
        cbar.outline.set_edgecolor(_DARK)

        # Label a few standout points (highest + lowest popularity)
        if cs:
            for idx in sorted(range(len(cs)), key=lambda i: cs[i])[-3:]:
                ax.annotate(
                    names[idx],
                    (xs[idx], ys[idx]),
                    fontsize=7, color=_WHITE, alpha=0.85,
                    xytext=(5, 4), textcoords="offset points",
                    path_effects=[pe.withStroke(linewidth=2, foreground=_BG)],
                )

        ax.set_xlabel(x_feature.capitalize(), fontsize=10)
        ax.set_ylabel(y_feature.capitalize(), fontsize=10)
        # title shown in HTML card header
        ax.grid(True, color=_DARK, linewidth=0.5, alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, **_SAVE_KW)
        plt.close(fig)

        return {
            "chart_b64": _encode(buf),
            "chart_type": "scatter",
            "title": title,
            "x_label": x_feature,
            "y_label": y_feature,
        }
    except Exception as exc:
        plt.close("all")
        return {"error": str(exc)}


def generate_radar_chart(
    tracks: list[dict],
    title: str = "Audio Feature Comparison",
) -> dict[str, Any]:
    """
    Radar/spider chart comparing audio features of 2-3 tracks.
    Features: energy, valence, danceability, acousticness, speechiness, instrumentalness

    Returns:
    {
        "chart_b64": str,
        "chart_type": "radar",
        "title": str,
        "tracks_compared": [{"name": str, "artist": str}, ...],
    }
    """
    try:
        # Choose features based on what's available in the first track
        first = tracks[0] if tracks else {}
        if "energy" in first:
            features = ["energy", "valence", "danceability", "acousticness", "speechiness", "instrumentalness"]
            labels   = ["Energy", "Valence", "Danceability", "Acousticness", "Speechiness", "Instr."]
            # Values already 0-1
            def _get_val(t: dict, f: str) -> float:
                return float(t.get(f, 0))
        else:
            # Deezer features — normalise to 0-1 using reasonable max values
            features = ["duration", "gain", "rank", "popularity"]
            labels   = ["Duration", "Loudness", "Chart Rank", "Popularity"]
            # gain is negative; map to loudness: 0 = quiet (-2 gain), 1 = loud (-15 gain)
            def _get_val(t: dict, f: str) -> float:  # noqa: E306
                raw = float(t.get(f, 0))
                if f == "duration":
                    return min(1.0, raw / 480.0)   # 8 min max
                if f == "gain":
                    # -15 → 1.0 (loud), -2 → 0.0 (quiet)
                    return min(1.0, max(0.0, (-raw - 2) / 13.0))
                if f == "rank":
                    # rank 1 → 1.0 (top), rank 100 → 0.0
                    return max(0.0, 1.0 - (raw - 1) / 99.0)
                if f == "popularity":
                    return min(1.0, raw / 100.0)
                return 0.0

        n = len(features)

        angles = [math.pi * 2 * i / n for i in range(n)]
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True}, facecolor=_BG)
        ax.set_facecolor(_BG)
        ax.spines["polar"].set_color(_DARK)
        ax.tick_params(colors=_GREY, labelsize=8)

        colors = _PALETTE
        tracks_compared = []

        for i, track in enumerate(tracks[:3]):
            values = [_get_val(track, f) for f in features]
            values += values[:1]
            color = colors[i % len(colors)]
            ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
            ax.fill(angles, values, color=color, alpha=0.15)
            tracks_compared.append({
                "name": track.get("name", ""),
                "artist": track.get("artist", ""),
            })

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color=_WHITE, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color=_GREY, fontsize=7)
        ax.yaxis.grid(True, color=_DARK, linewidth=0.5)
        ax.xaxis.grid(True, color=_DARK, linewidth=0.5)

        legend_labels = [f"{t['name']} – {t['artist']}" for t in tracks_compared]
        legend = ax.legend(
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            fontsize=8,
            labelcolor=_WHITE,
            facecolor=_BG,
            edgecolor=_DARK,
        )

        # title shown in HTML card header

        buf = io.BytesIO()
        plt.savefig(buf, **_SAVE_KW)
        plt.close(fig)

        return {
            "chart_b64": _encode(buf),
            "chart_type": "radar",
            "title": title,
            "tracks_compared": tracks_compared,
        }
    except Exception as exc:
        plt.close("all")
        return {"error": str(exc)}


def generate_bar_chart(
    data: dict[str, float],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    color: str = _GREEN,
) -> dict[str, Any]:
    """
    Horizontal bar chart for feature distributions, rankings, etc.
    data = {"label": value, ...}

    Returns:
    {
        "chart_b64": str,
        "chart_type": "bar",
        "title": str,
    }
    """
    try:
        if not data:
            return {"error": "No data for bar chart"}
        labels = list(data.keys())
        values = [float(v) for v in data.values()]

        # Validate: need at least 2 entries and non-all-zero values
        if len(labels) < 2:
            return {"error": "Bar chart needs at least 2 data points"}
        if max(abs(v) for v in values) == 0:
            return {"error": "Bar chart skipped: all values are zero"}

        # Sort by value descending
        pairs = sorted(zip(labels, values), key=lambda x: x[1])
        labels, values = zip(*pairs) if pairs else ([], [])

        fig_h = min(7, max(3, len(labels) * 0.42))
        fig, ax = _spotify_fig(figsize=(8, fig_h))

        # Use palette colors cycling if default color not overridden
        bar_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(values))] if color == _GREEN else color
        bars = ax.barh(labels, values, color=bar_colors, alpha=0.88, height=0.6)

        # Value labels at end of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha="left",
                color=_GREY, fontsize=8,
            )

        ax.set_xlabel(xlabel or "Value", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        # title shown in HTML card header
        ax.grid(True, axis="x", color=_DARK, linewidth=0.5, alpha=0.6)
        ax.set_xlim(0, max(values) * 1.15 if values else 1)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, **_SAVE_KW)
        plt.close(fig)

        return {
            "chart_b64": _encode(buf),
            "chart_type": "bar",
            "title": title,
        }
    except Exception as exc:
        plt.close("all")
        return {"error": str(exc)}


def generate_trend_line(
    time_series: list[dict],
    title: str,
    ylabel: str,
) -> dict[str, Any]:
    """
    Line chart showing how a feature changes over time / rank.
    Input: [{"period": str, "value": float, "label": str}, ...]

    Returns:
    {
        "chart_b64": str,
        "chart_type": "line",
        "title": str,
    }
    """
    try:
        if not time_series:
            return {"error": "No data for trend chart"}
        periods = [d.get("period", str(i)) for i, d in enumerate(time_series)]
        values  = [float(d.get("value", 0)) for d in time_series]
        point_labels = [d.get("label", "") for d in time_series]

        if len(periods) < 2:
            return {"error": "Trend chart needs at least 2 data points"}
        if len(set(values)) < 2:
            return {"error": f"Trend chart skipped: all values identical ({values[0]})"}

        fig, ax = _spotify_fig(figsize=(8, 4))

        ax.plot(periods, values, color=_GOLD, linewidth=2.2, marker="o",
                markersize=5, markerfacecolor=_WHITE, markeredgecolor=_GOLD)

        # Shade area under the line
        ax.fill_between(periods, values, alpha=0.12, color=_GOLD)

        # Annotate standout points (max and min)
        if values:
            for idx in {values.index(max(values)), values.index(min(values))}:
                ax.annotate(
                    point_labels[idx] or f"{values[idx]:.2f}",
                    (periods[idx], values[idx]),
                    fontsize=7, color=_WHITE,
                    xytext=(0, 10), textcoords="offset points", ha="center",
                    path_effects=[pe.withStroke(linewidth=2, foreground=_BG)],
                )

        # Rotate x-labels if many periods
        if len(periods) > 8:
            plt.xticks(rotation=45, ha="right")

        ax.set_xlabel("Period", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        # title shown in HTML card header
        ax.grid(True, color=_DARK, linewidth=0.5, alpha=0.6)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, **_SAVE_KW)
        plt.close(fig)

        return {
            "chart_b64": _encode(buf),
            "chart_type": "line",
            "title": title,
        }
    except Exception as exc:
        plt.close("all")
        return {"error": str(exc)}


def generate_histogram(
    tracks_with_features: list[dict],
    feature: str = "duration",
    title: str = "",
    xlabel: str = "",
    bins: int = 8,
) -> dict[str, Any]:
    """
    Histogram of a single numeric feature. Draws median line and colors bars
    with the app palette. Returns {"chart_b64": ..., "chart_type": "histogram"}.
    """
    try:
        import pandas as pd
        df = pd.DataFrame(tracks_with_features)
        if feature not in df.columns:
            return {"error": f"Feature '{feature}' not in data"}
        vals = df[feature].dropna().astype(float).values
        if len(vals) < 3:
            return {"error": "Not enough data for histogram"}
        if np.std(vals) < 1e-6:
            return {"error": "All values identical — no distribution to show"}

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_BG)
        ax.set_facecolor(_BG)

        n_bins = min(bins, len(vals))
        _, bin_edges, patches = ax.hist(vals, bins=n_bins, edgecolor=_BG, alpha=0.88, rwidth=0.82)
        for i, p in enumerate(patches):
            p.set_facecolor(_PALETTE[i % len(_PALETTE)])

        med = float(np.median(vals))
        ax.axvline(med, color=_GOLD, linewidth=1.5, linestyle="--", alpha=0.9)
        span = bin_edges[-1] - bin_edges[0] if bin_edges[-1] != bin_edges[0] else 1
        ax.text(med + span * 0.02, ax.get_ylim()[1] * 0.88,
                f"median\n{med:.1f}", color=_GOLD, fontsize=8.5, va="top")

        ax.set_xlabel(xlabel or feature.replace("_", " ").title(), color=_WHITE, fontsize=10)
        ax.set_ylabel("Tracks", color=_WHITE, fontsize=10)
        ax.tick_params(colors=_WHITE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(_DARK)
        ax.grid(True, axis="y", color=_DARK, linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, **_SAVE_KW)
        plt.close(fig)
        return {
            "chart_b64": _encode(buf),
            "chart_type": "histogram",
            "title": title or f"{feature.replace('_', ' ').title()} Distribution",
        }
    except Exception as exc:
        plt.close("all")
        return {"error": str(exc)}


def generate_correlation_heatmap(
    tracks_with_features: list[dict],
    title: str = "Feature Correlation Matrix",
) -> dict[str, Any]:
    """
    Heatmap of Pearson correlations between duration, gain, rank, popularity.
    Diverging colour: coral (−1) → dark (0) → teal (+1). Each cell shows r value.
    """
    try:
        import pandas as pd
        df = pd.DataFrame(tracks_with_features)
        feature_cols = [c for c in ("duration", "gain", "rank", "popularity")
                        if c in df.columns]
        if len(feature_cols) < 2:
            return {"error": "Need at least 2 numeric features for heatmap"}

        corr = df[feature_cols].dropna().astype(float).corr(method="pearson")
        n = len(feature_cols)
        fig, ax = plt.subplots(figsize=(max(4, n * 1.3), max(3.5, n * 1.1)), facecolor=_BG)
        ax.set_facecolor(_BG)

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("ma", [_CORAL, _DARK, _TEAL], N=256)
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                txt = _BG if abs(val) > 0.55 else _WHITE
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=txt, fontsize=9.5,
                        fontweight="bold" if i != j else "normal")

        labels = [c.title() for c in feature_cols]
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, color=_WHITE, fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, color=_WHITE, fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.ax.tick_params(colors=_WHITE, labelsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, **_SAVE_KW)
        plt.close(fig)
        return {
            "chart_b64": _encode(buf),
            "chart_type": "heatmap",
            "title": title,
        }
    except Exception as exc:
        plt.close("all")
        return {"error": str(exc)}
