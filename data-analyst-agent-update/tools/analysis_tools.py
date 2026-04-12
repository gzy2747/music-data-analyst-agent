"""
tools/analysis_tools.py
------------------------
Statistical analysis tools for music audio features.

Tools provided:
  - compute_feature_statistics  : Descriptive stats across audio features
  - compute_feature_correlations: Pearson correlations between features and popularity
  - detect_clusters             : K-means clustering on valence × energy space
  - run_python_analysis         : Execute custom pandas/numpy code in-process
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _to_df(tracks_with_features: list[dict]) -> pd.DataFrame:
    """Convert list of track+feature dicts to a DataFrame."""
    return pd.DataFrame(tracks_with_features)


# ──────────────────────────────────────────────────────────────
# ANALYSIS TOOLS
# ──────────────────────────────────────────────────────────────

def compute_feature_statistics(tracks_with_features: list[dict]) -> dict[str, Any]:
    """
    Compute descriptive statistics across audio features for a set of tracks.
    Input: list of merged track+feature dicts.

    Returns:
    {
        "n_tracks": int,
        "feature_stats": {
            "energy":        {"mean": float, "median": float, "std": float, "min": float, "max": float},
            "valence":       {...},
            "danceability":  {...},
            "tempo":         {...},
            "acousticness":  {...},
            "speechiness":   {...},
        },
        "popularity_stats": {"mean": float, "median": float, "std": float},
        "most_popular": {"name": str, "artist": str, "popularity": int},
        "least_popular": {"name": str, "artist": str, "popularity": int},
        "preview_available_pct": float,
    }
    """
    if not tracks_with_features:
        return {"error": "No tracks provided"}

    try:
        df = _to_df(tracks_with_features)

        # Support both Spotify features and Deezer features
        candidate_features = [
            "energy", "valence", "danceability", "tempo", "acousticness", "speechiness",
            "duration", "gain", "rank", "popularity",
        ]
        audio_features = [f for f in candidate_features if f in df.columns]
        feature_stats: dict[str, dict] = {}
        for feat in audio_features:
            if feat not in df.columns:
                continue
            col = df[feat].dropna().astype(float)
            feature_stats[feat] = {
                "mean":   round(float(col.mean()), 4),
                "median": round(float(col.median()), 4),
                "std":    round(float(col.std()), 4),
                "min":    round(float(col.min()), 4),
                "max":    round(float(col.max()), 4),
            }

        pop_col = df["popularity"].dropna().astype(float) if "popularity" in df.columns else pd.Series(dtype=float)
        popularity_stats = {
            "mean":   round(float(pop_col.mean()), 2) if not pop_col.empty else 0.0,
            "median": round(float(pop_col.median()), 2) if not pop_col.empty else 0.0,
            "std":    round(float(pop_col.std()), 2) if not pop_col.empty else 0.0,
        }

        most_popular: dict = {}
        least_popular: dict = {}
        if "popularity" in df.columns and not df.empty:
            idx_max = df["popularity"].idxmax()
            idx_min = df["popularity"].idxmin()
            most_popular = {
                "name":       df.loc[idx_max, "name"] if "name" in df.columns else "",
                "artist":     df.loc[idx_max, "artist"] if "artist" in df.columns else "",
                "popularity": int(df.loc[idx_max, "popularity"]),
            }
            least_popular = {
                "name":       df.loc[idx_min, "name"] if "name" in df.columns else "",
                "artist":     df.loc[idx_min, "artist"] if "artist" in df.columns else "",
                "popularity": int(df.loc[idx_min, "popularity"]),
            }

        preview_pct = 0.0
        if "preview_url" in df.columns:
            preview_pct = round(df["preview_url"].notna().mean() * 100, 1)

        return {
            "n_tracks": len(df),
            "feature_stats": feature_stats,
            "popularity_stats": popularity_stats,
            "most_popular": most_popular,
            "least_popular": least_popular,
            "preview_available_pct": preview_pct,
        }
    except Exception as exc:
        return {"error": str(exc)}


def compute_feature_correlations(tracks_with_features: list[dict]) -> dict[str, Any]:
    """
    Compute Pearson correlations between audio features and popularity.

    Returns:
    {
        "correlations_with_popularity": {
            "energy": float,
            "valence": float,
            "danceability": float,
            "tempo": float,
            "acousticness": float,
        },
        "strongest_predictor": str,
        "strongest_correlation": float,
        "feature_correlation_matrix": dict,
    }
    """
    if not tracks_with_features:
        return {"error": "No tracks provided"}

    try:
        df = _to_df(tracks_with_features)

        # Support both Spotify and Deezer feature sets
        feature_cols = [
            "energy", "valence", "danceability", "tempo", "acousticness",
            "speechiness", "instrumentalness", "liveness",
            "duration", "gain", "rank",
        ]
        available = [c for c in feature_cols if c in df.columns]

        if "popularity" not in df.columns:
            return {"error": "No popularity column found"}

        numeric_df = df[available + ["popularity"]].dropna().astype(float)
        if numeric_df.empty:
            return {"error": "No numeric data after dropping NaN"}

        correlations_with_popularity: dict[str, float] = {}
        for feat in available:
            r, _ = stats.pearsonr(numeric_df[feat], numeric_df["popularity"])
            correlations_with_popularity[feat] = round(float(r), 4)

        strongest = max(correlations_with_popularity, key=lambda k: abs(correlations_with_popularity[k]))

        matrix = numeric_df[available].corr(method="pearson").round(4).to_dict()

        return {
            "correlations_with_popularity": correlations_with_popularity,
            "strongest_predictor": strongest,
            "strongest_correlation": correlations_with_popularity[strongest],
            "feature_correlation_matrix": matrix,
        }
    except Exception as exc:
        return {"error": str(exc)}


def detect_clusters(tracks_with_features: list[dict], n_clusters: int = 3) -> dict[str, Any]:
    """
    Simple k-means clustering on valence × energy space.
    Labels each cluster based on centroid position.

    Returns:
    {
        "clusters": [
            {
                "id": int,
                "label": str,
                "n_tracks": int,
                "avg_energy": float,
                "avg_valence": float,
                "avg_popularity": float,
                "example_tracks": [{"name": str, "artist": str}, ...],
            }
        ],
        "most_popular_cluster": str,
    }
    """
    if not tracks_with_features:
        return {"error": "No tracks provided"}

    try:
        from scipy.cluster.vq import kmeans2, whiten  # noqa: PLC0415

        df = _to_df(tracks_with_features)

        # Choose cluster axes: prefer Spotify features, fall back to Deezer features
        if "energy" in df.columns and "valence" in df.columns:
            feat_x, feat_y = "energy", "valence"
        elif "duration" in df.columns and "gain" in df.columns:
            feat_x, feat_y = "duration", "gain"
        elif "duration" in df.columns and "rank" in df.columns:
            feat_x, feat_y = "duration", "rank"
        else:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            num_cols = [c for c in num_cols if c not in ("deezer_id",)]
            if len(num_cols) < 2:
                return {"error": "Need at least 2 numeric feature columns for clustering"}
            feat_x, feat_y = num_cols[0], num_cols[1]

        pop_col = "popularity" if "popularity" in df.columns else (
            "rank" if "rank" in df.columns else None
        )
        base_cols = [feat_x, feat_y, "name", "artist"]
        if pop_col:
            base_cols.append(pop_col)

        sub = df[base_cols].dropna(subset=[feat_x, feat_y]).copy()
        if len(sub) < n_clusters:
            return {"error": f"Not enough tracks ({len(sub)}) for {n_clusters} clusters"}

        # Normalise features to [0,1] before clustering so scale doesn't dominate
        raw_data = sub[[feat_x, feat_y]].astype(float).values
        col_min = raw_data.min(axis=0)
        col_max = raw_data.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        norm_data = (raw_data - col_min) / col_range

        np.random.seed(42)
        centroids, labels = kmeans2(norm_data, n_clusters, minit="points", iter=20)

        sub = sub.reset_index(drop=True)
        sub["cluster"] = labels

        def _label_cluster(nx: float, ny: float, fx: str, fy: str) -> str:
            """Label cluster based on normalised centroid position."""
            if fx == "energy" and fy == "valence":
                if nx >= 0.6 and ny >= 0.6:
                    return "High Energy / Happy"
                elif nx >= 0.6 and ny < 0.6:
                    return "High Energy / Dark"
                elif nx < 0.4 and ny >= 0.6:
                    return "Chill / Upbeat"
                else:
                    return "Melancholic / Quiet"
            elif fx == "duration":
                dur_label = "Long" if nx >= 0.6 else ("Short" if nx <= 0.35 else "Mid-length")
                if fy == "gain":
                    loud_label = "Loud" if ny >= 0.6 else ("Quiet" if ny <= 0.35 else "Mid-volume")
                    return f"{dur_label} / {loud_label}"
                elif fy == "rank":
                    pop_label = "High-ranking" if ny <= 0.35 else ("Low-ranking" if ny >= 0.6 else "Mid-ranking")
                    return f"{dur_label} / {pop_label}"
                return dur_label
            else:
                return f"Group {int(nx * 10)}-{int(ny * 10)}"

        clusters = []
        for cid in range(n_clusters):
            mask = sub["cluster"] == cid
            group = sub[mask]
            if group.empty:
                continue
            avg_x = float(group[feat_x].mean())
            avg_y = float(group[feat_y].mean())
            avg_popularity = float(group[pop_col].mean()) if pop_col and pop_col in group.columns else 0.0

            sort_by = pop_col if pop_col and pop_col in group.columns else None
            top3 = group.nlargest(3, sort_by) if sort_by else group.head(3)
            example_tracks = [
                {"name": row["name"], "artist": row["artist"]}
                for _, row in top3.iterrows()
            ]

            c_nx, c_ny = centroids[cid]
            label = _label_cluster(c_nx, c_ny, feat_x, feat_y)

            cluster_info: dict[str, Any] = {
                "id": cid,
                "label": label,
                "n_tracks": int(mask.sum()),
                f"avg_{feat_x}": round(avg_x, 4),
                f"avg_{feat_y}": round(avg_y, 4),
                "avg_popularity": round(avg_popularity, 2),
                "example_tracks": example_tracks,
            }
            # Keep legacy keys if using Spotify features
            if feat_x == "energy":
                cluster_info["avg_energy"] = cluster_info[f"avg_{feat_x}"]
                cluster_info["avg_valence"] = cluster_info[f"avg_{feat_y}"]
            clusters.append(cluster_info)

        most_popular_cluster = max(clusters, key=lambda c: c["avg_popularity"])["label"] if clusters else ""

        return {
            "clusters": clusters,
            "most_popular_cluster": most_popular_cluster,
        }
    except Exception as exc:
        return {"error": str(exc)}


def compute_derived_metrics(tracks_with_features: list[dict]) -> dict[str, Any]:
    """
    Compute derived metrics from Deezer track data beyond basic statistics:
    - Explicit content rate
    - Artist diversity and frequency (unique artists, top artists by track count)
    - Release recency (days since release_date, % from last 12 months)
    - Duration distribution buckets (short <3min / medium 3-4min / long >4min)
    - Rank tier analysis: compare top-10 vs bottom-10 on each feature

    Returns a dict with sub-dicts for each metric category.
    """
    if not tracks_with_features:
        return {"error": "No tracks provided"}

    try:
        from datetime import date as _date

        df = _to_df(tracks_with_features)
        result: dict[str, Any] = {}

        # ── 1. Explicit content rate ──────────────────────────
        if "explicit" in df.columns:
            explicit_count = int(df["explicit"].astype(bool).sum())
            total = len(df)
            rate = round(explicit_count / total * 100, 1) if total else 0.0
            result["explicit_content"] = {
                "explicit_count": explicit_count,
                "total_tracks": total,
                "explicit_rate_pct": rate,
                "insight": f"{explicit_count}/{total} tracks ({rate:.0f}%) are explicit",
            }

        # ── 2. Artist diversity ───────────────────────────────
        if "artist" in df.columns:
            counts = df["artist"].value_counts()
            unique = int(counts.nunique())
            total = len(df)
            result["artist_diversity"] = {
                "unique_artists": unique,
                "total_tracks": total,
                "diversity_ratio": round(unique / total, 3) if total else 0.0,
                "top_artists": {str(k): int(v) for k, v in counts.head(5).items()},
                "most_represented_artist": str(counts.index[0]),
                "most_represented_track_count": int(counts.iloc[0]),
                "insight": (
                    f"{unique} unique artists across {total} tracks; "
                    f"'{counts.index[0]}' leads with {int(counts.iloc[0])} track(s)"
                ),
            }

        # ── 3. Release recency ───────────────────────────────
        if "release_date" in df.columns:
            today = _date.today()
            dates = pd.to_datetime(df["release_date"], errors="coerce").dropna()
            if not dates.empty:
                ages = np.array([(today - d.date()).days for d in dates])
                recent_12m = int((ages <= 365).sum())
                recent_24m = int((ages <= 730).sum())
                n = len(ages)
                result["release_recency"] = {
                    "avg_age_days": round(float(ages.mean()), 0),
                    "median_age_days": round(float(np.median(ages)), 0),
                    "newest_days_ago": int(ages.min()),
                    "oldest_days_ago": int(ages.max()),
                    "from_last_12_months": recent_12m,
                    "from_last_12_months_pct": round(recent_12m / n * 100, 1),
                    "from_last_24_months": recent_24m,
                    "insight": (
                        f"{recent_12m}/{n} tracks ({recent_12m/n*100:.0f}%) "
                        f"released within the last 12 months; "
                        f"median age {round(float(np.median(ages))/30, 1)} months"
                    ),
                }

        # ── 4. Duration distribution ─────────────────────────
        if "duration" in df.columns:
            dur = df["duration"].dropna().astype(float)
            n = len(dur)
            short  = int((dur < 180).sum())          # < 3 min
            medium = int(((dur >= 180) & (dur < 240)).sum())  # 3-4 min
            long_  = int((dur >= 240).sum())          # > 4 min
            result["duration_distribution"] = {
                "short_under_3min":  short,
                "medium_3_to_4min":  medium,
                "long_over_4min":    long_,
                "short_pct":   round(short  / n * 100, 1) if n else 0.0,
                "medium_pct":  round(medium / n * 100, 1) if n else 0.0,
                "long_pct":    round(long_  / n * 100, 1) if n else 0.0,
                "avg_duration_sec": round(float(dur.mean()), 1),
                "insight": (
                    f"Short (<3min): {short} ({round(short/n*100,0):.0f}%), "
                    f"Medium (3-4min): {medium} ({round(medium/n*100,0):.0f}%), "
                    f"Long (>4min): {long_} ({round(long_/n*100,0):.0f}%)"
                ) if n else "no duration data",
            }

        # ── 5. Rank-tier analysis ────────────────────────────
        if "rank" in df.columns:
            df_s = df.sort_values("rank").reset_index(drop=True)
            top10 = df_s.head(10)
            bot10 = df_s.tail(10)
            tier: dict[str, Any] = {}
            for feat in ("duration", "gain", "popularity"):
                if feat not in df.columns:
                    continue
                t_avg = float(top10[feat].dropna().mean()) if not top10[feat].dropna().empty else None
                b_avg = float(bot10[feat].dropna().mean()) if not bot10[feat].dropna().empty else None
                if t_avg is not None and b_avg is not None:
                    diff_pct = round((t_avg - b_avg) / max(abs(b_avg), 0.001) * 100, 1)
                    tier[feat] = {
                        "top10_avg": round(t_avg, 3),
                        "bottom10_avg": round(b_avg, 3),
                        "difference": round(t_avg - b_avg, 3),
                        "difference_pct": diff_pct,
                    }
            if tier:
                result["rank_tier_analysis"] = tier

        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_python_analysis(tracks_with_features: list[dict], code: str) -> dict[str, Any]:
    """
    Execute custom pandas/numpy code for deeper analysis.
    Pre-injected variables: df (DataFrame of tracks+features), pd, np, stats.
    Agent sets result = {...} with findings.

    Returns:
    {"success": True, "result": dict}
    or {"success": False, "error": str}
    """
    try:
        df = _to_df(tracks_with_features)

        local_ns: dict[str, Any] = {
            "df": df,
            "pd": pd,
            "np": np,
            "stats": stats,
            "result": {},
        }
        exec(textwrap.dedent(code), local_ns)  # noqa: S102
        result = local_ns.get("result", {})
        if not isinstance(result, dict):
            result = {"output": str(result)}
        return {"success": True, "result": result}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
