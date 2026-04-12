"""
agents/orchestrator.py
----------------------
Multi-agent pipeline — powered by Google ADK (Agent Development Kit).

Three-step lifecycle:
  Step 1 – COLLECT     : MusicCollectorAgent fetches Deezer chart data
  Step 2 – EDA         : MusicEDAAgent runs statistical analysis + chart generation
                         with a CODE-LEVEL iterative refinement while-loop
  Step 3 – HYPOTHESIZE : Generator → Critic → (optional revision) pattern
                         HypothesisCriticAgent can REJECT and force a rewrite

Multi-agent patterns:
  • Orchestrator → handoff chain: Collector → EDA → Hypothesis
  • Generator-Critic with feedback loop: HypothesisAgent generates,
    HypothesisCriticAgent evaluates verdict (ACCEPT/REJECT),
    HypothesisAgent revises if rejected — up to MAX_CRITIC_ROUNDS
  • Iterative Refinement: code-level while-loop in run_eda() checks that
    ≥ 3 findings contain specific numbers; re-runs agent if not met
  • Parallel Execution: adk_parallel_analyze() runs statistics + correlations
    simultaneously via ThreadPoolExecutor

Framework: Google ADK (google-adk)
Model: gemini-2.5-flash via Vertex AI
Auth: Application Default Credentials (gcloud auth application-default login)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

# ── Google ADK imports ────────────────────────────────────────
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

# ── Music tool imports ────────────────────────────────────────
from tools.deezer_tools import (
    get_top_tracks_chart,
    get_track_info,
    get_tag_top_tracks,
    get_artist_top_tracks,
    get_tracks_details,
)
from tools.analysis_tools import (
    compute_feature_statistics,
    compute_feature_correlations,
    detect_clusters,
    run_python_analysis,
    compute_derived_metrics,
)
from tools.wikipedia_tools import (
    get_artist_wikipedia_summary,
    get_genre_wikipedia_overview,
)
from tools.chart_tools import (
    generate_scatter_chart,
    generate_radar_chart,
    generate_bar_chart,
    generate_trend_line,
    generate_histogram,
    generate_correlation_heatmap,
)
from tools.artifact_tools import save_artifact

MODEL    = "gemini-2.5-flash"
APP_NAME = "musicanalyst"

# How many critic rounds before we accept whatever we have.
# Round 0 → generate + critique, Round 1 → revise + critique,
# Round 2 → final revision, accepted without further critique.
MAX_CRITIC_ROUNDS = 3
# How many EDA refinement iterations before we stop
MAX_EDA_ITERATIONS = 3
# Minimum number of specific (numeric) findings before EDA is considered done
MIN_SPECIFIC_FINDINGS = 3


# ──────────────────────────────────────────────────────────────
# EDA session cache — thread-local so concurrent requests don't collide
# ──────────────────────────────────────────────────────────────

_tl = threading.local()


def _get_eda_tracks() -> list[dict]:
    return getattr(_tl, "tracks", [])


def _get_eda_charts() -> list[dict]:
    return getattr(_tl, "charts", [])


def _set_eda_tracks(tracks: list[dict]) -> None:
    _tl.tracks = tracks


def _reset_charts_cache() -> None:
    _tl.charts = []


# ──────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────

def _sanitize(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types."""
    import pandas as pd
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON; fall back to regex extraction."""
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
    clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"raw": text}


def _finding_is_specific(finding: dict) -> bool:
    """
    Return True if a finding contains a concrete measurement with a meaningful unit.
    This is the CODE-LEVEL gate for the iterative refinement loop.

    Rules (at least one must match):
      - Number + meaningful unit: %, s, dB, r=, avg, mean
      - Comparative phrasing: "rank 18 vs 21", "65% vs 28%"
      - Statistical term with a number: "average 198s", "correlation 0.41"
      - Bare counts like "50 tracks" do NOT qualify.
    """
    text = finding.get("finding", "")
    has_unit = bool(re.search(
        r'\d+\.?\d*\s*(%|s\b|dB\b|ms\b|r=|avg\b|mean\b)',
        text, re.IGNORECASE
    ))
    has_comparison = bool(re.search(
        r'\d+\.?\d*\s*(vs\.?|versus|compared)\s*\d+\.?\d*',
        text, re.IGNORECASE
    ))
    has_stat = bool(re.search(
        r'(average|median|mean|correlation|std|deviation|percentile|quartile)'
        r'.{0,30}\d+\.?\d*',
        text, re.IGNORECASE
    ))
    return has_unit or has_comparison or has_stat


# ──────────────────────────────────────────────────────────────
# ADK-wrapped tool functions
# ──────────────────────────────────────────────────────────────

# ── Deezer tools ──────────────────────────────────────────────

def adk_get_top_tracks_chart(limit: int = 100, page: int = 1) -> str:
    """Get Deezer global top-tracks chart with rank and track metadata."""
    result = get_top_tracks_chart(limit=limit, page=page)
    return json.dumps(_sanitize(result), default=str)


def adk_get_track_info(track_name: str, artist_name: str) -> str:
    """Get detailed Deezer info for a specific track: rank, duration, BPM, preview URL."""
    result = get_track_info(track_name=track_name, artist_name=artist_name)
    return json.dumps(_sanitize(result), default=str)


def adk_get_tag_top_tracks(tag: str, limit: int = 100) -> str:
    """Search Deezer for top tracks matching a genre/mood/era tag (e.g. 'indie pop', 'electronic', '2010s')."""
    result = get_tag_top_tracks(tag=tag, limit=limit)
    return json.dumps(_sanitize(result), default=str)


def adk_get_artist_top_tracks(artist_name: str, limit: int = 50) -> str:
    """Get an artist's top tracks on Deezer with rank and preview URLs."""
    result = get_artist_top_tracks(artist_name=artist_name, limit=limit)
    return json.dumps(_sanitize(result), default=str)


def adk_get_tracks_details(deezer_ids: list[int]) -> str:
    """Batch-fetch full track details (BPM, gain, album art, duration) for a list of Deezer track IDs."""
    result = get_tracks_details(deezer_ids=deezer_ids)
    return json.dumps(_sanitize(result), default=str)


# ── Analysis tools ─────────────────────────────────────────────
# These read from thread-local cache set by run_eda — no large data params.

def adk_compute_feature_statistics() -> str:
    """Compute descriptive statistics (mean/median/std/min/max) for duration, gain, rank, popularity across all collected tracks."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty — no data to analyse. Ensure collector ran successfully."})
    result = compute_feature_statistics(tracks_with_features=tracks)
    return json.dumps(_sanitize(result), default=str)


def adk_compute_feature_correlations() -> str:
    """Compute Pearson correlations between features (duration, gain, rank) and popularity. Returns strongest predictor."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = compute_feature_correlations(tracks_with_features=tracks)
    return json.dumps(_sanitize(result), default=str)


def adk_detect_clusters(n_clusters: int = 3) -> str:
    """K-means cluster collected tracks in duration × gain space and label each cluster."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = detect_clusters(tracks_with_features=tracks, n_clusters=n_clusters)
    return json.dumps(_sanitize(result), default=str)


def adk_run_python_analysis(code: str) -> str:
    """Execute custom pandas/numpy code against collected track data. Pre-injected variables: df (DataFrame of all tracks), pd, np, stats. Set result = {...} with your findings."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = run_python_analysis(tracks_with_features=tracks, code=code)
    return json.dumps(_sanitize(result), default=str)


def adk_compute_derived_metrics() -> str:
    """Compute derived music metrics not in basic statistics:
    - explicit_content: rate of explicit tracks
    - artist_diversity: unique artists, top artists by track count
    - release_recency: avg age in days, % from last 12/24 months
    - duration_distribution: short (<3min) / medium (3-4min) / long (>4min) buckets
    - rank_tier_analysis: top-10 vs bottom-10 comparison on duration, gain, popularity
    No parameters needed — uses cached track data automatically."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = compute_derived_metrics(tracks_with_features=tracks)
    return json.dumps(_sanitize(result), default=str)


# ── Chart tools ───────────────────────────────────────────────

def _store_chart(result: dict) -> str:
    """Store a chart in thread-local cache and return a small ack to the model."""
    if not hasattr(_tl, "charts"):
        _tl.charts = []
    if result.get("chart_b64"):
        _tl.charts.append({
            "chart_b64":  result["chart_b64"],
            "chart_type": result.get("chart_type", ""),
            "title":      result.get("title", ""),
        })
    elif result.get("error"):
        return json.dumps({"stored": False, "reason": result["error"]})
    return json.dumps({"stored": True, "title": result.get("title", ""), "chart_type": result.get("chart_type", "")})


def adk_generate_scatter_chart(
    x_feature: str = "duration",
    y_feature: str = "gain",
    color_by: str = "popularity",
    title: str = "Music Feature Map",
) -> str:
    """Generate a scatter plot of all collected tracks (default: duration vs gain, colored by popularity). Chart is stored automatically — do NOT copy chart_b64 into your output."""
    result = generate_scatter_chart(
        tracks_with_features=_get_eda_tracks(),
        x_feature=x_feature,
        y_feature=y_feature,
        color_by=color_by,
        title=title,
    )
    return _store_chart(_sanitize(result))


def adk_generate_radar_chart(track_indices: str = "0,1,2", title: str = "Track Feature Comparison") -> str:
    """Generate a radar chart comparing features of 2-3 tracks from the collected data.
    track_indices: comma-separated indices, e.g. '0,1,2'.
    Chart is stored automatically — do NOT copy chart_b64 into your output."""
    cache = _get_eda_tracks()
    indices = [int(i.strip()) for i in track_indices.split(",") if i.strip().isdigit()]
    tracks = [cache[i] for i in indices if i < len(cache)]
    result = generate_radar_chart(tracks=tracks, title=title)
    return _store_chart(_sanitize(result))


def adk_generate_bar_chart(
    labels_json: str,
    values_json: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
) -> str:
    """Generate a horizontal bar chart. Chart is stored automatically — do NOT copy chart_b64 into your output.
    labels_json: JSON array of label strings, e.g. '[\"Artist A\",\"Artist B\"]'.
    values_json: JSON array of numeric values, e.g. '[240, 195]'."""
    labels = labels_json if isinstance(labels_json, list) else json.loads(labels_json)
    values = values_json if isinstance(values_json, list) else json.loads(values_json)
    data = dict(zip(labels, values))
    result = generate_bar_chart(data=data, title=title, xlabel=xlabel, ylabel=ylabel)
    return _store_chart(_sanitize(result))


def adk_generate_histogram(feature: str, title: str = "", xlabel: str = "", bins: int = 8) -> str:
    """Generate a histogram of a single feature distribution across all collected tracks.
    feature: one of 'duration', 'gain', 'rank', 'popularity'.
    Shows median line; bars colored with app palette. Chart is stored automatically."""
    result = generate_histogram(
        tracks_with_features=_get_eda_tracks(),
        feature=feature,
        title=title,
        xlabel=xlabel,
        bins=bins,
    )
    return _store_chart(_sanitize(result))


def adk_generate_correlation_heatmap(title: str = "Feature Correlation Matrix") -> str:
    """Generate a Pearson correlation heatmap between duration, gain, rank, and popularity.
    Diverging colour (coral→teal). Chart is stored automatically — do NOT copy chart_b64 into output."""
    result = generate_correlation_heatmap(
        tracks_with_features=_get_eda_tracks(),
        title=title,
    )
    return _store_chart(_sanitize(result))


def adk_generate_trend_line(periods_json: str, values_json: str, labels_json: str, title: str, ylabel: str) -> str:
    """Generate a line chart showing how a feature changes across groups/periods. Chart is stored automatically — do NOT copy chart_b64 into your output.
    periods_json: JSON array of period strings, e.g. '[\"rank 1-10\",\"rank 11-20\"]'.
    values_json: JSON array of float values.
    labels_json: JSON array of point label strings."""
    periods     = periods_json  if isinstance(periods_json,  list) else json.loads(periods_json)
    values      = values_json   if isinstance(values_json,   list) else json.loads(values_json)
    labels_list = labels_json   if isinstance(labels_json,   list) else json.loads(labels_json)
    time_series = [{"period": p, "value": v, "label": l} for p, v, l in zip(periods, values, labels_list)]
    result = generate_trend_line(time_series=time_series, title=title, ylabel=ylabel)
    return _store_chart(_sanitize(result))


# ── Wikipedia tools (second data retrieval method) ────────────

def adk_get_artist_wikipedia_summary(artist_name: str) -> str:
    """Fetch a Wikipedia biography and genre description for a music artist.
    Returns description, biography extract, and Wikipedia URL.
    This is a SEPARATE data source from Deezer — uses Wikipedia REST API (no key needed)."""
    result = get_artist_wikipedia_summary(artist_name=artist_name)
    return json.dumps(_sanitize(result), default=str)


def adk_get_genre_wikipedia_overview(genre: str) -> str:
    """Fetch a Wikipedia overview for a music genre (e.g. 'hip-hop', 'K-pop', 'electronic').
    Provides historical and cultural context for genre-focused analyses.
    Uses Wikipedia REST API — separate from Deezer."""
    result = get_genre_wikipedia_overview(genre=genre)
    return json.dumps(_sanitize(result), default=str)


# ── Parallel analysis tool ────────────────────────────────────

def adk_parallel_analyze() -> str:
    """Run feature statistics AND feature correlations in PARALLEL using ThreadPoolExecutor.
    Returns combined results from both analyses simultaneously — faster than calling each separately.
    Use this as your first EDA step instead of calling statistics and correlations one by one."""
    import concurrent.futures
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty — no data to analyse."})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_stats = executor.submit(compute_feature_statistics, tracks)
        f_corr  = executor.submit(compute_feature_correlations, tracks)
        stats_result = f_stats.result()
        corr_result  = f_corr.result()
    combined = {"statistics": stats_result, "correlations": corr_result}
    return json.dumps(_sanitize(combined), default=str)


# ── Artifact tool ─────────────────────────────────────────────

def adk_save_artifact(name: str, content: str, kind: str = "markdown") -> str:
    """Save the final analysis report as a markdown file to the artifacts/ directory."""
    result = save_artifact(name=name, content=content, kind=kind)
    return json.dumps(_sanitize(result), default=str)


# ──────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────

# ── Version 1's more detailed Collector prompt ────────────────
COLLECTOR_SYSTEM = """
You are the MusicCollectorAgent, Step 1 of a music analysis pipeline.
Your ONLY job is to collect real-world music data from Deezer AND Wikipedia based on the user's question.

Data sources:
  1. Deezer (primary) — chart data, audio features, track rankings (no auth needed)
  2. Wikipedia (secondary) — artist biographies, genre context (no auth needed)

Available tools: adk_get_tag_top_tracks, adk_get_top_tracks_chart, adk_get_track_info,
                 adk_get_artist_top_tracks, adk_get_tracks_details,
                 adk_get_artist_wikipedia_summary, adk_get_genre_wikipedia_overview

Instructions:
1. Parse the user's question to identify genre, mood, era, or artist.
2. Collect AT LEAST 50 tracks using ONE of these Deezer strategies:
   - Genre/mood query  → adk_get_tag_top_tracks(tag="...", limit=100)
   - Global chart      → adk_get_top_tracks_chart(limit=100)
   - Artist focus      → adk_get_artist_top_tracks(artist_name="...", limit=50)
   - Specific track    → adk_get_track_info(track_name="...", artist_name="...")
   Each call already returns fully enriched tracks (gain, album_art, release_date, explicit).
3. ALSO fetch Wikipedia context using a SECOND data source — but ONLY when relevant:
   - If question is about a specific artist → adk_get_artist_wikipedia_summary(artist_name="...")
   - If question is about a specific genre  → adk_get_genre_wikipedia_overview(genre="...")
   - If question is about global charts / trending / top hits → SKIP Wikipedia entirely
   Include any Wikipedia info in collection_notes.
4. Compute popularity for each track as: popularity = max(0, 100 - rank).
   If rank is 0 or unavailable, set popularity = 50.

Return ONLY this JSON — no prose, no markdown fences:
{
  "query_interpretation": "...",
  "n_tracks": 0,
  "tracks": [
    {
      "id": 0,
      "name": "",
      "artist": "",
      "album": "",
      "rank": 0,
      "popularity": 0,
      "duration": 0,
      "gain": 0.0,
      "release_date": "",
      "explicit": false,
      "preview_url": null,
      "deezer_url": "",
      "album_art": ""
    }
  ],
  "deezer_context": {
    "source": "",
    "top_tracks": [],
    "chart_data": {}
  },
  "collection_notes": ""
}
"""

# ── Version 1's richer EDA prompt (more chart options, gain field explained) ──
EDA_SYSTEM = """
You are the MusicEDAAgent, Step 2 of a music analysis pipeline.

You receive a summary of the collected data (field names + aggregate stats only).
The FULL track dataset is pre-loaded in the analysis tools — call them directly,
do NOT ask for raw track data.

Available numeric features: rank (chart position, lower=more popular), popularity (0-100),
duration (seconds), gain (loudness normalization; more negative = naturally louder track).
Also available: release_date (YYYY-MM-DD string), explicit (bool).

Perform thorough Exploratory Data Analysis. Find SPECIFIC, NON-OBVIOUS patterns.
Do NOT output generic summaries like "most songs have similar duration."
DO find things like "tracks shorter than 200s rank 15% higher on average (rank 18 vs 21)."
The 'gain' field reveals loudness: values like -10 mean the track is very loud (compressed).

IMPORTANT: The analysis tools (statistics, correlations, clusters, scatter chart, radar chart) require
NO parameters — they automatically operate on all collected tracks. Just call them directly.

ANALYSIS STEPS — follow in order:
1. Call adk_parallel_analyze() — runs feature statistics AND correlations simultaneously (parallel execution)
2. Call adk_compute_derived_metrics() — gets explicit rate, artist diversity, release recency, duration buckets, rank-tier comparisons
3. If any |correlation| > 0.15 or interesting pattern found, dig deeper:
   - adk_run_python_analysis(code="result = {'finding': ...}")
4. Call adk_detect_clusters(n_clusters=3)
5. Generate AT LEAST 3 charts (charts are auto-stored — set "charts": [] in output, do NOT include chart_b64):

   ALWAYS generate these two first:
   a. adk_generate_correlation_heatmap(title="Feature Correlation Matrix")
   b. adk_generate_scatter_chart(x_feature="duration", y_feature="gain", color_by="popularity", title="Duration vs Loudness")

   Then pick 1-2 more from:
   c. adk_generate_histogram(feature="duration", title="Duration Distribution", xlabel="Seconds", bins=8)
   d. adk_generate_histogram(feature="gain", title="Mastering Gain Distribution", xlabel="dB", bins=7)
   e. adk_generate_bar_chart(labels_json='["Artist A","B"]', values_json='[240,195]', title="Artist Track Count", ylabel="tracks")
   f. adk_generate_bar_chart with top-10 vs bottom-10 avg features from rank_tier_analysis
   g. adk_generate_trend_line(periods_json='["rank 1-5","rank 6-10","rank 11-15","rank 16-20","rank 21-25"]', values_json='[...]', labels_json='[...]', title="Feature by Rank Band", ylabel="...")
   h. adk_generate_radar_chart(track_indices="0,1,2", title="Top Track Feature Comparison")

6. Keep iterating until >= 3 distinct, specific, numbered findings.

Available metrics to analyse (all computable from Deezer data):
- duration (seconds): track length
- gain (dB): loudness proxy — more negative = naturally louder/compressed
- rank: chart position (1 = most popular), derived from Deezer chart rank
- popularity: 100 - rank, a 0-100 score
- explicit (bool): parental advisory flag
- release_date (YYYY-MM-DD): when the track was released
- artist: artist name — use for frequency/diversity analysis

For representative_tracks, pick 6-8 tracks that best illustrate your findings.
CRITICAL: copy name, artist, and deezer_id EXACTLY from the track index provided.
Leave preview_url, deezer_url, album_art as empty strings — the server will fill them in.
Prefer tracks where has_preview=true so users can listen.

Formatting rule: in "finding" strings, wrap key numbers, percentages, track names, and artist names
in **double asterisks** so they render as bold in the UI. Example:
  "Tracks shorter than **200s** rank **15% higher** on average (**rank 18** vs **21** for longer tracks)."

Output ONLY this JSON (no prose, no fences):
{
  "findings": [
    {
      "id": 1,
      "category": "tempo_pattern|popularity_driver|genre_characteristic|duration_trend|anomaly|cluster",
      "finding": "",
      "evidence": {}
    }
  ],
  "charts": [],
  "representative_tracks": [
    {
      "name": "",
      "artist": "",
      "deezer_id": 0,
      "preview_url": "",
      "deezer_url": "",
      "album_art": "",
      "why_representative": "",
      "key_features": {
        "duration": 0,
        "gain": 0.0,
        "popularity": 0
      }
    }
  ],
  "tracks": []
}
"""

# ── Refinement prompt — injected when the while-loop triggers a re-run ──────
EDA_REFINEMENT_SYSTEM = """
You are the MusicEDAAgent running a REFINEMENT PASS.

Your previous analysis pass did not produce enough specific, numeric findings.
Below you will see: (1) what findings were already found, (2) which analytical
dimensions are still uncovered.

Your job: call MORE analysis tools to fill the gaps and produce findings that
contain CONCRETE NUMBERS (counts, percentages, seconds, dB values, correlation
coefficients). Do NOT repeat findings already listed.

The full track dataset is still pre-loaded in the analysis tools.

Return the SAME JSON schema as before — only NEW findings (the orchestrator will merge).
Output ONLY valid JSON, no prose, no markdown fences.
"""

# ── Version 1's richer Hypothesis prompt (highlight_tracks with why_this_track) ──
HYPOTHESIS_SYSTEM = """
You are the HypothesisAgent (Generator), Step 3 of a music analysis pipeline.

You receive EDA findings and must produce a compelling, data-grounded music hypothesis.

Rules:
- Your hypothesis must be SPECIFIC and NON-OBVIOUS.
  BAD:  "Popular songs tend to be energetic and danceable."
  GOOD: "In this dataset, loudness (gain) correlates with rank at r=−0.41, but only for tracks
         under 210s: shorter loud tracks capture the top quartile 65% of the time vs 28% for longer ones."
- Every claim must cite a specific number from the EDA findings.
- Do NOT invent data — only use what EDA found.
- List 1-2 alternative explanations that could challenge your hypothesis.
- Formatting: wrap key numbers, track names, artist names, and technical terms in **double asterisks**
  in "hypothesis", "summary_paragraph", and evidence "point"/"data" fields so they render bold.
- CRITICAL: The "hypothesis" field is MANDATORY — it must NEVER be empty or null. Even for yes/no
  questions, after writing direct_answer you MUST still write a full hypothesis explaining the
  deeper pattern, mechanism, or nuance behind the answer.

Instructions:
1. Review all EDA findings carefully.
2. If the original question is a yes/no question (e.g. "Are louder tracks more popular?") or asks
   "what X" (e.g. "What makes a song top the charts?"), write a concise 1-2 sentence direct_answer.
   Otherwise leave direct_answer as "".
3. Form ONE clear, non-trivial hypothesis grounded in the data numbers. This field is REQUIRED.
   Use the pattern: "[Feature/pattern] predicts [outcome] — specifically [number], suggesting [mechanism]."
4. Pick 5-7 highlight_tracks that best illustrate your hypothesis. Try to pick tracks
   different from representative_tracks in EDA for variety, but overlap is OK.
   CRITICAL: copy name, artist, and deezer_id EXACTLY from the track index provided.
   Leave preview_url, deezer_url, album_art as empty strings — the server fills them in.
   Prefer tracks where has_preview=true.
5. Write summary_paragraph as 1-2 SHORT plain-language sentences with NO numbers or jargon —
   it should feel like telling a friend what you found, not re-stating the hypothesis.
   BAD: "Tracks with gain > -8.3 dB achieve rank 4.0 vs 6.5 for louder tracks."
   GOOD: "Right now, newer and slightly less compressed songs are dominating the charts."
6. Call adk_save_artifact to persist the full markdown report.

Output ONLY this JSON (no prose, no fences):
{
  "direct_answer": "",
  "hypothesis": "",
  "confidence": "Low|Medium|High",
  "confidence_reasoning": "",
  "supporting_evidence": [
    {"point": "", "data": ""}
  ],
  "alternative_explanations": ["", ""],
  "summary_paragraph": "1-2 plain-language sentences for non-technical readers. NO numbers or jargon.",
  "highlight_tracks": [
    {
      "name": "",
      "artist": "",
      "deezer_id": 0,
      "preview_url": "",
      "deezer_url": "",
      "album_art": "",
      "why_this_track": "",
      "key_features": {"duration": 0, "gain": 0.0, "popularity": 0}
    }
  ],
  "artifact_path": ""
}
"""

CRITIC_SYSTEM = """
You are the HypothesisCriticAgent. Your job is to strictly evaluate a music hypothesis
produced by the HypothesisAgent.

Check ALL of the following:
1. Does every claim in "hypothesis" cite a specific number that appears in the EDA findings?
2. Is the hypothesis non-obvious? (Reject if it merely restates common knowledge.)
3. Does "supporting_evidence" have at least 2 entries, each with a concrete data point?
4. Is "hypothesis" non-empty?

If ALL checks pass → verdict: "ACCEPT"
If ANY check fails → verdict: "REJECT" with specific issues listed.

Output ONLY this JSON (no prose, no fences):
{
  "verdict": "ACCEPT" | "REJECT",
  "issues": ["describe each failed check"],
  "specific_gaps": ["e.g. claim X has no numeric citation from EDA"]
}
"""


# ──────────────────────────────────────────────────────────────
# Agent factories
# ──────────────────────────────────────────────────────────────

def _make_collector_agent() -> LlmAgent:
    return LlmAgent(
        name="MusicCollectorAgent",
        model=MODEL,
        instruction=COLLECTOR_SYSTEM,
        tools=[
            adk_get_tag_top_tracks,
            adk_get_top_tracks_chart,
            adk_get_track_info,
            adk_get_artist_top_tracks,
            adk_get_tracks_details,
            adk_get_artist_wikipedia_summary,
            adk_get_genre_wikipedia_overview,
        ],
    )


def _make_eda_agent(refinement_mode: bool = False) -> LlmAgent:
    return LlmAgent(
        name="MusicEDAAgent",
        model=MODEL,
        instruction=EDA_REFINEMENT_SYSTEM if refinement_mode else EDA_SYSTEM,
        tools=[
            adk_parallel_analyze,
            adk_compute_feature_statistics,
            adk_compute_feature_correlations,
            adk_compute_derived_metrics,
            adk_detect_clusters,
            adk_run_python_analysis,
            adk_generate_scatter_chart,
            adk_generate_radar_chart,
            adk_generate_bar_chart,
            adk_generate_trend_line,
            adk_generate_histogram,
            adk_generate_correlation_heatmap,
        ],
    )


def _make_hypothesis_agent() -> LlmAgent:
    return LlmAgent(
        name="HypothesisAgent",
        model=MODEL,
        instruction=HYPOTHESIS_SYSTEM,
        tools=[adk_save_artifact],
    )


def _make_critic_agent() -> LlmAgent:
    """Critic agent — no tools needed, pure evaluation."""
    return LlmAgent(
        name="HypothesisCriticAgent",
        model=MODEL,
        instruction=CRITIC_SYSTEM,
        tools=[],
    )


# ──────────────────────────────────────────────────────────────
# ADK runner helper
# ──────────────────────────────────────────────────────────────

def _run_adk_agent(
    agent: LlmAgent,
    message: str,
    progress_cb=None,
) -> tuple[str, list[dict]]:
    """
    Run a single ADK agent synchronously.
    Returns (final_text, tool_call_log).
    progress_cb(tool_name, detail): optional callable invoked on each tool call.
    """
    import asyncio

    async def _run():
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
        session = await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id="user",
        )
        tool_log = []
        final_text = ""
        all_agent_texts: list[str] = []

        async for event in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=message)],
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        tool_name = part.function_call.name
                        args = dict(part.function_call.args or {})
                        tool_log.append({"tool": tool_name, "input": args})
                        if progress_cb:
                            detail = ""
                            for v in args.values():
                                if isinstance(v, str) and len(v) < 80:
                                    detail = v
                                    break
                            try:
                                progress_cb(tool_name, detail)
                            except Exception:
                                pass
                    elif hasattr(part, "text") and part.text:
                        if event.author != "user":
                            all_agent_texts.append(part.text)
            if event.is_final_response():
                if event.content and event.content.parts:
                    t = " ".join(
                        p.text for p in event.content.parts if hasattr(p, "text") and p.text
                    ).strip()
                    if t:
                        final_text = t

        if not final_text and all_agent_texts:
            final_text = all_agent_texts[-1].strip()

        return final_text, tool_log

    import time as _time
    import re as _re

    def _execute():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(_run())
            else:
                return loop.run_until_complete(_run())
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_run())
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

    for attempt in range(6):
        try:
            return _execute()
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                m = _re.search(r"retry[_ ](?:after|in)[:\s]*(\d+)", msg, _re.IGNORECASE)
                explicit = int(m.group(1)) + 2 if m else None
                backoff = min(15 * (2 ** attempt), 240)
                wait = explicit if explicit and explicit > backoff else backoff
                if attempt < 5:
                    _time.sleep(wait)
                    continue
            raise


# ──────────────────────────────────────────────────────────────
# Deezer Python-level fallback
# ──────────────────────────────────────────────────────────────

def _direct_deezer_fallback(question: str) -> dict:
    """
    Python-level Deezer fallback — called when the agent fails to return tracks.
    Targets ≥ 50 tracks for a non-trivial dataset.
    """
    q = question.lower()

    # ── Artist query ──────────────────────────────────────────
    _MUSIC_NOUNS = r"(?:songs?|tracks?|music|top|most|charting|popular|hit|sound|style|discography|catalog)"
    _poss_hits = re.findall(
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})'s\s+" + _MUSIC_NOUNS,
        question
    )
    _NON_NAMES = {"What", "How", "Why", "Which", "Who", "Do", "Does", "Are",
                  "Analyze", "Compare", "Tell", "Give", "Find", "Show"}
    _poss_cleaned = []
    for n in _poss_hits:
        words = n.split()
        if words[0] in _NON_NAMES:
            rest = ' '.join(words[1:])
            if rest:
                _poss_cleaned.append(rest)
        else:
            _poss_cleaned.append(n)

    artist_name = None
    if _poss_cleaned:
        artist_name = min(_poss_cleaned, key=len).strip()
    else:
        artist_match = (
            re.search(r'\b(?:by|from|songs?\s+(?:by|from)|tracks?\s+(?:by|from))\s+([A-Za-z][A-Za-z\s\-\'\.]{2,28})',
                      question, re.IGNORECASE)
            or re.search(r'\b(?:analyze|about)\s+((?:[A-Z][A-Za-z]+\s+){1,3}[A-Z][A-Za-z]+)', question)
            or re.search(r'\bHow\s+does\s+((?:[A-Z][A-Za-z]+\s+){1,3}[A-Z][A-Za-z]+)', question)
        )
        artist_name = artist_match.group(1).strip() if artist_match else None

    if artist_name:
        result = get_artist_top_tracks(artist_name=artist_name, limit=50)
        if result.get("tracks"):
            tracks = result["tracks"]
            for i, t in enumerate(tracks):
                t["popularity"] = max(0, 100 - t.get("rank", i + 1))
            return {
                "query_interpretation": f"Top tracks by {artist_name}",
                "n_tracks": len(tracks),
                "tracks": tracks,
                "deezer_context": {"source": f"artist:{artist_name}", "top_tracks": [], "chart_data": {}},
                "collection_notes": "Collected via Python fallback (artist top tracks).",
            }

    global_keywords = ["global", "top", "chart", "hit", "popular", "trending",
                       "billboard", "worldwide", "right now", "current"]
    is_global = any(kw in q for kw in global_keywords)

    genre_keywords = ["pop", "hip.hop", "rap", "rock", "jazz", "electronic", "country",
                      "r&b", "soul", "metal", "indie", "chill", "sad", "happy", "dance",
                      "edm", "classical", "latin", "k.pop", "reggae"]
    genre_found = next((kw for kw in genre_keywords if re.search(r'\b' + kw + r'\b', q)), None)

    if genre_found and not is_global:
        result = get_tag_top_tracks(tag=genre_found, limit=100)
        if result.get("tracks"):
            tracks = result["tracks"]
            for i, t in enumerate(tracks):
                t["popularity"] = max(0, 100 - t.get("rank", i + 1))
            return {
                "query_interpretation": f"Top {genre_found} tracks",
                "n_tracks": len(tracks),
                "tracks": tracks,
                "deezer_context": {"source": f"genre:{genre_found}", "top_tracks": [], "chart_data": {}},
                "collection_notes": "Collected via Python fallback (genre search).",
            }

    result = get_top_tracks_chart(limit=100, page=1)
    if result.get("tracks"):
        tracks = result["tracks"]
        for i, t in enumerate(tracks):
            t["popularity"] = max(0, 100 - t.get("rank", i + 1))
        return {
            "query_interpretation": "Global top tracks chart",
            "n_tracks": len(tracks),
            "tracks": tracks,
            "deezer_context": {"source": "chart:global", "top_tracks": [], "chart_data": {}},
            "collection_notes": "Collected via Python fallback (global chart).",
        }

    return {"query_interpretation": "", "n_tracks": 0, "tracks": [],
            "deezer_context": {}, "collection_notes": ""}


# ──────────────────────────────────────────────────────────────
# Per-step runners
# ──────────────────────────────────────────────────────────────

def run_collector(question: str, progress_cb=None) -> tuple[dict, list[dict]]:
    """
    Step 1 — Collect.
    Runs MusicCollectorAgent. Falls back to direct Python Deezer calls if agent
    returns no tracks. Targets ≥ 50 tracks for a non-trivial dataset.
    """
    agent = _make_collector_agent()
    text, logs = _run_adk_agent(
        agent=agent,
        message=f"Collect music data for this analysis question: {question}",
        progress_cb=progress_cb,
    )
    result = _parse_json(text)

    if not result.get("tracks"):
        if progress_cb:
            try:
                progress_cb("deezer_fallback", "agent returned no tracks — fetching directly")
            except Exception:
                pass
        result = _direct_deezer_fallback(question)

    return result, logs


def _build_eda_initial_msg(question: str, tracks: list[dict]) -> str:
    """
    Build the EDA agent's initial message.
    Sends only schema + aggregate stats + compact track index.
    Full data is in thread-local cache accessed via tool calls.
    """
    n = len(tracks)
    schema_fields = list(tracks[0].keys()) if tracks else []

    durations = [t.get("duration", 0) for t in tracks if t.get("duration")]
    gains     = [t.get("gain", 0)     for t in tracks if t.get("gain") is not None]
    ranks     = [t.get("rank", 0)     for t in tracks if t.get("rank")]

    def _rng(lst):
        if not lst:
            return "N/A"
        return f"min={min(lst):.1f}, max={max(lst):.1f}, mean={sum(lst)/len(lst):.1f}"

    summary_stats = (
        f"duration_sec: {_rng(durations)}\n"
        f"gain_dB:      {_rng(gains)}\n"
        f"rank:         {_rng(ranks)}"
    )

    _INDEX_CAP = 20
    track_index = [
        {
            "i":           i,
            "id":          t.get("deezer_id"),
            "name":        t.get("name", ""),
            "artist":      t.get("artist", ""),
            "rank":        t.get("rank"),
            "has_preview": bool(t.get("preview_url")),
        }
        for i, t in enumerate(tracks[:_INDEX_CAP])
    ]

    return (
        f"Original question: {question}\n\n"
        f"Dataset loaded in analysis tools: {n} tracks\n"
        f"Schema fields: {schema_fields}\n\n"
        f"Aggregate stats (for orientation only — use tools for full analysis):\n"
        f"{summary_stats}\n\n"
        f"Track index (top {_INDEX_CAP} of {n} — pick representative_tracks from this list only):\n"
        f"{json.dumps(track_index, default=str)}\n\n"
        f"Now call the analysis tools to explore all {n} tracks. "
        f"Every finding must contain at least one concrete number with a unit "
        f"(e.g. '198s', '15%', 'r=-0.41', 'rank 18 vs 21')."
    )


def _build_eda_refinement_msg(
    question: str,
    tracks: list[dict],
    existing_findings: list[dict],
    specific_count: int,
    iteration: int,
) -> str:
    """Build the message for a refinement pass."""
    covered_categories = {f.get("category", "") for f in existing_findings}
    all_categories = {
        "duration_trend", "popularity_driver", "cluster",
        "anomaly", "genre_characteristic", "tempo_pattern",
    }
    missing = all_categories - covered_categories

    n = len(tracks)
    schema_fields = list(tracks[0].keys()) if tracks else []
    durations = [t.get("duration", 0) for t in tracks if t.get("duration")]
    gains     = [t.get("gain", 0)     for t in tracks if t.get("gain") is not None]

    def _rng(lst):
        if not lst:
            return "N/A"
        return f"min={min(lst):.1f}, max={max(lst):.1f}, mean={sum(lst)/len(lst):.1f}"

    _INDEX_CAP = 20
    track_index = [
        {"i": i, "id": t.get("deezer_id"), "name": t.get("name", ""),
         "artist": t.get("artist", ""), "rank": t.get("rank"),
         "has_preview": bool(t.get("preview_url"))}
        for i, t in enumerate(tracks[:_INDEX_CAP])
    ]

    return (
        f"REFINEMENT PASS {iteration} — Original question: {question}\n\n"
        f"Dataset: {n} tracks. Schema: {schema_fields}\n"
        f"duration_sec: {_rng(durations)}  |  gain_dB: {_rng(gains)}\n\n"
        f"Findings so far ({len(existing_findings)} total, {specific_count} specific):\n"
        f"{json.dumps(existing_findings, default=str)}\n\n"
        f"You need {max(0, MIN_SPECIFIC_FINDINGS - specific_count)} MORE findings with concrete "
        f"numbers + units (e.g. '198s', '15%', 'r=-0.41'). Bare counts like '50 tracks' do NOT qualify.\n"
        f"Uncovered categories: {', '.join(missing) if missing else 'all covered — go deeper'}\n\n"
        f"Call more tools (adk_run_python_analysis is especially useful for custom metrics).\n"
        f"Do NOT repeat findings already listed. Return ONLY new findings in the same JSON schema.\n\n"
        f"Track index for representative_tracks (top {_INDEX_CAP} of {n}):\n"
        f"{json.dumps(track_index, default=str)}"
    )


def run_eda(collected: dict, question: str, progress_cb=None) -> tuple[dict, list[dict]]:
    """
    Step 2 — Explore & Analyze (EDA).

    CODE-LEVEL iterative refinement loop:
      - Run EDA agent
      - Count findings that contain specific numbers (_finding_is_specific)
      - If < MIN_SPECIFIC_FINDINGS, run a refinement agent pass with gap context
      - Repeat up to MAX_EDA_ITERATIONS times
      - Merge findings across iterations; deduplicate by text
    """
    tracks = collected.get("tracks", [])
    if not tracks:
        return {
            "findings": [{
                "id": 1, "category": "error",
                "finding": (
                    "Data collection returned no tracks. The Deezer API may be "
                    "rate-limited or the query matched nothing. Please try a different query."
                ),
                "evidence": {},
            }],
            "charts": [], "representative_tracks": [], "tracks": [],
        }, []

    _set_eda_tracks(tracks)
    _reset_charts_cache()

    all_logs: list[dict] = []
    all_findings: list[dict] = []
    final_result: dict = {}

    # ── Initial EDA pass ──────────────────────────────────────
    initial_msg = _build_eda_initial_msg(question, tracks)
    eda_agent = _make_eda_agent(refinement_mode=False)
    text, logs = _run_adk_agent(agent=eda_agent, message=initial_msg, progress_cb=progress_cb)
    all_logs.extend(logs)
    result = _parse_json(text)
    final_result = result
    all_findings.extend(result.get("findings", []))

    # ── CODE-LEVEL refinement while-loop ──────────────────────
    iteration = 0
    while iteration < MAX_EDA_ITERATIONS:
        specific = [f for f in all_findings if _finding_is_specific(f)]
        if len(specific) >= MIN_SPECIFIC_FINDINGS:
            break

        iteration += 1
        if progress_cb:
            try:
                progress_cb(
                    "eda_refinement_loop",
                    f"iteration {iteration}: {len(specific)}/{MIN_SPECIFIC_FINDINGS} "
                    f"specific findings — running refinement pass",
                )
            except Exception:
                pass

        refinement_msg = _build_eda_refinement_msg(
            question=question,
            tracks=tracks,
            existing_findings=all_findings,
            specific_count=len(specific),
            iteration=iteration,
        )
        refinement_agent = _make_eda_agent(refinement_mode=True)
        ref_text, ref_logs = _run_adk_agent(
            agent=refinement_agent,
            message=refinement_msg,
            progress_cb=progress_cb,
        )
        all_logs.extend(ref_logs)
        ref_result = _parse_json(ref_text)
        new_findings = ref_result.get("findings", [])

        existing_texts = {f.get("finding", "").strip().lower() for f in all_findings}
        for f in new_findings:
            if f.get("finding", "").strip().lower() not in existing_texts:
                all_findings.append(f)
                existing_texts.add(f.get("finding", "").strip().lower())

        if not final_result.get("representative_tracks") and ref_result.get("representative_tracks"):
            final_result["representative_tracks"] = ref_result["representative_tracks"]

    # Re-number finding IDs sequentially
    for idx, f in enumerate(all_findings, start=1):
        f["id"] = idx

    final_result["findings"] = all_findings
    final_result["eda_iterations"] = iteration + 1

    charts = _get_eda_charts()
    if charts:
        final_result["charts"] = charts

    if not final_result.get("tracks"):
        final_result["tracks"] = collected.get("tracks", [])

    return final_result, all_logs


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric strings from text with rounded variants for fuzzy matching."""
    raw_nums = re.findall(r'-?\d+\.?\d*', text)
    result: set[str] = set()
    for n in raw_nums:
        result.add(n)
        try:
            f = float(n)
            result.add(str(round(f, 1)))
            result.add(str(int(f)))
        except ValueError:
            pass
    return result


def _verify_hypothesis_grounding(
    hypothesis: dict,
    eda_findings: list[dict],
) -> tuple[bool, list[str]]:
    """
    CODE-LEVEL grounding check — deterministic Python, no LLM involved.
    Checks that numbers cited in the hypothesis appear in EDA findings.
    Returns (is_grounded, unverified_numbers).
    is_grounded = True when unverified_ratio < 0.5
    """
    eda_text = " ".join(
        f.get("finding", "") + " " + json.dumps(f.get("evidence", {}), default=str)
        for f in eda_findings
    )
    eda_pool = _extract_numbers(eda_text)

    hyp_text = hypothesis.get("hypothesis", "")
    for ev in hypothesis.get("supporting_evidence", []):
        hyp_text += " " + ev.get("point", "") + " " + ev.get("data", "")

    hyp_numbers = _extract_numbers(hyp_text)
    meaningful = {n for n in hyp_numbers if abs(float(n)) >= 4}
    unverified = [n for n in meaningful if n not in eda_pool]

    ratio = len(unverified) / max(len(meaningful), 1)
    return ratio < 0.5, unverified


def _build_hyp_msg(question: str, eda_result: dict, collected: dict) -> str:
    """Build the hypothesis agent's message. Strips chart blobs to save tokens."""
    trimmed_eda = {k: v for k, v in eda_result.items() if k not in ("charts", "tracks")}
    payload = json.dumps(_sanitize(trimmed_eda), default=str)[:10000]

    tracks = collected.get("tracks", [])
    _HYP_INDEX_CAP = 20
    track_index = [
        {
            "i":           i,
            "id":          t.get("deezer_id"),
            "name":        t.get("name", ""),
            "artist":      t.get("artist", ""),
            "rank":        t.get("rank"),
            "has_preview": bool(t.get("preview_url")),
        }
        for i, t in enumerate(tracks[:_HYP_INDEX_CAP])
    ]
    return (
        f"Original user question: {question}\n\n"
        f"EDA findings and representative tracks:\n{payload}\n\n"
        f"Track index (top {_HYP_INDEX_CAP} of {len(tracks)}) — pick highlight_tracks ONLY from this list "
        f"using exact name/artist/deezer_id:\n"
        f"{json.dumps(track_index, default=str)}\n"
        f"Pick 5-7 highlight_tracks. Prefer tracks with has_preview=true."
    )


def run_hypothesis(
    eda_result: dict,
    question: str,
    collected: dict,
    progress_cb=None,
) -> tuple[dict, list[dict]]:
    """
    Step 3 — Hypothesize.

    Generator-Critic pattern with feedback loop:
      Round 0: HypothesisAgent generates
               → CODE-LEVEL grounding check (_verify_hypothesis_grounding)
               → HypothesisCriticAgent evaluates (ACCEPT / REJECT)
               → If REJECT: HypothesisAgent revises with feedback
      Round 1: Revision → grounding check → critic re-evaluates
      Round 2: Final revision, accepted regardless.
    """
    all_logs: list[dict] = []
    hypothesis: dict = {}
    critique: dict = {}
    eda_findings = eda_result.get("findings", [])

    hyp_msg = _build_hyp_msg(question, eda_result, collected)

    for round_num in range(MAX_CRITIC_ROUNDS):
        # ── Generator pass ────────────────────────────────────
        if round_num == 0:
            gen_msg = hyp_msg
        else:
            gen_msg = (
                f"Your previous hypothesis was REJECTED.\n\n"
                f"Issues:\n{json.dumps(critique.get('issues', []), indent=2)}\n\n"
                f"Specific gaps to fix:\n{json.dumps(critique.get('specific_gaps', []), indent=2)}\n\n"
                f"Previous hypothesis JSON:\n{json.dumps(_sanitize(hypothesis), default=str)}\n\n"
                f"Revise to fix every issue. Every number you cite MUST appear verbatim "
                f"in the EDA findings below.\n\n"
                f"EDA findings (ground truth — only use numbers from here):\n"
                f"{json.dumps(_sanitize(eda_findings), default=str)}\n\n"
                f"Track index for highlight_tracks (top 20 of {len(collected.get('tracks', []))}):\n"
                f"{json.dumps([{'i': i, 'id': t.get('deezer_id'), 'name': t.get('name',''), 'artist': t.get('artist',''), 'rank': t.get('rank'), 'has_preview': bool(t.get('preview_url'))} for i, t in enumerate(collected.get('tracks', [])[:20])], default=str)}"
            )

        if progress_cb:
            try:
                progress_cb(
                    "hypothesis_generation",
                    f"round {round_num + 1}/{MAX_CRITIC_ROUNDS}: generating hypothesis",
                )
            except Exception:
                pass

        gen_agent = _make_hypothesis_agent()
        gen_text, gen_logs = _run_adk_agent(
            agent=gen_agent, message=gen_msg, progress_cb=progress_cb
        )
        all_logs.extend(gen_logs)
        hypothesis = _parse_json(gen_text)

        # Skip critic on final round — accept as-is
        if round_num >= MAX_CRITIC_ROUNDS - 1:
            break

        # ── CODE-LEVEL grounding check (deterministic, no LLM) ───────
        is_grounded, unverified = _verify_hypothesis_grounding(hypothesis, eda_findings)

        if not is_grounded:
            critique = {
                "verdict": "REJECT",
                "issues": [
                    f"CODE-LEVEL GROUNDING FAILURE: {len(unverified)} number(s) cited in "
                    f"the hypothesis cannot be found in any EDA finding. "
                    f"Unverified values: {unverified}. "
                    f"You must ONLY use numbers that appear in the EDA findings provided."
                ],
                "specific_gaps": [
                    f"Number {n!r} not found in EDA findings" for n in unverified
                ],
            }
            if progress_cb:
                try:
                    progress_cb(
                        "hypothesis_grounding_fail",
                        f"round {round_num + 1}: {len(unverified)} unverified numbers — forcing revision",
                    )
                except Exception:
                    pass
            continue  # Skip LLM critic this round

        # ── LLM Critic pass (only if grounding check passed) ──────────
        critic_msg = (
            f"EDA findings (ground truth):\n"
            f"{json.dumps(_sanitize(eda_findings), default=str)}\n\n"
            f"Hypothesis to evaluate:\n{json.dumps(_sanitize(hypothesis), default=str)}"
        )
        if progress_cb:
            try:
                progress_cb(
                    "hypothesis_critique",
                    f"round {round_num + 1}: grounding ✓ — critic evaluating",
                )
            except Exception:
                pass

        critic_agent = _make_critic_agent()
        crit_text, crit_logs = _run_adk_agent(
            agent=critic_agent, message=critic_msg, progress_cb=progress_cb
        )
        all_logs.extend(crit_logs)
        critique = _parse_json(crit_text)

        verdict = critique.get("verdict", "ACCEPT").upper()
        if progress_cb:
            try:
                progress_cb("hypothesis_verdict", f"round {round_num + 1} verdict: {verdict}")
            except Exception:
                pass

        if verdict == "ACCEPT":
            break
        # REJECT → loop continues with critique feedback

    return hypothesis, all_logs


# ──────────────────────────────────────────────────────────────
# Top-level orchestrator (used for non-streaming calls)
# ──────────────────────────────────────────────────────────────

def run_pipeline(question: str) -> dict[str, Any]:
    """
    Main orchestrator. Chains the three ADK agents with handoffs.
    MusicCollectorAgent → MusicEDAAgent → HypothesisAgent (+ HypothesisCriticAgent)
    """
    result: dict[str, Any] = {"question": question, "steps": {}, "tool_logs": []}

    collected, c_logs = run_collector(question)
    result["steps"]["collect"] = collected
    result["tool_logs"].extend(c_logs)

    time.sleep(3)

    eda, e_logs = run_eda(collected, question)
    result["steps"]["eda"] = eda
    result["tool_logs"].extend(e_logs)

    time.sleep(3)

    hypothesis, h_logs = run_hypothesis(eda, question, collected)
    result["steps"]["hypothesis"] = hypothesis
    result["tool_logs"].extend(h_logs)

    return result