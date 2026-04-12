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

Framework: Google ADK (google-adk)
Model: gemini-2.5-flash via Vertex AI
Auth: Application Default Credentials (gcloud auth application-default login)
"""

from __future__ import annotations

import json
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
# Structure: round 0 → generate + critique, round 1 → revise + critique,
# round 2 → final revision, accept without critique.
# This means the Critic evaluates TWICE (rounds 0 and 1) before we give up.
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
    CODE-LEVEL gate for the iterative refinement loop.
    """
    import re
    text = finding.get("finding", "")


    # Pattern 1: number + meaningful unit (with optional whitespace before unit)
    has_unit = bool(re.search(
        r'\d+\.?\d*\s*(%|seconds?|s\b|dB\b|ms\b)',
        text, re.IGNORECASE
    )) or bool(re.search(
        r'r\s*=\s*-?\d+\.?\d*',   # r = -0.41 or r=-0.41
        text, re.IGNORECASE
    ))

    # Pattern 2: comparative phrasing with numbers
    has_comparison = bool(re.search(
        r'\d+\.?\d*\s*(vs\.?|versus|compared\s+to)\s*\d+\.?\d*',
        text, re.IGNORECASE
    ))

    # Pattern 3: statistical terms followed by a number (within 40 chars)
    has_stat = bool(re.search(
        r'(average|avg|median|mean|correlation|std|deviation|percentile|quartile|rate|pct|percent)'
        r'.{0,40}\d+\.?\d*',
        text, re.IGNORECASE
    ))

    # Pattern 4: number followed by statistical context
    has_stat_reverse = bool(re.search(
        r'\d+\.?\d*\s*(tracks?|songs?|artists?)\s+(are|have|show|rank)',
        text, re.IGNORECASE
    ))

    return has_unit or has_comparison or has_stat or has_stat_reverse




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
    """Search Deezer for top tracks matching a genre/mood/era tag."""
    result = get_tag_top_tracks(tag=tag, limit=limit)
    return json.dumps(_sanitize(result), default=str)


def adk_get_artist_top_tracks(artist_name: str, limit: int = 50) -> str:
    """Get an artist's top tracks on Deezer with rank and preview URLs."""
    result = get_artist_top_tracks(artist_name=artist_name, limit=limit)
    return json.dumps(_sanitize(result), default=str)


def adk_get_tracks_details(deezer_ids: list[int]) -> str:
    """Batch-fetch full track details for a list of Deezer track IDs."""
    result = get_tracks_details(deezer_ids=deezer_ids)
    return json.dumps(_sanitize(result), default=str)


# ── Analysis tools ────────────────────────────────────────────

def adk_compute_feature_statistics() -> str:
    """Compute descriptive statistics (mean/median/std/min/max) for duration, gain, rank, popularity."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = compute_feature_statistics(tracks_with_features=tracks)
    return json.dumps(_sanitize(result), default=str)


def adk_compute_feature_correlations() -> str:
    """Compute Pearson correlations between features and popularity. Returns strongest predictor."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = compute_feature_correlations(tracks_with_features=tracks)
    return json.dumps(_sanitize(result), default=str)


def adk_detect_clusters(n_clusters: int = 3) -> str:
    """K-means cluster collected tracks in duration × gain space."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = detect_clusters(tracks_with_features=tracks, n_clusters=n_clusters)
    return json.dumps(_sanitize(result), default=str)


def adk_run_python_analysis(code: str) -> str:
    """Execute custom pandas/numpy code. Pre-injected: df, pd, np, stats. Set result = {...}."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = run_python_analysis(tracks_with_features=tracks, code=code)
    return json.dumps(_sanitize(result), default=str)


def adk_compute_derived_metrics() -> str:
    """Compute explicit rate, artist diversity, release recency, duration buckets, rank-tier comparisons."""
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    result = compute_derived_metrics(tracks_with_features=tracks)
    return json.dumps(_sanitize(result), default=str)


# ── Chart tools ───────────────────────────────────────────────

def _store_chart(result: dict) -> str:
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
    """Generate a scatter plot of all collected tracks."""
    result = generate_scatter_chart(
        tracks_with_features=_get_eda_tracks(),
        x_feature=x_feature,
        y_feature=y_feature,
        color_by=color_by,
        title=title,
    )
    return _store_chart(_sanitize(result))


def adk_generate_radar_chart(track_indices: str = "0,1,2", title: str = "Track Feature Comparison") -> str:
    """Generate a radar chart comparing features of 2-3 tracks."""
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
    """Generate a horizontal bar chart."""
    labels = labels_json if isinstance(labels_json, list) else json.loads(labels_json)
    values = values_json if isinstance(values_json, list) else json.loads(values_json)
    data = dict(zip(labels, values))
    result = generate_bar_chart(data=data, title=title, xlabel=xlabel, ylabel=ylabel)
    return _store_chart(_sanitize(result))


def adk_generate_histogram(feature: str, title: str = "", xlabel: str = "", bins: int = 8) -> str:
    """Generate a histogram of a single feature distribution."""
    result = generate_histogram(
        tracks_with_features=_get_eda_tracks(),
        feature=feature,
        title=title,
        xlabel=xlabel,
        bins=bins,
    )
    return _store_chart(_sanitize(result))


def adk_generate_correlation_heatmap(title: str = "Feature Correlation Matrix") -> str:
    """Generate a Pearson correlation heatmap."""
    result = generate_correlation_heatmap(
        tracks_with_features=_get_eda_tracks(),
        title=title,
    )
    return _store_chart(_sanitize(result))


def adk_generate_trend_line(periods_json: str, values_json: str, labels_json: str, title: str, ylabel: str) -> str:
    """Generate a line chart showing how a feature changes across groups/periods."""
    periods     = periods_json  if isinstance(periods_json,  list) else json.loads(periods_json)
    values      = values_json   if isinstance(values_json,   list) else json.loads(values_json)
    labels_list = labels_json   if isinstance(labels_json,   list) else json.loads(labels_json)
    time_series = [{"period": p, "value": v, "label": l} for p, v, l in zip(periods, values, labels_list)]
    result = generate_trend_line(time_series=time_series, title=title, ylabel=ylabel)
    return _store_chart(_sanitize(result))


# ── Wikipedia tools ───────────────────────────────────────────

def adk_get_artist_wikipedia_summary(artist_name: str) -> str:
    """Fetch Wikipedia biography and genre description for a music artist."""
    result = get_artist_wikipedia_summary(artist_name=artist_name)
    return json.dumps(_sanitize(result), default=str)


def adk_get_genre_wikipedia_overview(genre: str) -> str:
    """Fetch Wikipedia overview for a music genre."""
    result = get_genre_wikipedia_overview(genre=genre)
    return json.dumps(_sanitize(result), default=str)


# ── Parallel analysis tool ────────────────────────────────────

def adk_parallel_analyze() -> str:
    """Run feature statistics AND correlations in parallel via ThreadPoolExecutor."""
    import concurrent.futures
    tracks = _get_eda_tracks()
    if not tracks:
        return json.dumps({"error": "Track cache is empty."})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_stats = executor.submit(compute_feature_statistics, tracks)
        f_corr  = executor.submit(compute_feature_correlations, tracks)
        stats_result = f_stats.result()
        corr_result  = f_corr.result()
    combined = {"statistics": stats_result, "correlations": corr_result}
    return json.dumps(_sanitize(combined), default=str)


# ── Artifact tool ─────────────────────────────────────────────

def adk_save_artifact(name: str, content: str, kind: str = "markdown") -> str:
    """Save the final analysis report as a markdown file."""
    result = save_artifact(name=name, content=content, kind=kind)
    return json.dumps(_sanitize(result), default=str)


# ──────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────

COLLECTOR_SYSTEM = """
You are the MusicCollectorAgent, Step 1 of a music analysis pipeline.
Your ONLY job is to collect real-world music data from Deezer AND Wikipedia.

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
   Each call returns fully enriched tracks (gain, album_art, release_date, explicit).
3. ALSO fetch Wikipedia context — ONLY when relevant:
   - Specific artist → adk_get_artist_wikipedia_summary(artist_name="...")
   - Specific genre  → adk_get_genre_wikipedia_overview(genre="...")
   - Global charts   → SKIP Wikipedia entirely
4. Compute popularity for each track as: popularity = max(0, 100 - rank).

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



EDA_SYSTEM = """
You are the MusicEDAAgent, Step 2 of a music analysis pipeline.

You receive a summary of the collected data (field names + aggregate stats only).
The FULL track dataset is pre-loaded in the analysis tools — call them directly,
do NOT ask for raw track data.

Available numeric features: rank, popularity, duration (seconds), gain (dB).
Also available: release_date (YYYY-MM-DD), explicit (bool), artist (string).

Perform thorough Exploratory Data Analysis. Find SPECIFIC, NON-OBVIOUS patterns.
Every finding MUST contain at least one concrete number (count, %, seconds, dB, r=).
Do NOT output generic summaries like “most songs have similar duration.”
DO find things like “tracks shorter than 200s rank 15% higher on average (rank 18 vs 21).”

═══════════════════════════════════════════════════════════════
STEP A — ALWAYS run these two calls first (parallel):

1. adk_parallel_analyze()         ← stats + correlations simultaneously
1. adk_compute_derived_metrics()  ← explicit rate, recency, duration buckets, rank-tiers
   ═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════
STEP B — ADAPT based on the question type (read the original question carefully):

IF question asks about a SPECIFIC ARTIST (e.g. “Taylor Swift”, “The Weeknd”):

- Focus: How consistent is this artist's sound? Do longer/louder tracks rank higher?
- Required extra calls:
  adk_run_python_analysis(code=”””
  result = {
  'duration_std': float(df['duration'].std()),
  'gain_std': float(df['gain'].std()),
  'duration_rank_corr': float(df[['duration','rank']].corr().iloc[0,1]),
  'gain_rank_corr': float(df[['gain','rank']].corr().iloc[0,1]),
  'release_years': df['release_date'].str[:4].value_counts().head(5).to_dict(),
  }
  “””)
- Chart focus: scatter(x=duration, y=gain, color_by=rank) + histogram(feature=duration)

IF question asks about a GENRE or MOOD (e.g. “hip-hop”, “indie pop”, “Latin”):

- Focus: What audio signature defines charting tracks in this genre?
- Required extra calls:
  adk_detect_clusters(n_clusters=3)
  adk_run_python_analysis(code=”””
  result = {
  'explicit_by_rank_tier': {
  'top25_explicit_pct': float(df[df['rank']<=25]['explicit'].mean()*100),
  'bottom25_explicit_pct': float(df[df['rank']>df['rank'].quantile(0.75)]['explicit'].mean()*100),
  },
  'duration_by_cluster': df.groupby(df['duration'].apply(
  lambda x: 'short' if x<180 else ('medium' if x<240 else 'long')
  ))['popularity'].mean().to_dict(),
  }
  “””)
- Chart focus: correlation heatmap + bar chart of cluster avg popularity

IF question is a YES/NO or COMPARISON (e.g. “Are shorter songs more popular?”):

- Focus: Directly test the stated hypothesis with data.
- Required extra calls:
  adk_run_python_analysis(code=”””
  import numpy as np
  median_dur = df['duration'].median()
  short = df[df['duration'] < median_dur]
  long_ = df[df['duration'] >= median_dur]
  result = {
  'median_duration_seconds': float(median_dur),
  'short_track_avg_rank': float(short['rank'].mean()),
  'long_track_avg_rank': float(long_['rank'].mean()),
  'short_track_count': int(len(short)),
  'long_track_count': int(len(long_)),
  'rank_difference': float(long_['rank'].mean() - short['rank'].mean()),
  'short_wins': bool(short['rank'].mean() < long_['rank'].mean()),
  }
  “””)
- Chart focus: histogram(feature=duration) + scatter(x=duration, y=rank)

IF question asks about GLOBAL CHARTS or CURRENT TRENDS (no specific artist/genre):

- Focus: What cross-genre patterns dominate the chart right now?
- Required extra calls:
  adk_detect_clusters(n_clusters=4)
  adk_run_python_analysis(code=”””
  result = {
  'top10_vs_bottom10_duration': {
  'top10_avg_s': float(df.nsmallest(10,'rank')['duration'].mean()),
  'bottom10_avg_s': float(df.nlargest(10,'rank')['duration'].mean()),
  },
  'explicit_rate_pct': float(df['explicit'].mean()*100),
  'recency': {
  'last_12mo_pct': float((df['release_date'] >= '2024-01-01').mean()*100),
  },
  'gain_quartile_popularity': df.groupby(
  df['gain'].apply(lambda g: 'loud' if g < -10 else ('medium' if g < -7 else 'quiet'))
  )['popularity'].mean().to_dict(),
  }
  “””)
- Chart focus: correlation heatmap + scatter(x=duration, y=gain, color_by=popularity)
  ═══════════════════════════════════════════════════════════════

STEP C — Always generate AT LEAST 3 charts:

- ALWAYS: adk_generate_correlation_heatmap(title=“Feature Correlation Matrix”)
- ALWAYS: adk_generate_scatter_chart(x_feature=“duration”, y_feature=“gain”, color_by=“popularity”)
- PLUS 1-2 more chosen from STEP B guidance above

For representative_tracks, pick 6-8 tracks from the track_index provided.
Copy name, artist, deezer_id EXACTLY. Leave preview_url/deezer_url/album_art empty.

Formatting: wrap key numbers in **double asterisks** so they render bold.

Output ONLY this JSON (no prose, no fences):
{
“findings”: [
{
“id”: 1,
“category”: “tempo_pattern|popularity_driver|genre_characteristic|duration_trend|anomaly|cluster”,
“finding”: “”,
“evidence”: {}
}
],
“charts”: [],
“representative_tracks”: [
{
“name”: “”,
“artist”: “”,
“deezer_id”: 0,
“preview_url”: “”,
“deezer_url”: “”,
“album_art”: “”,
“why_representative”: “”,
“key_features”: {“duration”: 0, “gain”: 0.0, “popularity”: 0}
}
],
“tracks”: []
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

HYPOTHESIS_SYSTEM = """
You are the HypothesisAgent (Generator), Step 3 of a music analysis pipeline.

You receive EDA findings and must produce a compelling, data-grounded music hypothesis.

Rules:
- Your hypothesis must be SPECIFIC and NON-OBVIOUS.
  BAD:  "Popular songs tend to be energetic and danceable."
  GOOD: "Loudness (gain) correlates with rank at r=−0.41, but only for tracks under 210s:
         shorter loud tracks hit the top quartile 65% of the time vs 28% for longer ones."
- Every claim must cite a specific number from the EDA findings.
- Do NOT invent data — only use what EDA found.
- List 1-2 alternative explanations that could challenge your hypothesis.
- Wrap key numbers, track names, artist names in **double asterisks** for bold rendering.
- The "hypothesis" field is MANDATORY — never empty.

Instructions:
1. If the question is yes/no, write a concise direct_answer (1-2 sentences).
2. Form ONE clear, non-trivial hypothesis grounded in data numbers.
3. Pick 5-7 highlight_tracks from the track_index. Copy name/artist/deezer_id EXACTLY.
   Leave preview_url/deezer_url/album_art empty — server fills them.
4. Write summary_paragraph as 1-2 SHORT plain-language sentences, NO numbers or jargon.
5. Call adk_save_artifact to persist the full markdown report.

Output ONLY this JSON (no prose, no fences):
{
  "direct_answer": "",
  "hypothesis": "",
  "confidence": "Low|Medium|High",
  "confidence_reasoning": "",
  "supporting_evidence": [{"point": "", "data": ""}],
  "alternative_explanations": ["", ""],
  "summary_paragraph": "",
  "highlight_tracks": [
    {
      "name": "", "artist": "", "deezer_id": 0,
      "preview_url": "", "deezer_url": "", "album_art": "",
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
    Collects up to 100 tracks to ensure a non-trivial dataset.
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

    # ── Global chart keywords ─────────────────────────────────
    global_keywords = ["global", "top", "chart", "hit", "popular", "trending",
                       "billboard", "worldwide", "right now", "current"]
    is_global = any(kw in q for kw in global_keywords)

    # ── Genre/mood ────────────────────────────────────────────
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

    # ── Default: global top chart ─────────────────────────────
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

    CRITICAL DESIGN: We do NOT dump the full track list into the message.
    Instead we send only:
      - field names (schema)
      - aggregate summary stats (min/max/mean for key fields)
      - a compact track index with just id/name/artist/rank for representative_track selection
    The full data is pre-loaded in thread-local cache and accessed via tool calls.
    This keeps the context lean and forces the agent to use analysis tools.
    """
    n = len(tracks)

    # Schema only — field names, no values
    schema_fields = list(tracks[0].keys()) if tracks else []

    # Aggregate summary — a few lines of stats the agent can orient itself with
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

    # Compact track index — ONLY for representative_track/highlight_track selection.
    # Capped at 20 entries: the agent uses tools to analyse all 100 tracks,
    # it only needs a short sample list to name representative_tracks from.
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

    # Same lean context: schema + stats only, NO full track dump
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

    Full track data is stored in thread-local cache and NEVER dumped into
    the agent message — only schema + aggregate stats are passed as context.
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

    # Pre-load full data into thread-local cache (analysis tools read from here)
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
    new_findings = result.get("findings", [])
    all_findings.extend(new_findings)

    # ── CODE-LEVEL refinement while-loop ──────────────────────
    iteration = 0
    while iteration < MAX_EDA_ITERATIONS:
        specific = [f for f in all_findings if _finding_is_specific(f)]
        if len(specific) >= MIN_SPECIFIC_FINDINGS:
            # Stopping condition met — enough specific findings
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

        # Deduplicate: skip findings whose text is already in all_findings
        existing_texts = {f.get("finding", "").strip().lower() for f in all_findings}
        for f in new_findings:
            if f.get("finding", "").strip().lower() not in existing_texts:
                all_findings.append(f)
                existing_texts.add(f.get("finding", "").strip().lower())

        # Carry over representative_tracks from refinement pass if initial was empty
        if not final_result.get("representative_tracks") and ref_result.get("representative_tracks"):
            final_result["representative_tracks"] = ref_result["representative_tracks"]

    # Re-number finding IDs sequentially
    for idx, f in enumerate(all_findings, start=1):
        f["id"] = idx

    final_result["findings"] = all_findings
    final_result["eda_iterations"] = iteration + 1

    # Inject charts from thread-local cache
    charts = _get_eda_charts()
    if charts:
        final_result["charts"] = charts

    # Pass-through full tracks for hypothesis stage
    if not final_result.get("tracks"):
        final_result["tracks"] = collected.get("tracks", [])

    return final_result, all_logs


def _build_hyp_msg(question: str, eda_result: dict, collected: dict) -> str:
    """
    Build the hypothesis agent's message.
    Strips chart blobs and full track data to save tokens.
    Passes EDA findings + a compact track index.
    """
    trimmed_eda = {k: v for k, v in eda_result.items() if k not in ("charts", "tracks")}
    payload = json.dumps(_sanitize(trimmed_eda), default=str)[:10000]

    tracks = collected.get("tracks", [])
    track_index = [
        {
            "i":          i,
            "id":         t.get("deezer_id"),
            "name":       t.get("name", ""),
            "artist":     t.get("artist", ""),
            "rank":       t.get("rank"),
            "has_preview": bool(t.get("preview_url")),
        }
        for i, t in enumerate(tracks)
    ]
    return (
        f"Original user question: {question}\n\n"
        f"EDA findings and representative tracks:\n{payload}\n\n"
        f"Track index — pick highlight_tracks ONLY from this list "
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

    TRUE Generator-Critic pattern with feedback loop:
      Round 1: HypothesisAgent generates hypothesis
              → HypothesisCriticAgent evaluates (ACCEPT / REJECT)
              → If REJECT: HypothesisAgent revises with critic feedback
      Round 2: HypothesisAgent revises
              → HypothesisCriticAgent re-evaluates
              → Accept regardless after MAX_CRITIC_ROUNDS
    The Critic can actually REJECT and force a rewrite — this is not just a
    one-way handoff.
    """
    all_logs: list[dict] = []
    hypothesis: dict = {}
    critique: dict = {}

    hyp_msg = _build_hyp_msg(question, eda_result, collected)

    for round_num in range(MAX_CRITIC_ROUNDS):
        # ── Generator pass ────────────────────────────────────
        if round_num == 0:
            gen_msg = hyp_msg
        else:
            # Critic rejected — give generator the feedback so it can revise
            gen_msg = (
                f"Your previous hypothesis was REJECTED by the critic.\n\n"
                f"Critic issues:\n{json.dumps(critique.get('issues', []), indent=2)}\n\n"
                f"Specific gaps to fix:\n{json.dumps(critique.get('specific_gaps', []), indent=2)}\n\n"
                f"Previous hypothesis JSON:\n{json.dumps(_sanitize(hypothesis), default=str)}\n\n"
                f"Revise to fix every issue. Every claim must cite a number from EDA findings.\n\n"
                f"EDA findings (ground truth):\n"
                f"{json.dumps(_sanitize(eda_result.get('findings', [])), default=str)}\n\n"
                f"Track index for highlight_tracks:\n"
                f"{json.dumps([{'i': i, 'id': t.get('deezer_id'), 'name': t.get('name',''), 'artist': t.get('artist',''), 'rank': t.get('rank'), 'has_preview': bool(t.get('preview_url'))} for i, t in enumerate(collected.get('tracks', []))], default=str)}"
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

        # ── Critic pass (skip on final round to save a call) ──
        if round_num < MAX_CRITIC_ROUNDS - 1:
            critic_msg = (
                f"EDA findings (ground truth):\n"
                f"{json.dumps(_sanitize(eda_result.get('findings', [])), default=str)}\n\n"
                f"Hypothesis to evaluate:\n{json.dumps(_sanitize(hypothesis), default=str)}"
            )
            if progress_cb:
                try:
                    progress_cb(
                        "hypothesis_critique",
                        f"round {round_num + 1}: critic evaluating",
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
                    progress_cb(
                        "hypothesis_verdict",
                        f"round {round_num + 1} verdict: {verdict}",
                    )
                except Exception:
                    pass

            if verdict == "ACCEPT":
                # Critic satisfied — no need for further rounds
                break
            # If REJECT, loop continues with the critic feedback injected above
        # If on final round, accept whatever we have

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
