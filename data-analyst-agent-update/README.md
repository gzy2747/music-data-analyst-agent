# Music Analyst — Multi-Agent Music Intelligence System

Multi-agent pipeline that analyses music trends using **Deezer** chart data.  
Four specialised AI agents — Collector → EDA → Hypothesis Generator → Hypothesis Critic — implement the full Collect → EDA → Hypothesize lifecycle.

Built on **Google ADK** (`google-adk`) + **Gemini 2.5 Flash via Vertex AI**, served via FastAPI with SSE streaming.

-----

## Live URL

**<https://musicanalyst-32zzvwbxsq-uc.a.run.app>**

Deployed on Google Cloud Run (us-central1) — no login required, open to the public.

-----

## How to Run Locally

```bash
# 1. Install dependencies (Python 3.11+ required)
pip install -e .

# 2. Authenticate with Google Cloud (one-time)
gcloud auth login
gcloud auth application-default login

# 3. Start the server
python main.py

# 4. Open in browser
open http://127.0.0.1:8080
```

> No API keys needed — the app uses **Vertex AI** via your Google Cloud account and **Deezer’s public API** (no key required).

-----

## Environment Variables

All configuration is set in `.env`:

```
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
GOOGLE_CLOUD_LOCATION=us-central1
```

-----

## Step 1 — Collect

**Agent:** `MusicCollectorAgent`  
**File:** `agents/orchestrator.py` → `run_collector()`, `COLLECTOR_SYSTEM`  
**Framework:** Google ADK `LlmAgent`

### What the agent does

The agent parses the user’s natural-language question to identify genre, mood, artist, or chart type. It then actively fetches live data from **two independent external sources** at runtime — no data is hardcoded or bundled.

### Data Source 1 — Deezer API (primary)

No API key required. All endpoints are public.

The agent selects one of three collection strategies based on the question:

|Question type                        |Deezer endpoint called                             |Tracks returned|
|-------------------------------------|---------------------------------------------------|---------------|
|Specific artist (e.g. “Taylor Swift”)|`/artist/{id}/top` via `get_artist_top_tracks`     |up to 50       |
|Genre/mood (e.g. “hip-hop”, “Latin”) |`/chart/{genre_id}/tracks` via `get_tag_top_tracks`|up to 100      |
|Global / trends                      |`/chart/0/tracks` via `get_top_tracks_chart`       |up to 100      |

After the primary fetch, `_enrich_tracks()` calls `/track/{id}` for every track to add `gain`, `release_date`, `explicit`, and high-res `album_art` — typically **100–200 Deezer API calls per run**.

|Function                              |File                   |What it fetches                                                                                 |
|--------------------------------------|-----------------------|------------------------------------------------------------------------------------------------|
|`get_top_tracks_chart(limit, page)`   |`tools/deezer_tools.py`|Global chart via `/chart/0/tracks`                                                              |
|`get_tag_top_tracks(tag, limit)`      |`tools/deezer_tools.py`|Genre chart via `/chart/{genre_id}/tracks`; falls back to ranked keyword search for unknown tags|
|`get_track_info(track_name, artist)`  |`tools/deezer_tools.py`|Single-track lookup via `/search`                                                               |
|`get_artist_top_tracks(artist, limit)`|`tools/deezer_tools.py`|Artist top tracks via `/artist/{id}/top`                                                        |
|`get_tracks_details(deezer_ids)`      |`tools/deezer_tools.py`|Batch enrichment via `/track/{id}`                                                              |

**Genre routing:** `get_tag_top_tracks` maps genre keywords to Deezer genre IDs (`"hip-hop"` → 116, `"pop"` → 132, `"rock"` → 152, `"latin"` → 197, etc.) and calls `/chart/{genre_id}/tracks` directly — returning actual chart data, not text search results.

### Data Source 2 — Wikipedia REST API (secondary)

A **second, distinct data retrieval method** — entirely separate from Deezer, no API key required.

Called only for artist/genre questions (skipped for global chart queries where biographical context is not relevant).

|Function                                   |File                      |What it fetches                                                                                                      |
|-------------------------------------------|--------------------------|---------------------------------------------------------------------------------------------------------------------|
|`get_artist_wikipedia_summary(artist_name)`|`tools/wikipedia_tools.py`|Biography extract, genre tags, Wikipedia URL via `/api/rest_v1/page/summary/{title}`; uses OpenSearch fallback on 404|
|`get_genre_wikipedia_overview(genre)`      |`tools/wikipedia_tools.py`|Genre history and cultural context                                                                                   |

### Why the dataset is non-trivial

The dataset cannot be hardcoded or pre-loaded into context for three reasons:

1. **Volume and assembly:** Up to 100 chart tracks assembled from **100–200 live Deezer API calls** (1 chart call + 1 enrichment call per track). The full dataset is stored in a thread-local cache (`_set_eda_tracks`) and structurally never passed to any agent — only schema + aggregate stats appear in the agent’s context window. The EDA agent has no way to see the raw track list; it must use tool calls to access any data.
1. **Currency:** Deezer chart rankings change daily. The same query run today vs. tomorrow returns different tracks and ranks.
1. **Query-dependence:** “Taylor Swift”, “hip-hop”, and “global chart” each call a different Deezer endpoint and return completely different datasets. The agent constructs the dataset at runtime based on the question.

### Python fallback

If the ADK agent returns malformed JSON, `run_collector()` calls `_direct_deezer_fallback()` directly — applying the same artist/genre/global routing logic — so the pipeline always has data.

### Track schema

**Base fields** (from `_parse_track()`, via chart/search response):

```json
{
  "name": "luther", "artist": "Kendrick Lamar", "album": "GNX",
  "album_art": "https://cdn-images.dzcdn.net/...",
  "rank": 3, "duration": 232, "gain": -8.3,
  "preview_url": "https://cdnt-preview.dzcdn.net/...",
  "deezer_url": "https://www.deezer.com/track/...", "deezer_id": 123456789
}
```

**Enriched fields** (added by `_enrich_tracks()` via `/track/{id}`):

```json
{ "release_date": "2024-11-22", "explicit": false }
```

**Derived field** (computed by collector agent after collection):

```json
{ "popularity": 97 }
```

`popularity = max(0, 100 - rank)`.

-----

## Data Metrics — Source & Derivation

|Metric                 |Source                      |Derivation                                                                                                                        |
|-----------------------|----------------------------|----------------------------------------------------------------------------------------------------------------------------------|
|`rank`                 |Deezer chart position       |Position in `/chart/0/tracks` or `/chart/{genre_id}/tracks` (1 = top); set by `_parse_track()`                                    |
|`popularity`           |Derived                     |`max(0, 100 - rank)` — added by collector agent                                                                                   |
|`duration`             |Deezer chart + `/track/{id}`|Track length in seconds                                                                                                           |
|`gain`                 |Deezer `/track/{id}`        |Replay gain offset in dB — **more negative = louder master**. Set initially by `_parse_track()`, overwritten by `_enrich_tracks()`|
|`release_date`         |Deezer `/track/{id}`        |`YYYY-MM-DD`; added by `_enrich_tracks()` only                                                                                    |
|`explicit`             |Deezer `/track/{id}`        |Boolean `explicit_lyrics`; added by `_enrich_tracks()` only                                                                       |
|`preview_url`          |Deezer chart/search         |30-second MP3 URL; may be `None`                                                                                                  |
|`album_art`            |Deezer chart + `/track/{id}`|`cover_xl → cover_big → cover_medium`; overwritten by enrichment                                                                  |
|`deezer_id`            |Deezer chart/search         |Numeric track ID used for enrichment and deep-links                                                                               |
|`artist_diversity`     |Derived (EDA)               |Unique artist count, diversity ratio = unique/total                                                                               |
|`release_recency`      |Derived (EDA)               |% released in last 12/24 months                                                                                                   |
|`duration_distribution`|Derived (EDA)               |Short (<3 min) / medium (3–4 min) / long (>4 min) with percentages                                                                |
|`rank_tier_analysis`   |Derived (EDA)               |Avg duration/gain/popularity for top-10 vs bottom-10 ranked tracks                                                                |
|`explicit_rate`        |Derived (EDA)               |Fraction of tracks with `explicit = true`                                                                                         |

-----

## Step 2 — Explore & Analyze (EDA)

**Agent:** `MusicEDAAgent`  
**File:** `agents/orchestrator.py` → `run_eda()`, `EDA_SYSTEM`  
**Framework:** Google ADK `LlmAgent`

### Why the EDA is dynamic

The `EDA_SYSTEM` prompt contains **four explicit question-type branches** that route the agent to different tools and analytical focus areas depending on what the user asked:

|Question type                                             |Extra tools called                                                                                                              |Analytical focus                                              |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
|Specific artist                                           |`adk_run_python_analysis` computing duration std, gain std, duration-rank and gain-rank correlations, release year distribution |Sound consistency; what makes this artist’s tracks rank higher|
|Genre/mood                                                |`adk_detect_clusters` + `adk_run_python_analysis` computing explicit rate by rank tier, duration-bucket popularity means        |What audio signature defines charting tracks in this genre    |
|Yes/No comparison (e.g. “Are shorter songs more popular?”)|`adk_run_python_analysis` doing a median split to directly test the stated hypothesis with counts and average ranks             |Direct numeric answer to the stated comparison                |
|Global chart / trends                                     |`adk_detect_clusters(n_clusters=4)` + `adk_run_python_analysis` computing top-10 vs bottom-10 duration, gain-quartile popularity|Cross-genre patterns dominating the global chart              |

“Are shorter songs more popular?” and “What defines Taylor Swift’s sound?” trigger **different tool calls with different pandas code** — the EDA is not a fixed sequence applied to every question.

### Thread-local data cache

Before the EDA agent runs, all collected tracks are stored in `threading.local()` via `_set_eda_tracks()`. Analysis tools read from this cache directly — they take no data parameters. The agent is structurally forced to use tool calls to access any data: it cannot see the track list from its context window.

### Iterative Refinement Loop (Grab-Bag elective)

`run_eda()` contains a **Python `while` loop** (up to `MAX_EDA_ITERATIONS = 3`) that enforces finding quality:

```
Initial EDA pass (MusicEDAAgent with EDA_SYSTEM)
    ↓
_finding_is_specific() counts findings with concrete numbers + units
    ↓  fewer than MIN_SPECIFIC_FINDINGS (3)?
Build refinement message: existing findings, uncovered categories, shortfall count
    ↓
New MusicEDAAgent instance with EDA_REFINEMENT_SYSTEM (different prompt)
    ↓
Deduplicate new findings by text → merge → re-number IDs
    ↓
Repeat up to MAX_EDA_ITERATIONS times or until stopping condition met
```

`_finding_is_specific()` requires each finding to contain a number with a meaningful unit (`%`, `s`/`seconds`, `dB`, `r=`, or a statistical term like `average`/`median`/`correlation` followed by a number within 40 characters). Bare dataset-size counts (“50 tracks collected”) fail the gate.

> ❌ “Most songs have similar duration.”  
> ✅ “Tracks shorter than 200s rank 15% higher on average (mean rank 18 vs 21 for longer tracks).”

### EDA tools

|Tool                                                |File                     |What it computes                                                                                                               |
|----------------------------------------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|`adk_parallel_analyze()`                            |`agents/orchestrator.py` |Runs `compute_feature_statistics` **and** `compute_feature_correlations` simultaneously via `ThreadPoolExecutor(max_workers=2)`|
|`adk_compute_feature_statistics()`                  |`tools/analysis_tools.py`|Mean/median/std/min/max for duration, gain, rank, popularity; `preview_available_pct`                                          |
|`adk_compute_feature_correlations()`                |`tools/analysis_tools.py`|Pearson r vs popularity for all numeric features; `strongest_predictor`; full correlation matrix                               |
|`adk_compute_derived_metrics()`                     |`tools/analysis_tools.py`|Explicit rate, artist diversity, release recency (% last 12/24 months), duration buckets, rank-tier comparisons                |
|`adk_detect_clusters(n_clusters)`                   |`tools/analysis_tools.py`|K-means on duration × gain (normalised [0,1]); labels clusters by centroid position                                            |
|`adk_run_python_analysis(code)`                     |`tools/analysis_tools.py`|Agent writes pandas/numpy code as string; run via `exec(textwrap.dedent(code))`; pre-injected: `df`, `pd`, `np`, `stats`       |
|`adk_generate_scatter_chart(x, y, color_by)`        |`tools/chart_tools.py`   |Scatter plot; skips if all x or y values identical                                                                             |
|`adk_generate_correlation_heatmap(title)`           |`tools/chart_tools.py`   |Pearson heatmap; diverging coral→teal colormap; r values in each cell                                                          |
|`adk_generate_histogram(feature, bins)`             |`tools/chart_tools.py`   |Feature distribution with median line; skips if < 3 data points                                                                |
|`adk_generate_radar_chart(track_indices)`           |`tools/chart_tools.py`   |Spider chart for 2–3 tracks; Deezer features normalised to [0,1]                                                               |
|`adk_generate_bar_chart(labels_json, values_json)`  |`tools/chart_tools.py`   |Horizontal bar chart; skips if < 2 entries or all-zero                                                                         |
|`adk_generate_trend_line(periods_json, values_json)`|`tools/chart_tools.py`   |Line chart; skips if < 2 points or all values identical                                                                        |

**Chart storage:** Charts stored in `_tl.charts` (thread-local). Model receives only `{"stored": True, "title": "..."}` — never sees raw base64.

-----

## Step 3 — Hypothesize

**Agents:** `HypothesisAgent` (Generator) + `HypothesisCriticAgent` (Critic)  
**File:** `agents/orchestrator.py` → `run_hypothesis()`, `HYPOTHESIS_SYSTEM`, `CRITIC_SYSTEM`  
**Framework:** Google ADK `LlmAgent`

### Generator-Critic Pattern with Feedback Loop

Two distinct agents with different system prompts and responsibilities:

- **`HypothesisAgent`** (`HYPOTHESIS_SYSTEM`): Receives EDA findings and produces a data-grounded music hypothesis JSON. Its only tool is `adk_save_artifact`.
- **`HypothesisCriticAgent`** (`CRITIC_SYSTEM`): Receives the EDA findings and the hypothesis; outputs only ACCEPT or REJECT with specific issues. Has no tools.

```
Round 0: HypothesisAgent generates hypothesis JSON
              ↓
         HypothesisCriticAgent → ACCEPT or REJECT + issues[]
              ↓ (if REJECT)
Round 1: HypothesisAgent revises — critic issues[] injected into message
              ↓
         HypothesisCriticAgent → ACCEPT or REJECT
              ↓ (if still REJECT)
Round 2: HypothesisAgent final revision — no further critic call (accepted regardless)
```

`MAX_CRITIC_ROUNDS = 3`. Critic runs at most twice (rounds 0 and 1).

### What the Critic checks

```
1. Does every claim in "hypothesis" cite a specific number from EDA findings?
2. Is the hypothesis non-obvious (not restating common knowledge)?
3. Does "supporting_evidence" have ≥ 2 entries with concrete data points?
4. Is "hypothesis" non-empty?
```

All four must pass for ACCEPT. On REJECT, `issues` and `specific_gaps` are injected verbatim into the generator’s next message.

### Hypothesis quality enforced

> ❌ “Popular songs tend to be short and loud.”  
> ✅ “Tracks under 200s with gain < −8 dB appear in the top quartile 68% of the time vs 31% for longer, quieter tracks — driven by the ‘Short / Loud’ cluster which contains **34** of the top-50 tracks.”

The hypothesis is grounded in EDA findings only — `HYPOTHESIS_SYSTEM` explicitly instructs “Do NOT invent data — only use what EDA found.” `HypothesisAgent` also calls `adk_save_artifact` to write the full markdown report to disk.

-----

## Core Requirements Checklist

|Requirement            |File + Function                                                        |Notes                                                                                                                                                           |
|-----------------------|-----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**Frontend**           |`static/index.html`                                                    |SSE streaming, pipeline status, charts, audio previews, light/dark mode, analysis history                                                                       |
|**Agent framework**    |`agents/orchestrator.py` → `_make_*_agent()`                           |Google ADK `LlmAgent` + `InMemoryRunner`                                                                                                                        |
|**Tool calling**       |`tools/` directory                                                     |12 EDA tools + 5 collector tools + 1 artifact tool; all return JSON                                                                                             |
|**Non-trivial dataset**|`tools/deezer_tools.py`, `agents/orchestrator.py` → `_set_eda_tracks()`|Up to 100 tracks from 100–200 live API calls; full dataset in thread-local cache, never in agent context; rankings change daily                                 |
|**Multi-agent pattern**|`agents/orchestrator.py` → all `_make_*_agent()` functions             |**Four distinct agents**, two patterns: (1) orchestrator handoff chain Collector→EDA→Hypothesis, (2) Generator-Critic loop HypothesisAgent↔HypothesisCriticAgent|
|**Deployed**           |Cloud Run                                                              |`https://musicanalyst-32zzvwbxsq-uc.a.run.app` — public, no login                                                                                               |
|**README**             |This file                                                              |All three steps described with file + function locations                                                                                                        |

-----

## Grab-Bag Electives

### 1. Iterative Refinement Loop ✅

**File:** `agents/orchestrator.py` → `run_eda()`, `_finding_is_specific()`, `_build_eda_refinement_msg()`

The agent queries, analyzes, identifies gaps in its understanding, and queries again until the stopping condition (`MIN_SPECIFIC_FINDINGS = 3` numeric findings) is met or `MAX_EDA_ITERATIONS = 3` is reached. A distinct agent instance with `EDA_REFINEMENT_SYSTEM` is used for each refinement pass.

### 2. Parallel Execution ✅

**File:** `agents/orchestrator.py` → `adk_parallel_analyze()`

`ThreadPoolExecutor(max_workers=2)` submits `compute_feature_statistics` and `compute_feature_correlations` simultaneously. Both futures submitted before either is awaited; results combined into one JSON response.

### 3. Code Execution ✅

**File:** `tools/analysis_tools.py` → `run_python_analysis()`, wrapped as `adk_run_python_analysis(code)`

EDAAgent writes pandas/numpy code as a string; executed via `exec(textwrap.dedent(code))`. Pre-injected: `df`, `pd`, `np`, `stats`. Agent sets `result = {...}`; returned as JSON.

### 4. Data Visualization ✅

**File:** `tools/chart_tools.py`

Six chart types (scatter, heatmap, histogram, radar, bar, trend line) rendered as 200 dpi base64 PNG via `matplotlib`. Model receives only `{"stored": True}` — base64 never enters model context. Frontend renders charts 2-per-row with color-coded type badges.

### 5. Artifacts ✅

**File:** `tools/artifact_tools.py` → `save_artifact(name, content, kind)`

`HypothesisAgent` writes full markdown reports to `artifacts/`. Auto-prunes to `_MAX_ARTIFACTS = 5`. Downloadable via `/artifacts` and `/artifact/{filename}` in `main.py`.

### 6. Structured Output ✅

**File:** `agents/orchestrator.py` → `_parse_json()`, all `run_*()` functions

Every agent outputs only valid JSON (enforced by system prompt). `_parse_json()` strips markdown fences, calls `json.loads()`, falls back to `\{.*\}` regex. Typed dicts passed between all pipeline stages via typed SSE events.

### 7. Second Independent Data Source ✅

**File:** `tools/wikipedia_tools.py`

Wikipedia REST API — separate from Deezer, no API key. Two-stage lookup: direct slug → OpenSearch fallback on 404. Triggered for artist/genre questions only.

-----

## Example Queries

|Question                                                         |Deezer endpoint                                                 |EDA branch triggered|
|-----------------------------------------------------------------|----------------------------------------------------------------|--------------------|
|`"What do Taylor Swift's most popular songs have in common?"`    |`get_artist_top_tracks("Taylor Swift", limit=50)`               |Artist              |
|`"What makes a hip-hop track chart in 2025?"`                    |`get_tag_top_tracks("hip-hop", limit=100)` → `/chart/116/tracks`|Genre               |
|`"What audio patterns define today's top Latin hits?"`           |`get_tag_top_tracks("latin", limit=100)` → `/chart/197/tracks`  |Genre               |
|`"Are shorter songs more popular on the global chart right now?"`|`get_top_tracks_chart(limit=100)`                               |Yes/No comparison   |
|`"What patterns show up across today's global top 25 hits?"`     |`get_top_tracks_chart(limit=100)`                               |Global trends       |

-----

## Deployment

```bash
gcloud run deploy musicanalyst \
  --source . \
  --region us-central1 \
  --project agentics-ai-488106 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 300 \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=1,GOOGLE_CLOUD_PROJECT=agentics-ai-488106,GOOGLE_CLOUD_LOCATION=us-central1"
```

Live URL: **https://musicanalyst-32zzvwbxsq-uc.a.run.app**

-----

## Project Structure

```
music-analyst-agent/
├── Dockerfile
├── main.py                     # FastAPI: SSE streaming, _backfill_art(), /preview proxy
├── pyproject.toml
├── .env                        # Vertex AI config (gitignored)
├── README.md
├── agents/
│   ├── __init__.py
│   └── orchestrator.py         # 4 LlmAgents, adk_* wrappers, run_*() runners,
│                               # _finding_is_specific(), iterative refinement loop,
│                               # generator-critic loop, _direct_deezer_fallback()
├── tools/
│   ├── __init__.py
│   ├── deezer_tools.py         # Deezer API: chart, genre charts, artist/track, _enrich_tracks()
│   ├── wikipedia_tools.py      # Wikipedia REST API (second data source)
│   ├── analysis_tools.py       # Stats, correlations, derived metrics, k-means, code execution
│   ├── chart_tools.py          # Matplotlib charts → base64 PNG (6 types)
│   └── artifact_tools.py       # save_artifact() → artifacts/
├── static/
│   └── index.html              # SSE client, audio player, charts, history, light/dark mode
└── artifacts/                  # Markdown reports (auto-created, max 5 kept)
```