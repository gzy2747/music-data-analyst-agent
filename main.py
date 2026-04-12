"""
main.py
-------
FastAPI application serving the Music Analyst multi-agent system.
Uses Gemini 2.5 Flash via Vertex AI (Application Default Credentials).

Endpoints:
  GET  /          – serve index.html
  POST /analyze   – run the 3-step pipeline, stream progress via SSE
  GET  /artifacts – list saved artifact files
  GET  /artifact/{filename} – download a specific artifact
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import traceback
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Music Analyst – Multi-Agent Music Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Static files & index
# ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/artifacts")
async def list_artifacts():
    files = sorted(ARTIFACTS_DIR.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [
        {
            "filename": f.name,
            "size": f.stat().st_size,
            "url": f"/artifact/{f.name}",
        }
        for f in files[:20]
    ]


@app.get("/artifact/{filename}")
async def get_artifact(filename: str):
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        return {"error": "Not found"}
    return FileResponse(path)


@app.get("/preview/{deezer_id}")
async def preview_proxy(deezer_id: int):
    """
    Fetch a fresh (non-expired) Deezer preview URL for a track and redirect to it.
    Deezer CDN preview URLs are signed tokens that expire in ~10-15 minutes,
    so we re-fetch on every play request to guarantee a valid URL.
    """
    import requests as _req
    from fastapi.responses import RedirectResponse, JSONResponse
    try:
        r = _req.get(f"https://api.deezer.com/track/{deezer_id}", timeout=8)
        data = r.json()
        preview = data.get("preview")
        if not preview:
            return JSONResponse({"error": "No preview available"}, status_code=404)
        return RedirectResponse(url=preview, status_code=302)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


# ──────────────────────────────────────────────────────────────
# SSE helpers
# ──────────────────────────────────────────────────────────────

def _sse(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ──────────────────────────────────────────────────────────────
# Streaming pipeline
# ──────────────────────────────────────────────────────────────

async def stream_pipeline(question: str) -> AsyncIterator[str]:
    yield _sse("status", {"message": "Starting music analysis pipeline…", "step": "init"})

    try:
        from agents.orchestrator import run_collector, run_eda, run_hypothesis
        loop = asyncio.get_event_loop()

        # ── Step 1: Collect ──────────────────────────────────
        yield _sse("status", {
            "message": "Step 1: Collecting data from Deezer…",
            "step": "collect",
        })

        collect_q: asyncio.Queue = asyncio.Queue()

        def _cb1(tool_name, detail=""):
            try:
                loop.call_soon_threadsafe(collect_q.put_nowait, ("p", tool_name, detail))
            except Exception:
                pass

        def _collect_worker():
            try:
                r = run_collector(question, progress_cb=_cb1)
                loop.call_soon_threadsafe(collect_q.put_nowait, ("done", r, None))
            except Exception as e:
                loop.call_soon_threadsafe(collect_q.put_nowait, ("err", None, e))

        threading.Thread(target=_collect_worker, daemon=True).start()
        collect_logs = []
        collected = None
        while True:
            kind, payload, exc = await collect_q.get()
            if kind == "p":
                yield _sse("tool_progress", {"tool": payload, "detail": exc or "", "step": "collect"})
            elif kind == "done":
                collected, collect_logs = payload
                break
            else:
                raise exc

        tracks = collected.get("tracks", [])
        preview_count = sum(1 for t in tracks if t.get("preview_url"))

        # Build full track lookup by (name, artist) so we can backfill any missing fields
        # into model-generated representative_tracks / highlight_tracks lists.
        _track_map = {
            (t.get("name", "").lower(), t.get("artist", "").lower()): t
            for t in tracks
        }
        # Secondary lookup by name only — handles cases where model outputs
        # slightly different artist name (extra spaces, abbreviations, etc.)
        _track_map_by_name: dict = {}
        for t in tracks:
            name_key = t.get("name", "").lower().strip()
            if name_key and name_key not in _track_map_by_name:
                _track_map_by_name[name_key] = t
        # Tertiary lookup by deezer_id
        _track_map_by_id = {
            t.get("deezer_id"): t
            for t in tracks
            if t.get("deezer_id")
        }

        # Build normalized (no feat/parens) name lookup as 4th-tier fallback
        def _norm(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r'\s*(feat\.?|ft\.?|featuring|with)\s+.*$', '', s, flags=re.IGNORECASE)
            s = re.sub(r'\s*\(.*?\)', '', s)
            s = re.sub(r'[^\w\s]', '', s)
            return s.strip()

        _track_map_normalized = {_norm(t.get("name", "")): t for t in reversed(tracks) if t.get("name")}

        # 5th-tier: strip ALL non-alphanumeric chars (handles curly quotes, dashes, etc.)
        def _bare(s: str) -> str:
            import unicodedata
            s = unicodedata.normalize("NFKD", s.lower())
            return re.sub(r'[^a-z0-9]', '', s)

        _track_map_bare: dict = {}
        for t in reversed(tracks):
            bk = _bare(t.get("name", ""))
            if bk:
                _track_map_bare[bk] = t

        def _backfill_art(track_list: list) -> list:
            """
            Overwrite URLs and backfill metadata from authoritative collected tracks.
            album_art / preview_url / deezer_url are ALWAYS replaced when a match is found
            — agents frequently output wrong/hallucinated URLs that must be corrected.
            """
            for t in track_list:
                key = (t.get("name", "").lower().strip(), t.get("artist", "").lower().strip())
                src = (
                    _track_map.get(key)
                    or _track_map_by_id.get(t.get("deezer_id"))
                    or _track_map_by_name.get(t.get("name", "").lower().strip())
                    or _track_map_normalized.get(_norm(t.get("name", "")))
                    or _track_map_bare.get(_bare(t.get("name", "")))
                    or {}
                )
                if not src:
                    continue
                # Always overwrite URL fields — agent output is unreliable
                t["album_art"]   = src.get("album_art", t.get("album_art", ""))
                t["preview_url"] = src.get("preview_url") or t.get("preview_url")
                t["deezer_url"]  = src.get("deezer_url", t.get("deezer_url", ""))
                t["deezer_id"]   = src.get("deezer_id",  t.get("deezer_id"))
                # Backfill numeric/string fields only if missing
                if not t.get("gain"):        t["gain"]         = src.get("gain")
                if not t.get("duration"):    t["duration"]     = src.get("duration", 0)
                if not t.get("popularity"):  t["popularity"]   = src.get("popularity", 0)
                if not t.get("rank"):        t["rank"]         = src.get("rank", 0)
                if not t.get("release_date"):t["release_date"] = src.get("release_date", "")
                if t.get("explicit") is None:t["explicit"]     = src.get("explicit", False)
            return track_list

        yield _sse("collect_done", {
            "n_tracks": collected.get("n_tracks", len(tracks)),
            "query_interpretation": collected.get("query_interpretation", ""),
            "preview_available": preview_count,
            "deezer_source": collected.get("deezer_context", {}).get("source", ""),
            "tool_calls": len(collect_logs),
        })

        # ── Step 2: EDA ──────────────────────────────────────
        yield _sse("status", {
            "message": "Step 2: Running EDA — statistics, correlations, clustering, charts…",
            "step": "eda",
        })

        eda_q: asyncio.Queue = asyncio.Queue()

        def _cb2(tool_name, detail=""):
            try:
                loop.call_soon_threadsafe(eda_q.put_nowait, ("p", tool_name, detail))
            except Exception:
                pass

        def _eda_worker():
            try:
                r = run_eda(collected, question, progress_cb=_cb2)
                loop.call_soon_threadsafe(eda_q.put_nowait, ("done", r, None))
            except Exception as e:
                loop.call_soon_threadsafe(eda_q.put_nowait, ("err", None, e))

        threading.Thread(target=_eda_worker, daemon=True).start()
        eda = None
        eda_logs = []
        while True:
            kind, payload, exc = await eda_q.get()
            if kind == "p":
                yield _sse("tool_progress", {"tool": payload, "detail": exc or "", "step": "eda"})
            elif kind == "done":
                eda, eda_logs = payload
                break
            else:
                raise exc

        yield _sse("eda_done", {
            "findings": eda.get("findings", []),
            "charts": eda.get("charts", []),
            "representative_tracks": _backfill_art(eda.get("representative_tracks", [])),
            "tool_calls": len(eda_logs),
        })

        # ── Step 3: Hypothesis ───────────────────────────────
        yield _sse("status", {
            "message": "Step 3: Forming evidence-backed music hypothesis…",
            "step": "hypothesis",
        })

        hyp_q: asyncio.Queue = asyncio.Queue()

        def _cb3(tool_name, detail=""):
            try:
                loop.call_soon_threadsafe(hyp_q.put_nowait, ("p", tool_name, detail))
            except Exception:
                pass

        def _hyp_worker():
            try:
                r = run_hypothesis(eda, question, collected, progress_cb=_cb3)
                loop.call_soon_threadsafe(hyp_q.put_nowait, ("done", r, None))
            except Exception as e:
                loop.call_soon_threadsafe(hyp_q.put_nowait, ("err", None, e))

        threading.Thread(target=_hyp_worker, daemon=True).start()
        hypothesis = None
        hyp_logs = []
        while True:
            kind, payload, exc = await hyp_q.get()
            if kind == "p":
                yield _sse("tool_progress", {"tool": payload, "detail": exc or "", "step": "hypothesis"})
            elif kind == "done":
                hypothesis, hyp_logs = payload
                break
            else:
                raise exc

        # Send tool_log BEFORE hypothesis_done so renderResults has the full log
        all_logs = collect_logs + eda_logs + hyp_logs
        yield _sse("tool_log", {"calls": all_logs})

        yield _sse("hypothesis_done", {
            "direct_answer": hypothesis.get("direct_answer", ""),
            "hypothesis": hypothesis.get("hypothesis", ""),
            "confidence": hypothesis.get("confidence", ""),
            "confidence_reasoning": hypothesis.get("confidence_reasoning", ""),
            "supporting_evidence": hypothesis.get("supporting_evidence", []),
            "alternative_explanations": hypothesis.get("alternative_explanations", []),
            "summary_paragraph": hypothesis.get("summary_paragraph", ""),
            "highlight_tracks": _backfill_art(hypothesis.get("highlight_tracks", [])),
            "artifact_path": hypothesis.get("artifact_path", ""),
            "tool_calls": len(hyp_logs),
        })
        yield _sse("done", {"message": "Analysis complete"})

    except Exception as exc:
        tb = traceback.format_exc()
        yield _sse("error", {"message": str(exc), "traceback": tb})


# ──────────────────────────────────────────────────────────────
# Analyze endpoint
# ──────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return {"error": "No question provided"}

    return StreamingResponse(
        stream_pipeline(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import os
    host = "0.0.0.0" if os.getenv("K_SERVICE") else "127.0.0.1"  # 0.0.0.0 on Cloud Run
    uvicorn.run(app, host=host, port=8080)
