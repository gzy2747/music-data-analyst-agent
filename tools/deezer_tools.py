"""
tools/deezer_tools.py
---------------------
Deezer API integration — no API key required, public endpoints only.

Tools provided:
  - get_top_tracks_chart  : Deezer global top-tracks chart
  - get_track_info        : Detailed info for a specific track
  - get_tag_top_tracks    : Top tracks for a genre/mood tag (via search)
  - get_artist_top_tracks : An artist's top tracks on Deezer
  - get_tracks_details    : Batch-fetch full track details
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

_BASE = "https://api.deezer.com"
_TIMEOUT = 10


def _get(path: str, params: dict | None = None) -> dict:
    """GET a Deezer endpoint and return parsed JSON. Raises on HTTP error."""
    r = requests.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data["error"])))
    return data


def _parse_track(item: dict) -> dict:
    """Normalise a Deezer track object to a common schema."""
    artist = item.get("artist", {})
    album  = item.get("album", {})
    album_art = ""
    if isinstance(album, dict):
        album_art = album.get("cover_xl") or album.get("cover_big") or album.get("cover_medium") or ""
    return {
        "name":        item.get("title", ""),
        "artist":      artist.get("name", "") if isinstance(artist, dict) else str(artist),
        "album":       album.get("title", "") if isinstance(album, dict) else str(album),
        "album_art":   album_art,
        "rank":        item.get("rank", item.get("position", 0)),
        "duration":    item.get("duration", 0),           # seconds
        "gain":        float(item.get("gain", 0.0)),      # loudness proxy
        "preview_url": item.get("preview"),               # 30-second mp3 (may be None)
        "deezer_url":  item.get("link", ""),
        "deezer_id":   item.get("id", 0),
    }


def _enrich_one(t: dict) -> None:
    """Fetch /track/{id} for a single track and merge fields in-place."""
    tid = t.get("deezer_id")
    if not tid:
        return
    try:
        detail = _get(f"/track/{tid}")
        album = detail.get("album", {})
        if isinstance(album, dict):
            art = album.get("cover_xl") or album.get("cover_big") or album.get("cover_medium") or ""
            if art:
                t["album_art"] = art
        gain = detail.get("gain")
        if gain is not None:
            t["gain"] = float(gain)
        rd = detail.get("release_date")
        if rd:
            t["release_date"] = rd
        t["explicit"] = bool(detail.get("explicit_lyrics", False))
    except Exception:
        pass


def _enrich_tracks(tracks: list[dict]) -> list[dict]:
    """Fetch /track/{id} for each track in parallel and merge fields in-place."""
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(_enrich_one, t) for t in tracks]
        for f in as_completed(futures):
            f.result()
    return tracks


# ──────────────────────────────────────────────────────────────
# PUBLIC TOOL FUNCTIONS
# ──────────────────────────────────────────────────────────────

def get_top_tracks_chart(limit: int = 100, page: int = 1) -> dict[str, Any]:
    """
    Get Deezer global top-tracks chart.
    Default limit raised to 100 to ensure a non-trivial dataset size.

    Returns:
    {
        "tracks": [ { "name", "artist", "rank", "duration", "preview_url",
                      "deezer_url", "deezer_id", "gain", "release_date", "explicit" } ],
        "total": int,
        "page": int,
    }
    """
    try:
        offset = (page - 1) * limit
        data = _get("/chart/0/tracks", params={"limit": min(limit, 100), "index": offset})
        items = data.get("data", [])
        tracks = []
        for rank_offset, item in enumerate(items, start=offset + 1):
            t = _parse_track(item)
            t["rank"] = rank_offset
            tracks.append(t)
        _enrich_tracks(tracks)
        return {"tracks": tracks, "total": len(tracks), "page": page}
    except Exception as exc:
        return {"error": str(exc)}


def get_track_info(track_name: str, artist_name: str) -> dict[str, Any]:
    """
    Get detailed info for a specific track from Deezer (via search).

    Returns:
    {
        "name": str, "artist": str, "album": str,
        "duration": int, "rank": int, "preview_url": str | None,
        "deezer_url": str, "deezer_id": int, "bpm": float,
    }
    On error: {"error": "...", "track": track_name, "artist": artist_name}
    """
    try:
        query = f'track:"{track_name}" artist:"{artist_name}"'
        data = _get("/search", params={"q": query, "limit": 1})
        items = data.get("data", [])
        if not items:
            data = _get("/search", params={"q": f"{track_name} {artist_name}", "limit": 1})
            items = data.get("data", [])
        if not items:
            return {"error": "Track not found", "track": track_name, "artist": artist_name}

        item = items[0]
        result = _parse_track(item)

        try:
            detail = _get(f"/track/{item['id']}")
            result["bpm"]  = detail.get("bpm", 0.0)
            result["gain"] = detail.get("gain", 0.0)
        except Exception:
            result["bpm"] = 0.0

        return result
    except Exception as exc:
        return {"error": str(exc), "track": track_name, "artist": artist_name}


# Deezer genre IDs — maps common keywords to /chart/{id}/tracks
_GENRE_IDS: dict[str, int] = {
    "pop":          132,
    "rap":          116,
    "hip-hop":      116,
    "hiphop":       116,
    "hip hop":      116,
    "reggaeton":    122,
    "rock":         152,
    "dance":        113,
    "edm":          113,
    "r&b":          165,
    "rnb":          165,
    "r and b":      165,
    "alternative":  85,
    "alt":          85,
    "christian":    186,
    "electro":      106,
    "electronic":   106,
    "folk":         466,
    "reggae":       144,
    "jazz":         129,
    "country":      84,
    "classical":    98,
    "metal":        464,
    "soul":         169,
    "funk":         169,
    "latin":        197,
    "k-pop":        16,
    "kpop":         16,
    "k pop":        16,
    "blues":        153,
    "indie":        85,
}


def _genre_id(tag: str) -> int | None:
    """Return Deezer genre ID for a tag, or None if not a known genre."""
    key = tag.lower().strip()
    if key in _GENRE_IDS:
        return _GENRE_IDS[key]
    for k, v in _GENRE_IDS.items():
        if k in key:
            return v
    return None


def get_tag_top_tracks(tag: str, limit: int = 100) -> dict[str, Any]:
    """
    Get top tracks for a genre/mood/era tag.
    Default limit raised to 100.
    For known genres uses Deezer genre chart; falls back to keyword search.
    """
    try:
        gid = _genre_id(tag)
        if gid is not None:
            data = _get(f"/chart/{gid}/tracks", params={"limit": min(limit, 100)})
            items = data.get("data", [])
            tracks = []
            for rank, item in enumerate(items, start=1):
                t = _parse_track(item)
                t["rank"] = rank
                tracks.append(t)
            _enrich_tracks(tracks)
            return {"tag": tag, "genre_id": gid, "tracks": tracks}
        else:
            data = _get("/search", params={"q": tag, "limit": min(limit, 100), "order": "RANKING"})
            items = data.get("data", [])
            tracks = []
            for rank, item in enumerate(items, start=1):
                t = _parse_track(item)
                t["rank"] = rank
                tracks.append(t)
            _enrich_tracks(tracks)
            return {"tag": tag, "tracks": tracks}
    except Exception as exc:
        return {"error": str(exc), "tag": tag}


def get_tracks_details(deezer_ids: list[int]) -> dict[str, Any]:
    """
    Batch-fetch full track details for a list of Deezer track IDs.

    Returns:
    { "tracks": [ { "deezer_id", "gain", "album_art", "duration",
                    "release_date", "explicit" } ], "count": int }
    """
    def _fetch_one(track_id: int) -> dict:
        try:
            detail = _get(f"/track/{track_id}")
            album = detail.get("album", {})
            album_art = (
                album.get("cover_xl") or album.get("cover_big") or
                album.get("cover_medium") or ""
            ) if isinstance(album, dict) else ""
            return {
                "deezer_id":    track_id,
                "gain":         float(detail.get("gain", 0.0)),
                "album_art":    album_art,
                "duration":     int(detail.get("duration", 0)),
                "release_date": detail.get("release_date", ""),
                "explicit":     bool(detail.get("explicit_lyrics", False)),
            }
        except Exception:
            return {
                "deezer_id": track_id, "gain": 0.0, "album_art": "",
                "duration": 0, "release_date": "", "explicit": False,
            }

    ids = deezer_ids[:100]
    results = [None] * len(ids)
    with ThreadPoolExecutor(max_workers=20) as pool:
        future_to_idx = {pool.submit(_fetch_one, tid): i for i, tid in enumerate(ids)}
        for f in as_completed(future_to_idx):
            results[future_to_idx[f]] = f.result()
    return {"tracks": results, "count": len(results)}


def get_artist_top_tracks(artist_name: str, limit: int = 50) -> dict[str, Any]:
    """
    Get an artist's top tracks on Deezer.
    Default limit raised to 50 for a richer dataset.

    Returns:
    { "artist": str, "tracks": [ { "name", "rank", "duration",
                                   "preview_url", "deezer_url", "deezer_id" } ] }
    """
    try:
        search = _get("/search/artist", params={"q": artist_name, "limit": 1})
        artists = search.get("data", [])
        if not artists:
            return {"error": f"Artist not found: {artist_name}", "artist": artist_name}

        artist_id = artists[0]["id"]
        data = _get(f"/artist/{artist_id}/top", params={"limit": min(limit, 50)})
        items = data.get("data", [])
        tracks = []
        for rank, item in enumerate(items, start=1):
            t = _parse_track(item)
            t["rank"] = rank
            tracks.append(t)

        _enrich_tracks(tracks)
        return {"artist": artist_name, "tracks": tracks}
    except Exception as exc:
        return {"error": str(exc), "artist": artist_name}