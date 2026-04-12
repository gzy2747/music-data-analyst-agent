"""
tools/spotify_tools.py
----------------------
Spotify API integration using Client Credentials Flow (no user auth needed).

Tools provided:
  - search_tracks           : Search Spotify by query (artist, genre, mood, era, etc.)
  - get_audio_features      : Batch-fetch audio features for up to 100 tracks
  - get_playlist_tracks     : Fetch tracks from a public Spotify playlist
  - search_by_genre_and_era : Search by genre + year range using Spotify filters
"""

from __future__ import annotations

import os
import time
from typing import Any

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

_SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
_SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")


def _get_client() -> spotipy.Spotify:
    """Return an authenticated Spotipy client (Client Credentials Flow)."""
    credentials = SpotifyClientCredentials(
        client_id=_SPOTIFY_CLIENT_ID,
        client_secret=_SPOTIFY_CLIENT_SECRET,
    )
    return spotipy.Spotify(auth_manager=credentials)


def _parse_track(track: dict) -> dict:
    """Extract the standardised track fields from a raw Spotify track object."""
    images = track.get("album", {}).get("images", [])
    album_art = images[0]["url"] if images else ""
    return {
        "id": track.get("id", ""),
        "name": track.get("name", ""),
        "artist": ", ".join(a["name"] for a in track.get("artists", [])),
        "album": track.get("album", {}).get("name", ""),
        "release_date": track.get("album", {}).get("release_date", ""),
        "popularity": track.get("popularity", 0),
        "preview_url": track.get("preview_url"),
        "spotify_url": track.get("external_urls", {}).get("spotify", ""),
        "album_art": album_art,
        "duration_ms": track.get("duration_ms", 0),
    }


# ──────────────────────────────────────────────────────────────
# PUBLIC TOOL FUNCTIONS
# ──────────────────────────────────────────────────────────────

def search_tracks(query: str, limit: int = 50) -> dict[str, Any]:
    """
    Search Spotify for tracks matching a query (artist, genre, mood, era, etc.)
    Uses Client Credentials Flow — no user auth needed.

    Returns:
    {
        "tracks": [
            {
                "id": str,
                "name": str,
                "artist": str,
                "album": str,
                "release_date": str,       # "2023-04-07"
                "popularity": int,          # 0-100
                "preview_url": str | None,  # 30s mp3 URL or null
                "spotify_url": str,         # https://open.spotify.com/track/...
                "album_art": str,           # image URL (640x640)
                "duration_ms": int,
            }
        ],
        "total": int,
        "query": str,
    }
    On error: {"error": "...", "query": query}
    """
    # Spotify search endpoint allows max 50 per request
    limit = min(limit, 50)
    try:
        sp = _get_client()
        response = sp.search(q=query, type="track", limit=limit)
        items = response.get("tracks", {}).get("items", [])
        tracks = [_parse_track(t) for t in items if t]
        return {
            "tracks": tracks,
            "total": len(tracks),
            "query": query,
        }
    except spotipy.exceptions.SpotifyException as exc:
        if exc.http_status == 429:
            retry_after = int(exc.headers.get("Retry-After", 1)) if exc.headers else 1
            time.sleep(retry_after + 0.5)
            try:
                sp = _get_client()
                response = sp.search(q=query, type="track", limit=limit)
                items = response.get("tracks", {}).get("items", [])
                tracks = [_parse_track(t) for t in items if t]
                return {"tracks": tracks, "total": len(tracks), "query": query}
            except Exception as retry_exc:
                return {"error": str(retry_exc), "query": query}
        return {"error": str(exc), "query": query}
    except Exception as exc:
        return {"error": str(exc), "query": query}


def get_audio_features(track_ids: list[str]) -> dict[str, Any]:
    """
    Batch fetch audio features for up to 100 tracks.

    Returns:
    {
        "features": [
            {
                "id": str,
                "danceability": float,    # 0.0-1.0
                "energy": float,          # 0.0-1.0
                "valence": float,         # 0.0-1.0 (sad→happy)
                "tempo": float,           # BPM
                "acousticness": float,    # 0.0-1.0
                "instrumentalness": float,# 0.0-1.0
                "speechiness": float,     # 0.0-1.0
                "liveness": float,        # 0.0-1.0
                "loudness": float,        # dB, typically -60 to 0
                "key": int,               # 0-11
                "mode": int,              # 0=minor, 1=major
                "time_signature": int,
            }
        ],
        "count": int,
    }
    On error: {"error": "..."}
    """
    if not track_ids:
        return {"features": [], "count": 0}

    # Spotify allows max 100 per batch call
    batch = track_ids[:100]
    try:
        sp = _get_client()
        raw = sp.audio_features(batch)
        features = []
        for item in raw:
            if item is None:
                continue
            features.append({
                "id": item.get("id", ""),
                "danceability": item.get("danceability", 0.0),
                "energy": item.get("energy", 0.0),
                "valence": item.get("valence", 0.0),
                "tempo": item.get("tempo", 0.0),
                "acousticness": item.get("acousticness", 0.0),
                "instrumentalness": item.get("instrumentalness", 0.0),
                "speechiness": item.get("speechiness", 0.0),
                "liveness": item.get("liveness", 0.0),
                "loudness": item.get("loudness", 0.0),
                "key": item.get("key", 0),
                "mode": item.get("mode", 0),
                "time_signature": item.get("time_signature", 4),
            })
        return {"features": features, "count": len(features)}
    except spotipy.exceptions.SpotifyException as exc:
        if exc.http_status == 429:
            retry_after = int(exc.headers.get("Retry-After", 1)) if exc.headers else 1
            time.sleep(retry_after + 0.5)
            try:
                sp = _get_client()
                raw = sp.audio_features(batch)
                features = []
                for item in raw:
                    if item is None:
                        continue
                    features.append({
                        "id": item.get("id", ""),
                        "danceability": item.get("danceability", 0.0),
                        "energy": item.get("energy", 0.0),
                        "valence": item.get("valence", 0.0),
                        "tempo": item.get("tempo", 0.0),
                        "acousticness": item.get("acousticness", 0.0),
                        "instrumentalness": item.get("instrumentalness", 0.0),
                        "speechiness": item.get("speechiness", 0.0),
                        "liveness": item.get("liveness", 0.0),
                        "loudness": item.get("loudness", 0.0),
                        "key": item.get("key", 0),
                        "mode": item.get("mode", 0),
                        "time_signature": item.get("time_signature", 4),
                    })
                return {"features": features, "count": len(features)}
            except Exception as retry_exc:
                return {"error": str(retry_exc)}
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


def get_playlist_tracks(playlist_id: str, limit: int = 100) -> dict[str, Any]:
    """
    Fetch tracks from a public Spotify playlist.
    Good for well-known playlists like "Today's Top Hits" (ID: 37i9dQZF1DXcBWIGoYBM5M)

    Returns same schema as search_tracks, plus:
    {
        "playlist_name": str,
        "playlist_description": str,
        "tracks": [...],
        "total": int,
    }
    """
    try:
        sp = _get_client()
        playlist = sp.playlist(playlist_id, fields="name,description,tracks.total")
        playlist_name = playlist.get("name", "")
        playlist_description = playlist.get("description", "")

        tracks = []
        offset = 0
        batch_size = min(100, limit)

        while len(tracks) < limit:
            fetch_limit = min(batch_size, limit - len(tracks))
            results = sp.playlist_tracks(
                playlist_id,
                limit=fetch_limit,
                offset=offset,
                fields="items(track(id,name,artists,album,popularity,preview_url,external_urls,duration_ms))",
            )
            items = results.get("items", [])
            if not items:
                break
            for item in items:
                track = item.get("track")
                if track and track.get("id"):
                    tracks.append(_parse_track(track))
            offset += len(items)
            if len(items) < fetch_limit:
                break

        return {
            "playlist_name": playlist_name,
            "playlist_description": playlist_description,
            "tracks": tracks,
            "total": len(tracks),
        }
    except spotipy.exceptions.SpotifyException as exc:
        if exc.http_status == 429:
            retry_after = int(exc.headers.get("Retry-After", 1)) if exc.headers else 1
            time.sleep(retry_after + 0.5)
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


def search_by_genre_and_era(
    genre: str,
    year_from: int,
    year_to: int,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Search tracks by genre and year range using Spotify's search filters.
    Uses query: "genre:{genre} year:{year_from}-{year_to}"

    Returns same schema as search_tracks.
    """
    query = f"genre:{genre} year:{year_from}-{year_to}"
    return search_tracks(query=query, limit=limit)
