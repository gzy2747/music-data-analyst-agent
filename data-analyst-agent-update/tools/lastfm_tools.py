"""
tools/lastfm_tools.py
---------------------
Last.fm API integration using pylast.

Tools provided:
  - get_top_tracks_chart  : Global weekly top tracks chart
  - get_track_info        : Detailed info for a specific track
  - get_tag_top_tracks    : Top tracks for a genre/mood/era tag
  - get_artist_top_tracks : An artist's most played tracks
"""

from __future__ import annotations

import os
import time
from typing import Any

import pylast
from dotenv import load_dotenv

load_dotenv()

_LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")
_LASTFM_SHARED_SECRET = os.getenv("LASTFM_SHARED_SECRET", "")


def _get_network() -> pylast.LastFMNetwork:
    """Return an authenticated LastFMNetwork client."""
    return pylast.LastFMNetwork(
        api_key=_LASTFM_API_KEY,
        api_secret=_LASTFM_SHARED_SECRET,
    )


# ──────────────────────────────────────────────────────────────
# PUBLIC TOOL FUNCTIONS
# ──────────────────────────────────────────────────────────────

def get_top_tracks_chart(limit: int = 50, page: int = 1) -> dict[str, Any]:
    """
    Get Last.fm global weekly top tracks chart.

    Returns:
    {
        "tracks": [
            {
                "name": str,
                "artist": str,
                "playcount": int,
                "listeners": int,
                "rank": int,
            }
        ],
        "total": int,
        "page": int,
    }
    """
    try:
        network = _get_network()
        chart_tracks = network.get_top_tracks(limit=limit)
        tracks = []
        for rank, item in enumerate(chart_tracks, start=1 + (page - 1) * limit):
            track = item.item
            try:
                playcount = int(item.weight) if item.weight else 0
            except (TypeError, ValueError):
                playcount = 0
            try:
                listeners = int(track.get_listener_count())
            except Exception:
                listeners = 0
            time.sleep(0.1)
            tracks.append({
                "name": track.title,
                "artist": track.artist.name,
                "playcount": playcount,
                "listeners": listeners,
                "rank": rank,
            })
        return {
            "tracks": tracks,
            "total": len(tracks),
            "page": page,
        }
    except pylast.WSError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


def get_track_info(track_name: str, artist_name: str) -> dict[str, Any]:
    """
    Get detailed info for a specific track from Last.fm.

    Returns:
    {
        "name": str,
        "artist": str,
        "playcount": int,
        "listeners": int,
        "tags": [str, ...],      # genre/mood tags e.g. ["pop", "indie", "sad"]
        "wiki_summary": str,     # short description if available
        "duration": int,         # seconds
    }
    On error: {"error": "...", "track": track_name, "artist": artist_name}
    """
    try:
        network = _get_network()
        track = network.get_track(artist_name, track_name)

        try:
            playcount = int(track.get_playcount())
        except Exception:
            playcount = 0

        try:
            listeners = int(track.get_listener_count())
        except Exception:
            listeners = 0

        try:
            top_tags = track.get_top_tags(limit=10)
            tags = [t.item.name for t in top_tags]
        except Exception:
            tags = []

        try:
            wiki = track.get_wiki_summary()
            wiki_summary = wiki if wiki else ""
        except Exception:
            wiki_summary = ""

        try:
            duration_ms = track.get_duration()
            duration = int(duration_ms) // 1000 if duration_ms else 0
        except Exception:
            duration = 0

        return {
            "name": track_name,
            "artist": artist_name,
            "playcount": playcount,
            "listeners": listeners,
            "tags": tags,
            "wiki_summary": wiki_summary,
            "duration": duration,
        }
    except pylast.WSError as exc:
        return {"error": str(exc), "track": track_name, "artist": artist_name}
    except Exception as exc:
        return {"error": str(exc), "track": track_name, "artist": artist_name}


def get_tag_top_tracks(tag: str, limit: int = 50) -> dict[str, Any]:
    """
    Get top tracks for a Last.fm tag (genre, mood, era).
    Examples: tag="indie pop", tag="2010s", tag="sad", tag="electronic"

    Returns:
    {
        "tag": str,
        "tracks": [
            {
                "name": str,
                "artist": str,
                "rank": int,
                "url": str,
            }
        ],
    }
    """
    try:
        network = _get_network()
        lastfm_tag = network.get_tag(tag)
        top_tracks = lastfm_tag.get_top_tracks(limit=limit)
        tracks = []
        for rank, item in enumerate(top_tracks, start=1):
            track = item.item
            try:
                url = track.get_url()
            except Exception:
                url = ""
            time.sleep(0.1)
            tracks.append({
                "name": track.title,
                "artist": track.artist.name,
                "rank": rank,
                "url": url,
            })
        return {
            "tag": tag,
            "tracks": tracks,
        }
    except pylast.WSError as exc:
        return {"error": str(exc), "tag": tag}
    except Exception as exc:
        return {"error": str(exc), "tag": tag}


def get_artist_top_tracks(artist_name: str, limit: int = 20) -> dict[str, Any]:
    """
    Get an artist's most played tracks on Last.fm.

    Returns:
    {
        "artist": str,
        "tracks": [
            {
                "name": str,
                "playcount": int,
                "listeners": int,
                "rank": int,
            }
        ],
    }
    """
    try:
        network = _get_network()
        artist = network.get_artist(artist_name)
        top_tracks = artist.get_top_tracks(limit=limit)
        tracks = []
        for rank, item in enumerate(top_tracks, start=1):
            track = item.item
            try:
                playcount = int(item.weight) if item.weight else 0
            except (TypeError, ValueError):
                playcount = 0
            try:
                listeners = int(track.get_listener_count())
            except Exception:
                listeners = 0
            time.sleep(0.1)
            tracks.append({
                "name": track.title,
                "playcount": playcount,
                "listeners": listeners,
                "rank": rank,
            })
        return {
            "artist": artist_name,
            "tracks": tracks,
        }
    except pylast.WSError as exc:
        return {"error": str(exc), "artist": artist_name}
    except Exception as exc:
        return {"error": str(exc), "artist": artist_name}
