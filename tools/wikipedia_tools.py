"""
tools/wikipedia_tools.py
------------------------
Second data retrieval method: Wikipedia REST API (distinct from Deezer).

Uses the Wikipedia REST v1 summary endpoint — no API key or registration required.
Provides artist biography and genre context to enrich music analysis.
"""

from __future__ import annotations
import time
import requests

_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKI_SEARCH_URL  = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "MusicAnalystAgent/1.0 (academic project)"}


def get_artist_wikipedia_summary(artist_name: str) -> dict:
    """
    Fetch Wikipedia summary for a music artist.

    Returns biography snippet, genre labels, and Wikipedia URL.
    Uses the Wikipedia REST v1 /page/summary endpoint — no key needed.
    """
    try:
        # First try direct title lookup (most artists have a page matching their name)
        slug = artist_name.strip().replace(" ", "_")
        resp = requests.get(
            _WIKI_SUMMARY_URL.format(title=slug),
            headers=_HEADERS,
            timeout=8,
        )
        if resp.status_code == 404:
            # Fall back to Wikipedia search API to find the right page title
            search_resp = requests.get(
                _WIKI_SEARCH_URL,
                params={
                    "action": "opensearch",
                    "search": artist_name,
                    "limit": 3,
                    "namespace": 0,
                    "format": "json",
                },
                headers=_HEADERS,
                timeout=8,
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()
            # opensearch returns [query, [titles], [descriptions], [urls]]
            titles = search_data[1] if len(search_data) > 1 else []
            if not titles:
                return {"error": f"No Wikipedia page found for '{artist_name}'"}
            slug = titles[0].replace(" ", "_")
            resp = requests.get(
                _WIKI_SUMMARY_URL.format(title=slug),
                headers=_HEADERS,
                timeout=8,
            )

        resp.raise_for_status()
        data = resp.json()

        # Extract useful fields
        extract = data.get("extract", "")
        # Truncate to ~400 chars for token efficiency
        if len(extract) > 400:
            extract = extract[:400].rsplit(" ", 1)[0] + "…"

        return {
            "artist": artist_name,
            "wikipedia_title": data.get("title", ""),
            "wikipedia_url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "description": data.get("description", ""),   # short tagline e.g. "American rapper"
            "extract": extract,
            "thumbnail_url": (data.get("thumbnail") or {}).get("source", ""),
        }

    except requests.RequestException as exc:
        return {"error": f"Wikipedia request failed: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}


def get_genre_wikipedia_overview(genre: str) -> dict:
    """
    Fetch a short Wikipedia overview for a music genre (e.g. 'hip-hop', 'K-pop').

    Useful for contextualising trends in genre-focused queries.
    """
    try:
        slug = (genre.strip() + " music").replace(" ", "_")
        resp = requests.get(
            _WIKI_SUMMARY_URL.format(title=slug),
            headers=_HEADERS,
            timeout=8,
        )
        if resp.status_code == 404:
            # Try without " music" suffix
            slug = genre.strip().replace(" ", "_")
            resp = requests.get(
                _WIKI_SUMMARY_URL.format(title=slug),
                headers=_HEADERS,
                timeout=8,
            )
        resp.raise_for_status()
        data = resp.json()

        extract = data.get("extract", "")
        if len(extract) > 400:
            extract = extract[:400].rsplit(" ", 1)[0] + "…"

        return {
            "genre": genre,
            "wikipedia_title": data.get("title", ""),
            "wikipedia_url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "description": data.get("description", ""),
            "extract": extract,
        }

    except requests.RequestException as exc:
        return {"error": f"Wikipedia request failed: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}
