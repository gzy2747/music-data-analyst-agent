"""
tools/artifact_tools.py
-----------------------
Artifact persistence for the Music Analyst pipeline.

Tools provided:
  - save_artifact : Write markdown/CSV/JSON reports to the artifacts/ directory.
                    Auto-prunes old files so only the 5 most recent are kept.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

_MAX_ARTIFACTS = 5


def save_artifact(
    name: str,
    content: str,
    kind: str = "markdown",
) -> dict[str, Any]:
    """
    Grab-bag: Artifacts — Write analysis outputs to disk as markdown, CSV, or JSON.
    Automatically prunes old artifacts so only the _MAX_ARTIFACTS most recent are kept.
    Returns the file path and byte size.
    """
    ext = {"markdown": ".md", "csv": ".csv", "json": ".json"}.get(kind, ".txt")
    filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{name}{ext}"
    path = ARTIFACTS_DIR / filename
    path.write_text(content, encoding="utf-8")

    # Auto-prune: delete oldest files beyond the cap
    all_files = sorted(ARTIFACTS_DIR.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
    for old in all_files[_MAX_ARTIFACTS:]:
        try:
            old.unlink()
        except OSError:
            pass

    return {
        "saved": True,
        "path": str(path),
        "filename": filename,
        "bytes": path.stat().st_size,
        "kind": kind,
    }
