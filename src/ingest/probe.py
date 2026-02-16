from __future__ import annotations

from pathlib import Path


def probe_media(vod_path: str) -> dict:
    """Return minimal media metadata placeholder for MVP scaffold."""

    return {
        "vod_path": str(Path(vod_path).resolve()),
        "status": "not_implemented",
        "next": "integrate ffprobe JSON parsing",
    }
