from __future__ import annotations


def extract_audio_tracks(vod_path: str, cache_dir: str) -> dict:
    """Describe intended audio extraction outputs for MVP scaffold."""

    return {
        "vod_path": vod_path,
        "cache_dir": cache_dir,
        "tracks": ["mic", "discord", "mix"],
        "status": "not_implemented",
    }
