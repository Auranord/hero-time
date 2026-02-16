from __future__ import annotations


def detect_audio_events(audio_path: str) -> dict:
    """Audio event proxy contract (loudness/reaction cues)."""

    return {
        "audio_path": audio_path,
        "events": [],
        "status": "not_implemented",
    }
