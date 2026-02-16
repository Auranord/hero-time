from __future__ import annotations


def run_diarization(audio_path: str) -> dict:
    """Diarization interface contract for pyannote integration."""

    return {
        "audio_path": audio_path,
        "speaker_turns": [],
        "overlap_ratio": 0.0,
        "status": "not_implemented",
    }
