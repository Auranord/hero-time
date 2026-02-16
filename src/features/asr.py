from __future__ import annotations


def run_asr(audio_path: str, language: str = "de") -> dict:
    """ASR interface contract for faster-whisper integration."""

    return {
        "audio_path": audio_path,
        "language": language,
        "segments": [],
        "status": "not_implemented",
    }
