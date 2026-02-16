from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli import _fallback_diarization_payload, _select_default_audio_track


def test_select_default_audio_track_uses_first_track() -> None:
    result = {
        "tracks": [
            {"path": "/tmp/track_1.wav"},
            {"path": "/tmp/track_2.wav"},
        ]
    }
    assert _select_default_audio_track(result) == "/tmp/track_1.wav"


def test_select_default_audio_track_raises_when_empty() -> None:
    with pytest.raises(ValueError, match="No audio tracks"):
        _select_default_audio_track({"tracks": []})


def test_fallback_diarization_payload_writes_artifact(tmp_path: Path) -> None:
    payload = _fallback_diarization_payload(
        asr_result={"duration_seconds": 35.0},
        window_seconds=30,
        window_overlap_seconds=15,
        cache_dir=tmp_path,
        audio_track_path="/tmp/audio.wav",
    )

    assert payload["status"] == "fallback"
    assert payload["speaker_count"] == 1
    assert len(payload["window_overlap_stats"]) >= 2

    artifact = Path(payload["diarization_path"])
    assert artifact.exists()

    parsed = json.loads(artifact.read_text(encoding="utf-8"))
    assert parsed["pipeline_model"] == "fallback-single-speaker"
