from __future__ import annotations

import wave

import numpy as np

from src.features.audio_events import detect_audio_events


def _write_test_wav(path, sample_rate: int = 16000) -> None:
    seconds = 2
    t = np.linspace(0, seconds, sample_rate * seconds, endpoint=False)
    signal = 0.05 * np.sin(2 * np.pi * 220 * t)
    signal[sample_rate : sample_rate + 2000] = 0.9  # loud burst for spike detection
    data = (signal * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def test_detect_audio_events_outputs_artifact_and_windows(tmp_path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_test_wav(wav_path)

    result = detect_audio_events(str(wav_path), cache_dir=str(tmp_path), window_seconds=1, window_overlap_seconds=0)

    assert result["status"] == "ok"
    assert result["event_count"] >= 1
    assert len(result["window_features"]) >= 2
    assert (tmp_path / "features" / "audio_events" / "sample" / "audio_events.json").exists()
