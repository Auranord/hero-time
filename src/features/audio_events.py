from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import numpy as np


def detect_audio_events(
    audio_path: str,
    cache_dir: str = "data/cache",
    window_seconds: int = 30,
    window_overlap_seconds: int = 15,
    frame_seconds: float = 0.5,
) -> dict[str, Any]:
    """Detect lightweight loudness/reaction proxies and persist feature artifacts."""

    source_path = Path(audio_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    samples, sample_rate = _read_wav_mono(source_path)
    frame_size = max(int(round(sample_rate * max(frame_seconds, 0.1))), 1)

    frames = _build_frames(samples=samples, sample_rate=sample_rate, frame_size=frame_size)
    rms_values = np.array([frame["rms"] for frame in frames], dtype=np.float64)

    baseline = float(np.median(rms_values)) if len(rms_values) else 0.0
    scale = float(np.std(rms_values)) if len(rms_values) else 0.0
    threshold = baseline + max(scale * 2.0, 0.02)

    events = [
        {
            "event_id": f"evt_{idx:06d}",
            "timestamp_seconds": frame["start_seconds"],
            "type": "loudness_spike",
            "intensity": round(frame["rms"], 6),
            "z_score": round((frame["rms"] - baseline) / max(scale, 1e-6), 3) if scale > 0 else 0.0,
        }
        for idx, frame in enumerate(frames)
        if frame["rms"] >= threshold
    ]

    duration_seconds = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    window_features = _window_audio_features(
        frames=frames,
        duration_seconds=duration_seconds,
        window_seconds=window_seconds,
        window_overlap_seconds=window_overlap_seconds,
        spike_threshold=threshold,
    )

    artifact_dir = Path(cache_dir).expanduser().resolve() / "features" / "audio_events" / source_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "ok",
        "audio_path": str(source_path),
        "cache_dir": str(artifact_dir),
        "sample_rate": sample_rate,
        "duration_seconds": round(duration_seconds, 3),
        "frame_seconds": frame_seconds,
        "frame_count": len(frames),
        "loudness_baseline": round(baseline, 6),
        "loudness_std": round(scale, 6),
        "loudness_spike_threshold": round(threshold, 6),
        "events": events,
        "event_count": len(events),
        "window_seconds": window_seconds,
        "window_overlap_seconds": window_overlap_seconds,
        "window_features": window_features,
    }

    artifact_path = artifact_dir / "audio_events.json"
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        **payload,
        "audio_events_path": str(artifact_path),
    }


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = int(wav_file.getframerate())
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV input is supported for audio event extraction.")

    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    normalized = samples.astype(np.float32) / 32768.0
    return normalized, sample_rate


def _build_frames(samples: np.ndarray, sample_rate: int, frame_size: int) -> list[dict[str, float]]:
    if sample_rate <= 0 or len(samples) == 0:
        return []

    frames: list[dict[str, float]] = []
    for start in range(0, len(samples), frame_size):
        end = min(start + frame_size, len(samples))
        segment = samples[start:end]
        if len(segment) == 0:
            continue

        rms = float(np.sqrt(np.mean(np.square(segment))))
        peak = float(np.max(np.abs(segment)))
        frames.append(
            {
                "start_seconds": round(start / sample_rate, 3),
                "end_seconds": round(end / sample_rate, 3),
                "rms": round(rms, 6),
                "peak": round(peak, 6),
            }
        )
    return frames


def _window_audio_features(
    *,
    frames: list[dict[str, float]],
    duration_seconds: float,
    window_seconds: int,
    window_overlap_seconds: int,
    spike_threshold: float,
) -> list[dict[str, Any]]:
    if duration_seconds <= 0:
        return []

    step_seconds = max(window_seconds - window_overlap_seconds, 1)
    windows: list[dict[str, Any]] = []
    index = 0
    start = 0.0

    while start < duration_seconds:
        end = min(start + window_seconds, duration_seconds)
        in_window = [frame for frame in frames if start <= frame["start_seconds"] < end]
        rms_values = [frame["rms"] for frame in in_window]
        peak_values = [frame["peak"] for frame in in_window]
        spike_count = sum(1 for frame in in_window if frame["rms"] >= spike_threshold)

        windows.append(
            {
                "window_index": index,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "sample_count": len(in_window),
                "rms_mean": round(sum(rms_values) / len(rms_values), 6) if rms_values else 0.0,
                "rms_peak": round(max(rms_values), 6) if rms_values else 0.0,
                "peak_amplitude": round(max(peak_values), 6) if peak_values else 0.0,
                "spike_count": spike_count,
                "spike_rate": round(spike_count / (end - start), 6) if end > start else 0.0,
                "loudness_spike_score": round(max(rms_values), 6) if rms_values else 0.0,
            }
        )

        start += float(step_seconds)
        index += 1

    return windows
