from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any


def run_asr(
    audio_path: str,
    cache_dir: str = "data/cache",
    language: str = "de",
    model_size: str = "small",
    chunk_seconds: int = 120,
    device: str = "auto",
    compute_type: str = "default",
) -> dict[str, Any]:
    """Transcribe an audio track with faster-whisper and persist ASR artifacts."""

    source_path = Path(audio_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    duration_seconds = _probe_duration_seconds(source_path)
    asr_dir = Path(cache_dir).expanduser().resolve() / "features" / "asr" / source_path.stem
    chunks_dir = asr_dir / "chunks"
    asr_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    normalized_segments: list[dict[str, Any]] = []

    chunk_starts = _build_chunk_starts(duration_seconds, chunk_seconds)
    for chunk_index, chunk_start in enumerate(chunk_starts):
        chunk_duration = min(chunk_seconds, max(duration_seconds - chunk_start, 0.0))
        if chunk_duration <= 0:
            continue

        chunk_path = chunks_dir / f"chunk_{chunk_index:04d}.wav"
        _extract_chunk(
            source_path=source_path,
            chunk_path=chunk_path,
            chunk_start=chunk_start,
            chunk_duration=chunk_duration,
        )

        segments_iter, info = model.transcribe(
            str(chunk_path),
            language=language,
            vad_filter=True,
            word_timestamps=False,
        )

        for segment in segments_iter:
            normalized_segments.append(
                {
                    "id": f"seg_{len(normalized_segments):06d}",
                    "chunk_index": chunk_index,
                    "start_seconds": round(float(segment.start + chunk_start), 3),
                    "end_seconds": round(float(segment.end + chunk_start), 3),
                    "duration_seconds": round(float(segment.end - segment.start), 3),
                    "text": segment.text.strip(),
                    "avg_logprob": _safe_float(getattr(segment, "avg_logprob", None)),
                    "no_speech_prob": _safe_float(getattr(segment, "no_speech_prob", None)),
                    "compression_ratio": _safe_float(getattr(segment, "compression_ratio", None)),
                    "confidence": _segment_confidence(getattr(segment, "avg_logprob", None)),
                    "language": language,
                }
            )

    transcript_payload = {
        "status": "ok",
        "audio_path": str(source_path),
        "cache_dir": str(asr_dir),
        "language": language,
        "model_size": model_size,
        "device": device,
        "compute_type": compute_type,
        "duration_seconds": duration_seconds,
        "chunk_seconds": chunk_seconds,
        "chunk_count": len(chunk_starts),
        "segment_count": len(normalized_segments),
        "segments": normalized_segments,
    }

    transcript_path = asr_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(transcript_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        **transcript_payload,
        "transcript_path": str(transcript_path),
    }


def _probe_duration_seconds(audio_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(completed.stdout.strip())


def _build_chunk_starts(duration_seconds: float, chunk_seconds: int) -> list[float]:
    if duration_seconds <= 0:
        return [0.0]

    starts: list[float] = []
    current = 0.0
    while current < duration_seconds:
        starts.append(round(current, 3))
        current += float(chunk_seconds)
    return starts


def _extract_chunk(
    source_path: Path,
    chunk_path: Path,
    chunk_start: float,
    chunk_duration: float,
) -> None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{chunk_start:.3f}",
        "-t",
        f"{chunk_duration:.3f}",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(chunk_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _segment_confidence(avg_logprob: Any) -> float | None:
    if avg_logprob is None:
        return None

    value = float(avg_logprob)
    # map log-probability to [0, 1] in a simple, monotonic way
    return round(min(max(math.exp(value), 0.0), 1.0), 4)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
