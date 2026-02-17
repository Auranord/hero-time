from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_diarization(
    audio_path: str,
    cache_dir: str = "data/cache",
    transcript_path: str | None = None,
    window_seconds: int = 30,
    window_overlap_seconds: int = 15,
    hf_auth_token: str | None = None,
    pipeline_model: str = "pyannote/speaker-diarization-3.1",
    device: str = "auto",
) -> dict[str, Any]:
    """Run pyannote diarization, align turns to transcript timeline, and cache artifacts."""

    source_path = Path(audio_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    from pyannote.audio import Pipeline

    resolved_device = _resolve_torch_device(device)

    pipeline = _load_pyannote_pipeline(
        pipeline_class=Pipeline,
        pipeline_model=pipeline_model,
        hf_auth_token=hf_auth_token,
    )
    pipeline.to(resolved_device)
    diarization = pipeline(str(source_path))

    speaker_turns = [
        {
            "id": f"turn_{index:06d}",
            "speaker": str(speaker),
            "start_seconds": round(float(segment.start), 3),
            "end_seconds": round(float(segment.end), 3),
            "duration_seconds": round(float(segment.end - segment.start), 3),
        }
        for index, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True))
    ]

    transcript_segments = _load_transcript_segments(transcript_path)
    aligned_segments = _align_transcript_segments(transcript_segments, speaker_turns)

    duration_seconds = _estimate_duration_seconds(speaker_turns, aligned_segments)
    windows = _compute_window_overlap_stats(
        speaker_turns=speaker_turns,
        duration_seconds=duration_seconds,
        window_seconds=window_seconds,
        window_overlap_seconds=window_overlap_seconds,
    )

    global_overlap = _compute_overlap_duration(
        speaker_turns=speaker_turns,
        range_start=0.0,
        range_end=duration_seconds,
    )
    unique_speakers = sorted({turn["speaker"] for turn in speaker_turns})
    speaking_time_seconds = {
        speaker: round(
            sum(turn["duration_seconds"] for turn in speaker_turns if turn["speaker"] == speaker),
            3,
        )
        for speaker in unique_speakers
    }

    diarization_dir = Path(cache_dir).expanduser().resolve() / "features" / "diarization" / source_path.stem
    diarization_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "ok",
        "audio_path": str(source_path),
        "cache_dir": str(diarization_dir),
        "pipeline_model": pipeline_model,
        "device": str(resolved_device),
        "speaker_count": len(unique_speakers),
        "speakers": unique_speakers,
        "duration_seconds": duration_seconds,
        "speaker_turn_count": len(speaker_turns),
        "speaker_turns": speaker_turns,
        "aligned_transcript_segment_count": len(aligned_segments),
        "aligned_transcript_segments": aligned_segments,
        "overlap": {
            "global_overlap_seconds": round(global_overlap, 3),
            "global_overlap_ratio": round(global_overlap / duration_seconds, 4) if duration_seconds > 0 else 0.0,
        },
        "window_seconds": window_seconds,
        "window_overlap_seconds": window_overlap_seconds,
        "window_overlap_stats": windows,
        "speaking_time_seconds": speaking_time_seconds,
    }

    artifact_path = diarization_dir / "diarization.json"
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        **payload,
        "diarization_path": str(artifact_path),
    }


def _load_pyannote_pipeline(*, pipeline_class: Any, pipeline_model: str, hf_auth_token: str | None) -> Any:
    if hf_auth_token:
        try:
            return pipeline_class.from_pretrained(pipeline_model, use_auth_token=hf_auth_token)
        except TypeError as exc:
            if "use_auth_token" not in str(exc):
                raise
            return pipeline_class.from_pretrained(pipeline_model, token=hf_auth_token)

    return pipeline_class.from_pretrained(pipeline_model)


def _resolve_torch_device(device: str) -> Any:
    import torch

    normalized = device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(normalized)


def _load_transcript_segments(transcript_path: str | None) -> list[dict[str, Any]]:
    if not transcript_path:
        return []

    path = Path(transcript_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("segments", [])


def _align_transcript_segments(
    transcript_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    for segment in transcript_segments:
        start = float(segment.get("start_seconds", 0.0))
        end = float(segment.get("end_seconds", start))
        overlaps = []
        for turn in speaker_turns:
            overlap_seconds = _intersection_duration(start, end, turn["start_seconds"], turn["end_seconds"])
            if overlap_seconds <= 0:
                continue
            overlaps.append((turn["speaker"], overlap_seconds))

        if overlaps:
            best_speaker = max(overlaps, key=lambda pair: pair[1])[0]
        else:
            best_speaker = "unknown"

        aligned.append(
            {
                **segment,
                "speaker": best_speaker,
            }
        )

    return aligned


def _estimate_duration_seconds(
    speaker_turns: list[dict[str, Any]],
    aligned_segments: list[dict[str, Any]],
) -> float:
    turn_max = max((turn["end_seconds"] for turn in speaker_turns), default=0.0)
    transcript_max = max((float(seg.get("end_seconds", 0.0)) for seg in aligned_segments), default=0.0)
    return round(max(turn_max, transcript_max), 3)


def _compute_window_overlap_stats(
    speaker_turns: list[dict[str, Any]],
    duration_seconds: float,
    window_seconds: int,
    window_overlap_seconds: int,
) -> list[dict[str, Any]]:
    if duration_seconds <= 0:
        return []

    step_seconds = max(window_seconds - window_overlap_seconds, 1)
    windows: list[dict[str, Any]] = []
    start = 0.0
    index = 0
    while start < duration_seconds:
        end = min(start + window_seconds, duration_seconds)
        overlap_seconds = _compute_overlap_duration(speaker_turns, start, end)
        windows.append(
            {
                "window_index": index,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "overlap_seconds": round(overlap_seconds, 3),
                "overlap_ratio": round(overlap_seconds / (end - start), 4) if end > start else 0.0,
                "active_speakers": _active_speaker_count(speaker_turns, start, end),
            }
        )
        start += float(step_seconds)
        index += 1

    return windows


def _compute_overlap_duration(
    speaker_turns: list[dict[str, Any]],
    range_start: float,
    range_end: float,
) -> float:
    boundaries = {range_start, range_end}
    for turn in speaker_turns:
        start = max(turn["start_seconds"], range_start)
        end = min(turn["end_seconds"], range_end)
        if end > start:
            boundaries.add(start)
            boundaries.add(end)

    sorted_points = sorted(boundaries)
    overlap = 0.0
    for idx in range(len(sorted_points) - 1):
        slice_start = sorted_points[idx]
        slice_end = sorted_points[idx + 1]
        if slice_end <= slice_start:
            continue

        active = 0
        for turn in speaker_turns:
            if _intersection_duration(slice_start, slice_end, turn["start_seconds"], turn["end_seconds"]) > 0:
                active += 1

        if active >= 2:
            overlap += slice_end - slice_start

    return overlap


def _active_speaker_count(speaker_turns: list[dict[str, Any]], start: float, end: float) -> int:
    speakers = {
        turn["speaker"]
        for turn in speaker_turns
        if _intersection_duration(start, end, turn["start_seconds"], turn["end_seconds"]) > 0
    }
    return len(speakers)


def _intersection_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))
