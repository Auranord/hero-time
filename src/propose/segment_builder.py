from __future__ import annotations

from src.models import CandidateClip


def build_segments(vod_id: str, scored_windows: list[dict]) -> list[CandidateClip]:
    """Convert scored windows into candidate clip proposals."""

    clips: list[CandidateClip] = []
    for idx, window in enumerate(scored_windows, start=1):
        start = float(window.get("start_seconds", 0.0))
        end = float(window.get("end_seconds", start + 90.0))
        score = float(window.get("score", 0.0))
        clip = CandidateClip(
            vod_id=vod_id,
            clip_id=f"c_{idx:04d}",
            start_seconds=start,
            duration_seconds=max(1, int(end - start)),
            score=score,
            reason_tags=list(window.get("reason_tags", [])),
        )
        clips.append(clip)
    return clips
