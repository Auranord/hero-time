from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.models import CandidateClip


def build_segments(
    vod_id: str,
    scored_windows: list[dict[str, Any]],
    top_k: int = 20,
    min_duration_seconds: int = 20,
    max_duration_seconds: int = 40,
    merge_gap_seconds: float = 3.0,
    cooldown_seconds: float = 15.0,
) -> list[CandidateClip]:
    """Build stable clip candidates from scored windows.

    Pipeline:
    1) normalize/shape scored windows into candidate clips (20-40s duration band)
    2) top-k by score (with deterministic tie-breakers)
    3) merge neighboring/overlapping candidates
    4) apply cooldown to reduce near-duplicate picks
    """

    normalized = _normalize_windows(
        vod_id=vod_id,
        scored_windows=scored_windows,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
    )
    top_ranked = _select_top_k(normalized, top_k=top_k)
    merged = _merge_neighbors(top_ranked, merge_gap_seconds=merge_gap_seconds)
    cooled = _apply_cooldown(merged, cooldown_seconds=cooldown_seconds)

    # stable final ordering in timeline order with deterministic IDs
    timeline_sorted = sorted(cooled, key=lambda clip: (clip.start_seconds, -clip.score, clip.duration_seconds))
    return [replace(clip, clip_id=f"c_{idx:04d}") for idx, clip in enumerate(timeline_sorted, start=1)]


def _normalize_windows(
    vod_id: str,
    scored_windows: list[dict[str, Any]],
    min_duration_seconds: int,
    max_duration_seconds: int,
) -> list[CandidateClip]:
    clips: list[CandidateClip] = []

    target_duration = max(min_duration_seconds, min(max_duration_seconds, 30))
    for idx, window in enumerate(scored_windows, start=1):
        start = float(window.get("start_seconds", 0.0))
        window_end = window.get("end_seconds")
        default_end = start + target_duration
        end = float(window_end) if window_end is not None else default_end

        duration = max(1, int(round(end - start)))
        duration = max(min_duration_seconds, min(max_duration_seconds, duration))

        score = float(window.get("score", 0.0))
        reason_tags = sorted(set(window.get("reason_tags", [])))

        clips.append(
            CandidateClip(
                vod_id=vod_id,
                clip_id=f"w_{idx:04d}",
                start_seconds=round(start, 3),
                duration_seconds=duration,
                score=score,
                reason_tags=reason_tags,
            )
        )

    return clips


def _select_top_k(clips: list[CandidateClip], top_k: int) -> list[CandidateClip]:
    ranked = sorted(
        clips,
        key=lambda clip: (-clip.score, clip.start_seconds, clip.duration_seconds, clip.clip_id),
    )
    return ranked[: max(top_k, 0)]


def _merge_neighbors(clips: list[CandidateClip], merge_gap_seconds: float) -> list[CandidateClip]:
    if not clips:
        return []

    timeline = sorted(clips, key=lambda clip: (clip.start_seconds, -(clip.score)))
    merged: list[CandidateClip] = [timeline[0]]

    for clip in timeline[1:]:
        last = merged[-1]
        last_end = last.start_seconds + last.duration_seconds
        clip_end = clip.start_seconds + clip.duration_seconds

        should_merge = clip.start_seconds <= (last_end + merge_gap_seconds)
        if not should_merge:
            merged.append(clip)
            continue

        merged_start = min(last.start_seconds, clip.start_seconds)
        merged_end = max(last_end, clip_end)
        merged_duration = int(round(merged_end - merged_start))

        merged_score = max(last.score, clip.score)
        merged_tags = sorted(set(last.reason_tags + clip.reason_tags))

        merged[-1] = CandidateClip(
            vod_id=last.vod_id,
            clip_id=last.clip_id,
            start_seconds=round(merged_start, 3),
            duration_seconds=max(1, merged_duration),
            score=merged_score,
            reason_tags=merged_tags,
        )

    return merged


def _apply_cooldown(clips: list[CandidateClip], cooldown_seconds: float) -> list[CandidateClip]:
    if not clips:
        return []

    selected: list[CandidateClip] = []
    ranked = sorted(clips, key=lambda clip: (-clip.score, clip.start_seconds, clip.duration_seconds, clip.clip_id))

    for clip in ranked:
        if _violates_cooldown(clip, selected, cooldown_seconds=cooldown_seconds):
            continue
        selected.append(clip)

    return selected


def _violates_cooldown(clip: CandidateClip, selected: list[CandidateClip], cooldown_seconds: float) -> bool:
    clip_end = clip.start_seconds + clip.duration_seconds

    for kept in selected:
        kept_end = kept.start_seconds + kept.duration_seconds

        if clip.start_seconds < kept_end and clip_end > kept.start_seconds:
            return True

        min_allowed_start = kept.start_seconds - cooldown_seconds
        max_allowed_start = kept.start_seconds + cooldown_seconds
        if min_allowed_start <= clip.start_seconds <= max_allowed_start:
            return True

    return False
