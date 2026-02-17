from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.models import CandidateClip, FeatureWindow
from src.propose.segment_builder import build_segments
from src.scoring.fuse import explain_fusion
from src.scoring.heuristic_score import score_window
from src.scoring.llm_rerank import rerank_top_n_candidates


def build_candidates_from_artifacts(
    *,
    vod_id: str,
    asr_payload: dict[str, Any],
    diarization_payload: dict[str, Any],
    video_motion_payload: dict[str, Any],
    audio_events_payload: dict[str, Any],
    weights: dict[str, float],
    strategy: str,
    hybrid_alpha: float,
    top_k: int,
    rerank_top_n: int,
    llm_endpoint: str,
    llm_model: str,
    llm_timeout_seconds: int,
) -> list[CandidateClip]:
    windows = _build_feature_windows(
        asr_payload=asr_payload,
        diarization_payload=diarization_payload,
        video_motion_payload=video_motion_payload,
        audio_events_payload=audio_events_payload,
    )

    scored_rows: list[dict[str, Any]] = []
    for window in windows:
        heuristic_details = score_window(window, weights=weights)
        scored_rows.append(
            {
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "base_score": heuristic_details.score,
                "score": heuristic_details.score,
                "reason_tags": heuristic_details.reason_tags,
                "feature_values": heuristic_details.feature_values,
                "transcript_excerpt": str(window.metadata.get("transcript_excerpt", "")),
            }
        )

    if rerank_top_n > 0:
        scored_rows = rerank_top_n_candidates(
            scored_rows,
            top_n=rerank_top_n,
            endpoint=llm_endpoint,
            model=llm_model,
            timeout_seconds=llm_timeout_seconds,
        )

    final_rows: list[dict[str, Any]] = []
    for row in scored_rows:
        llm_score = row.get("llm_score")
        fusion = explain_fusion(
            heuristic=float(row["base_score"]),
            llm=float(llm_score) if llm_score is not None else None,
            strategy=strategy,
            alpha=hybrid_alpha,
        )
        reason_tags = list(row.get("reason_tags", []))
        llm_type = row.get("llm_primary_type")
        if llm_type:
            reason_tags.append(f"type:{llm_type}")

        final_rows.append(
            {
                "start_seconds": row.get("adjusted_start_seconds", row["start_seconds"]),
                "end_seconds": row.get("adjusted_end_seconds", row["end_seconds"]),
                "score": fusion.score,
                "reason_tags": sorted(set(reason_tags)),
                "llm_summary": row.get("llm_reason"),
            }
        )

    clips = build_segments(vod_id=vod_id, scored_windows=final_rows, top_k=top_k)
    summaries = {round(row["start_seconds"], 3): row.get("llm_summary") for row in final_rows if row.get("llm_summary")}

    hydrated: list[CandidateClip] = []
    for clip in clips:
        summary = summaries.get(round(clip.start_seconds, 3))
        hydrated.append(
            CandidateClip(
                vod_id=clip.vod_id,
                clip_id=clip.clip_id,
                start_seconds=clip.start_seconds,
                duration_seconds=clip.duration_seconds,
                score=clip.score,
                reason_tags=clip.reason_tags,
                llm_summary=summary,
            )
        )

    return hydrated


def _build_feature_windows(
    *,
    asr_payload: dict[str, Any],
    diarization_payload: dict[str, Any],
    video_motion_payload: dict[str, Any],
    audio_events_payload: dict[str, Any],
) -> list[FeatureWindow]:
    diarization_windows = diarization_payload.get("window_overlap_stats", [])
    video_windows = _index_windows(video_motion_payload.get("window_features", []))
    audio_windows = _index_windows(audio_events_payload.get("window_features", []))
    transcript_segments = asr_payload.get("segments", [])

    speech_rate_per_window: dict[int, float] = defaultdict(float)
    excitement_per_window: dict[int, float] = defaultdict(float)
    transcript_lines_per_window: dict[int, list[str]] = defaultdict(list)

    for segment in transcript_segments:
        start = float(segment.get("start_seconds", 0.0))
        end = max(float(segment.get("end_seconds", start)), start)
        duration = max(end - start, 1e-6)
        text = str(segment.get("text", ""))
        words = [word for word in text.split() if word.strip()]
        rate = len(words) / duration
        window_index = int(start // 15)
        speech_rate_per_window[window_index] += rate
        if text.strip():
            transcript_lines_per_window[window_index].append(text.strip())

        excitement = _transcript_excitement_score(text)
        excitement_per_window[window_index] = max(excitement_per_window[window_index], excitement)

    max_rate = max(speech_rate_per_window.values(), default=1.0)

    windows: list[FeatureWindow] = []
    for row in diarization_windows:
        index = int(row.get("window_index", len(windows)))
        start = float(row.get("start_seconds", 0.0))
        end = float(row.get("end_seconds", start + 30.0))

        video = video_windows.get(index, {})
        audio = audio_windows.get(index, {})

        features = {
            "loudness_spike": float(audio.get("loudness_spike_score", 0.0)),
            "overlap_speech": float(row.get("overlap_ratio", 0.0)),
            "speech_rate_burst": min(speech_rate_per_window.get(index, 0.0) / max(max_rate, 1e-6), 1.0),
            "transcript_excitement": excitement_per_window.get(index, 0.0),
            "motion_peak": float(video.get("motion_peak", 0.0)),
            "scene_change_rate": float(video.get("scene_change_frequency", 0.0)),
        }

        windows.append(
            FeatureWindow(
                start_seconds=start,
                end_seconds=end,
                values=features,
                metadata={
                    "window_index": index,
                    "transcript_excerpt": " ".join(transcript_lines_per_window.get(index, [])),
                },
            )
        )

    return windows


def _index_windows(windows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(windows):
        key = int(row.get("window_index", idx))
        indexed[key] = row
    return indexed


def _transcript_excitement_score(text: str) -> float:
    lowered = text.lower()
    markers = ["!", "lol", "lmao", "krass", "geil", "omg", "nein", "jaaa", "hype"]
    hits = sum(1 for marker in markers if marker in lowered)
    return min(hits / 3.0, 1.0)
