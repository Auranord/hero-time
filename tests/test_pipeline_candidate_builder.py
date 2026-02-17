from __future__ import annotations

from src.pipeline_candidate_builder import build_candidates_from_artifacts


def test_build_candidates_from_artifacts_end_to_end_without_llm() -> None:
    asr_payload = {
        "segments": [
            {"start_seconds": 1, "end_seconds": 3, "text": "krass das war geil!"},
            {"start_seconds": 16, "end_seconds": 18, "text": "ruhig"},
        ]
    }
    diarization_payload = {
        "window_overlap_stats": [
            {"window_index": 0, "start_seconds": 0, "end_seconds": 15, "overlap_ratio": 0.5},
            {"window_index": 1, "start_seconds": 15, "end_seconds": 30, "overlap_ratio": 0.1},
        ]
    }
    video_motion_payload = {
        "window_features": [
            {"window_index": 0, "motion_peak": 0.8, "scene_change_frequency": 0.2},
            {"window_index": 1, "motion_peak": 0.2, "scene_change_frequency": 0.05},
        ]
    }
    audio_events_payload = {
        "window_features": [
            {"window_index": 0, "loudness_spike_score": 0.9},
            {"window_index": 1, "loudness_spike_score": 0.2},
        ]
    }

    clips = build_candidates_from_artifacts(
        vod_id="vod1",
        asr_payload=asr_payload,
        diarization_payload=diarization_payload,
        video_motion_payload=video_motion_payload,
        audio_events_payload=audio_events_payload,
        weights={
            "loudness_spike": 0.2,
            "overlap_speech": 0.2,
            "speech_rate_burst": 0.2,
            "transcript_excitement": 0.1,
            "motion_peak": 0.2,
            "scene_change_rate": 0.1,
        },
        strategy="hybrid",
        hybrid_alpha=0.65,
        top_k=5,
        rerank_top_n=0,
        llm_endpoint="http://localhost:11434",
        llm_model="dummy",
        llm_timeout_seconds=1,
    )

    assert clips
    assert clips[0].vod_id == "vod1"
    assert clips[0].score >= 0


def test_build_candidates_includes_transcript_excerpt_in_llm_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_rerank_top_n_candidates(candidates, **kwargs):
        captured["candidates"] = candidates
        return candidates

    monkeypatch.setattr(
        "src.pipeline_candidate_builder.rerank_top_n_candidates",
        fake_rerank_top_n_candidates,
    )

    asr_payload = {
        "segments": [
            {"start_seconds": 1, "end_seconds": 3, "text": "krass das war geil!"},
        ]
    }
    diarization_payload = {
        "window_overlap_stats": [
            {"window_index": 0, "start_seconds": 0, "end_seconds": 15, "overlap_ratio": 0.5},
        ]
    }
    video_motion_payload = {"window_features": [{"window_index": 0, "motion_peak": 0.8, "scene_change_frequency": 0.2}]}
    audio_events_payload = {"window_features": [{"window_index": 0, "loudness_spike_score": 0.9}]}

    build_candidates_from_artifacts(
        vod_id="vod1",
        asr_payload=asr_payload,
        diarization_payload=diarization_payload,
        video_motion_payload=video_motion_payload,
        audio_events_payload=audio_events_payload,
        weights={
            "loudness_spike": 0.2,
            "overlap_speech": 0.2,
            "speech_rate_burst": 0.2,
            "transcript_excitement": 0.1,
            "motion_peak": 0.2,
            "scene_change_rate": 0.1,
        },
        strategy="hybrid",
        hybrid_alpha=0.65,
        top_k=1,
        rerank_top_n=1,
        llm_endpoint="http://localhost:11434",
        llm_model="dummy",
        llm_timeout_seconds=1,
    )

    candidates = captured["candidates"]
    assert isinstance(candidates, list)
    assert candidates[0]["transcript_excerpt"] == "krass das war geil!"
