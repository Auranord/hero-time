from __future__ import annotations

import pytest

from src.scoring import llm_rerank


def test_rerank_with_local_llm_success_with_schema_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_rerank,
        "_request_ollama",
        lambda **_kwargs: '{"clip_worthiness":0.87,"primary_type":"hype","reason":"strong payoff","boundary_adjustment_seconds":3}',
    )

    result = llm_rerank.rerank_with_local_llm({"start_seconds": 10, "end_seconds": 30, "base_score": 0.5})

    assert result["clip_worthiness"] == pytest.approx(0.87)
    assert result["primary_type"] == "hype"
    assert result["boundary_adjustment_seconds"] == 3
    assert result["adjusted_start_seconds"] == pytest.approx(13.0)
    assert result["adjusted_end_seconds"] == pytest.approx(33.0)


def test_rerank_with_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            '{"clip_worthiness": 1.2, "primary_type": "hype", "reason": "x", "boundary_adjustment_seconds": 0}',
            '{"clip_worthiness": 0.6, "primary_type": "funny", "reason": "clear joke", "boundary_adjustment_seconds": -2}',
        ]
    )
    monkeypatch.setattr(llm_rerank, "_request_ollama", lambda **_kwargs: next(responses))

    result = llm_rerank.rerank_with_local_llm({"start_seconds": 5, "end_seconds": 15, "base_score": 0.4}, max_retries=2)

    assert result["clip_worthiness"] == pytest.approx(0.6)
    assert result["primary_type"] == "funny"
    assert result["boundary_adjustment_seconds"] == -2


def test_rerank_fallback_when_all_retries_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(["not json", "still not json"])
    monkeypatch.setattr(llm_rerank, "_request_ollama", lambda **_kwargs: next(responses))

    result = llm_rerank.rerank_with_local_llm({"base_score": 0.33}, max_retries=1)

    assert result["clip_worthiness"] == pytest.approx(0.33)
    assert result["primary_type"] == "unknown"
    assert result["boundary_adjustment_seconds"] == 0


def test_rerank_top_n_candidates_only_updates_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(['{"clip_worthiness":0.91,"primary_type":"story","reason":"good arc","boundary_adjustment_seconds":1}'])
    monkeypatch.setattr(llm_rerank, "_request_ollama", lambda **_kwargs: next(responses))

    candidates = [
        {"id": "a", "score": 0.2, "start_seconds": 0, "end_seconds": 10},
        {"id": "b", "score": 0.9, "start_seconds": 10, "end_seconds": 20},
    ]

    out = llm_rerank.rerank_top_n_candidates(candidates, top_n=1)

    assert "llm_score" not in out[0]
    assert out[1]["llm_score"] == pytest.approx(0.91)
    assert out[1]["llm_primary_type"] == "story"
