from __future__ import annotations

import pytest

from src.models import FeatureWindow
from src.scoring.fuse import explain_fusion, fuse_scores, resolve_fused_score
from src.scoring.heuristic_score import heuristic_score, score_window


def test_heuristic_score_uses_normalized_weighted_sum() -> None:
    window = FeatureWindow(start_seconds=0, end_seconds=30, values={"a": 1.0, "b": 0.5})

    score = heuristic_score(window, weights={"a": 2.0, "b": 1.0})

    assert score == pytest.approx((1.0 * (2 / 3)) + (0.5 * (1 / 3)))


def test_score_window_returns_reason_tags_for_highest_contributions() -> None:
    window = FeatureWindow(
        start_seconds=0,
        end_seconds=30,
        values={
            "loudness_spike": 0.9,
            "overlap_speech": 0.8,
            "motion_peak": 0.4,
        },
    )

    details = score_window(window, weights={"loudness_spike": 0.6, "overlap_speech": 0.4, "motion_peak": 0.2})

    assert details.reason_tags == ["signal:loudness_spike", "signal:overlap_speech"]
    assert details.score == pytest.approx(0.7833333333333334)


def test_score_window_falls_back_to_top_signals_when_no_threshold_hits() -> None:
    window = FeatureWindow(start_seconds=0, end_seconds=30, values={"a": 0.3, "b": 0.2})

    details = score_window(window, weights={"a": 0.5, "b": 0.5}, reason_threshold=0.8)

    assert details.reason_tags == ["signal:a", "signal:b"]


def test_fuse_scores_clamps_values() -> None:
    assert fuse_scores(heuristic=2.0, llm=-1.0, alpha=0.25) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("strategy", "llm_score", "expected", "used_llm"),
    [
        ("heuristic", 0.9, 0.4, False),
        ("llm", 0.9, 0.9, True),
        ("llm", None, 0.4, False),
        ("hybrid", 0.9, (0.5 * 0.4) + (0.5 * 0.9), True),
        ("hybrid", None, 0.4, False),
    ],
)
def test_fusion_strategy_switch(strategy: str, llm_score: float | None, expected: float, used_llm: bool) -> None:
    details = explain_fusion(heuristic=0.4, llm=llm_score, strategy=strategy, alpha=0.5)

    assert details.score == pytest.approx(expected)
    assert details.used_llm is used_llm
    assert resolve_fused_score(heuristic=0.4, llm=llm_score, strategy=strategy, alpha=0.5) == pytest.approx(expected)


def test_invalid_fusion_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported scoring strategy"):
        explain_fusion(heuristic=0.4, llm=0.7, strategy="invalid")
