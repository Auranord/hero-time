from __future__ import annotations

from dataclasses import dataclass

from src.models import FeatureWindow


DEFAULT_WEIGHTS = {
    "loudness_spike": 0.2,
    "overlap_speech": 0.15,
    "speech_rate_burst": 0.1,
    "transcript_excitement": 0.2,
    "motion_peak": 0.2,
    "scene_change_rate": 0.15,
}


@dataclass(slots=True)
class HeuristicScoreDetails:
    """Explainable output for deterministic window scoring."""

    score: float
    reason_tags: list[str]
    weighted_contributions: dict[str, float]
    feature_values: dict[str, float]
    weights: dict[str, float]


def heuristic_score(window: FeatureWindow, weights: dict[str, float] | None = None) -> float:
    """Compute deterministic baseline score from weighted feature values."""

    return score_window(window=window, weights=weights).score


def score_window(
    window: FeatureWindow,
    weights: dict[str, float] | None = None,
    *,
    reason_threshold: float = 0.55,
    max_reason_tags: int = 3,
) -> HeuristicScoreDetails:
    """Score a feature window and emit deterministic explainability metadata."""

    resolved_weights = _resolve_weights(weights)
    if not resolved_weights:
        return HeuristicScoreDetails(
            score=0.0,
            reason_tags=[],
            weighted_contributions={},
            feature_values={},
            weights={},
        )

    weighted_contributions: dict[str, float] = {}
    feature_values: dict[str, float] = {}

    for key, weight in resolved_weights.items():
        value = _clamp(window.values.get(key, 0.0))
        feature_values[key] = value
        weighted_contributions[key] = value * weight

    score = _clamp(sum(weighted_contributions.values()))
    reason_tags = _generate_reason_tags(
        feature_values=feature_values,
        weighted_contributions=weighted_contributions,
        reason_threshold=reason_threshold,
        max_reason_tags=max_reason_tags,
    )

    return HeuristicScoreDetails(
        score=score,
        reason_tags=reason_tags,
        weighted_contributions=weighted_contributions,
        feature_values=feature_values,
        weights=resolved_weights,
    )


def _resolve_weights(weights: dict[str, float] | None) -> dict[str, float]:
    active_weights = weights or DEFAULT_WEIGHTS

    non_negative = {
        feature_name: max(0.0, raw_weight)
        for feature_name, raw_weight in active_weights.items()
    }
    total_weight = sum(non_negative.values())
    if total_weight == 0:
        return {}

    return {
        feature_name: weight / total_weight
        for feature_name, weight in non_negative.items()
    }


def _generate_reason_tags(
    *,
    feature_values: dict[str, float],
    weighted_contributions: dict[str, float],
    reason_threshold: float,
    max_reason_tags: int,
) -> list[str]:
    if max_reason_tags <= 0:
        return []

    tagged = [
        feature_name
        for feature_name, value in feature_values.items()
        if value >= _clamp(reason_threshold)
    ]
    if not tagged:
        tagged = [
            feature_name
            for feature_name, contribution in weighted_contributions.items()
            if contribution > 0
        ]

    tagged.sort(key=lambda key: (-weighted_contributions.get(key, 0.0), key))
    return [f"signal:{feature_name}" for feature_name in tagged[:max_reason_tags]]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
