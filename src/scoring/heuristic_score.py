from __future__ import annotations

from src.models import FeatureWindow


DEFAULT_WEIGHTS = {
    "loudness_spike": 0.2,
    "overlap_speech": 0.15,
    "speech_rate_burst": 0.1,
    "transcript_excitement": 0.2,
    "motion_peak": 0.2,
    "scene_change_rate": 0.15,
}


def heuristic_score(window: FeatureWindow, weights: dict[str, float] | None = None) -> float:
    """Compute deterministic baseline score from weighted feature values."""

    active_weights = weights or DEFAULT_WEIGHTS
    weighted_sum = 0.0
    total_weight = 0.0

    for key, weight in active_weights.items():
        weighted_sum += window.values.get(key, 0.0) * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return max(0.0, min(1.0, weighted_sum / total_weight))
