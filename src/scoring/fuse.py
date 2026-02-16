from __future__ import annotations


def fuse_scores(heuristic: float, llm: float, alpha: float = 0.65) -> float:
    """Blend heuristic and LLM scores for hybrid ranking."""

    alpha = max(0.0, min(1.0, alpha))
    return max(0.0, min(1.0, (alpha * heuristic) + ((1 - alpha) * llm)))
