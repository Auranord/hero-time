from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FusionStrategy = Literal["heuristic", "llm", "hybrid"]


@dataclass(slots=True)
class FusionDetails:
    """Explainable output for score fusion decisions."""

    score: float
    strategy: FusionStrategy
    heuristic_score: float
    llm_score: float | None
    alpha: float
    used_llm: bool


def fuse_scores(heuristic: float, llm: float, alpha: float = 0.65) -> float:
    """Blend heuristic and LLM scores for hybrid ranking."""

    alpha = _clamp(alpha)
    return _clamp((alpha * _clamp(heuristic)) + ((1 - alpha) * _clamp(llm)))


def resolve_fused_score(
    heuristic: float,
    llm: float | None,
    *,
    strategy: FusionStrategy = "hybrid",
    alpha: float = 0.65,
) -> float:
    """Select final score using configured strategy (heuristic|llm|hybrid)."""

    return explain_fusion(heuristic=heuristic, llm=llm, strategy=strategy, alpha=alpha).score


def explain_fusion(
    heuristic: float,
    llm: float | None,
    *,
    strategy: FusionStrategy = "hybrid",
    alpha: float = 0.65,
) -> FusionDetails:
    """Select and explain score fusion using configured strategy."""

    normalized_strategy = _normalize_strategy(strategy)
    heuristic_score = _clamp(heuristic)
    llm_score = _clamp(llm) if llm is not None else None
    alpha = _clamp(alpha)

    if normalized_strategy == "heuristic":
        return FusionDetails(
            score=heuristic_score,
            strategy=normalized_strategy,
            heuristic_score=heuristic_score,
            llm_score=llm_score,
            alpha=alpha,
            used_llm=False,
        )

    if normalized_strategy == "llm":
        if llm_score is None:
            return FusionDetails(
                score=heuristic_score,
                strategy=normalized_strategy,
                heuristic_score=heuristic_score,
                llm_score=llm_score,
                alpha=alpha,
                used_llm=False,
            )

        return FusionDetails(
            score=llm_score,
            strategy=normalized_strategy,
            heuristic_score=heuristic_score,
            llm_score=llm_score,
            alpha=alpha,
            used_llm=True,
        )

    if llm_score is None:
        return FusionDetails(
            score=heuristic_score,
            strategy=normalized_strategy,
            heuristic_score=heuristic_score,
            llm_score=llm_score,
            alpha=alpha,
            used_llm=False,
        )

    return FusionDetails(
        score=fuse_scores(heuristic=heuristic_score, llm=llm_score, alpha=alpha),
        strategy=normalized_strategy,
        heuristic_score=heuristic_score,
        llm_score=llm_score,
        alpha=alpha,
        used_llm=True,
    )


def _normalize_strategy(strategy: str) -> FusionStrategy:
    normalized = strategy.lower().strip()
    if normalized not in {"heuristic", "llm", "hybrid"}:
        msg = (
            f"Unsupported scoring strategy '{strategy}'. "
            "Expected one of: heuristic, llm, hybrid."
        )
        raise ValueError(msg)
    return normalized  # type: ignore[return-value]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
