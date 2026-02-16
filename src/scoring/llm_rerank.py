from __future__ import annotations


def rerank_with_local_llm(candidate_payload: dict) -> dict:
    """Return schema-shaped placeholder for local Ollama reranking."""

    return {
        "clip_worthiness": candidate_payload.get("base_score", 0.0),
        "primary_type": "unknown",
        "reason": "LLM reranker is scaffolded but not yet connected.",
        "boundary_adjustment_seconds": 0,
    }
