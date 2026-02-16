from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_RETRIES = 2
DEFAULT_BOUNDARY_ADJUSTMENT_LIMIT_SECONDS = 12
ALLOWED_PRIMARY_TYPES = {"funny", "hype", "gameplay", "story", "cozy", "unknown"}
PROMPT_TEMPLATE_PATH = Path("prompts/rerank_prompt.txt")


def rerank_with_local_llm(
    candidate_payload: dict[str, Any],
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    boundary_adjustment_limit_seconds: int = DEFAULT_BOUNDARY_ADJUSTMENT_LIMIT_SECONDS,
) -> dict[str, Any]:
    """Rerank one candidate with a local Ollama model using strict JSON validation."""

    fallback = _fallback_result(candidate_payload, reason="LLM reranker unavailable; used deterministic fallback.")

    for _ in range(max(0, max_retries) + 1):
        try:
            prompt = _format_prompt(candidate_payload)
            response_text = _request_ollama(
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
            )
            parsed = json.loads(response_text)
            rerank = _validate_rerank_schema(parsed)
            rerank = _apply_boundary_adjustment(
                candidate_payload=candidate_payload,
                rerank_payload=rerank,
                limit_seconds=boundary_adjustment_limit_seconds,
            )
            return rerank
        except (json.JSONDecodeError, ValueError, HTTPError, URLError, TimeoutError, OSError, KeyError, TypeError):
            continue

    return fallback


def rerank_top_n_candidates(
    candidates: list[dict[str, Any]],
    *,
    top_n: int,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    boundary_adjustment_limit_seconds: int = DEFAULT_BOUNDARY_ADJUSTMENT_LIMIT_SECONDS,
) -> list[dict[str, Any]]:
    """Rerank top-N candidates by base score and attach LLM/boundary outputs."""

    if top_n <= 0 or not candidates:
        return candidates

    ranked_indices = sorted(
        range(len(candidates)),
        key=lambda idx: float(candidates[idx].get("score", candidates[idx].get("base_score", 0.0))),
        reverse=True,
    )

    selected = set(ranked_indices[:top_n])
    reranked: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates):
        item = dict(candidate)
        if idx in selected:
            llm_result = rerank_with_local_llm(
                item,
                endpoint=endpoint,
                model=model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                boundary_adjustment_limit_seconds=boundary_adjustment_limit_seconds,
            )
            item["llm_score"] = llm_result["clip_worthiness"]
            item["llm_primary_type"] = llm_result["primary_type"]
            item["llm_reason"] = llm_result["reason"]
            item["boundary_adjustment_seconds"] = llm_result["boundary_adjustment_seconds"]
            if "adjusted_start_seconds" in llm_result:
                item["adjusted_start_seconds"] = llm_result["adjusted_start_seconds"]
            if "adjusted_end_seconds" in llm_result:
                item["adjusted_end_seconds"] = llm_result["adjusted_end_seconds"]
        reranked.append(item)

    return reranked


def _format_prompt(candidate_payload: dict[str, Any]) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
    candidate_json = json.dumps(candidate_payload, ensure_ascii=False, indent=2, sort_keys=True)
    return f"{template}\n\nCandidate JSON:\n{candidate_json}\n"


def _request_ollama(*, endpoint: str, model: str, prompt: str, timeout_seconds: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
    ).encode("utf-8")

    req = request.Request(
        f"{endpoint.rstrip('/')}/api/generate",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with request.urlopen(req, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    content = payload.get("response")
    if not isinstance(content, str):
        raise ValueError("Ollama response missing JSON text in 'response' field.")
    return content


def _validate_rerank_schema(payload: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {"clip_worthiness", "primary_type", "reason", "boundary_adjustment_seconds"}
    payload_keys = set(payload.keys())
    if payload_keys != expected_keys:
        raise ValueError(f"LLM output keys mismatch. Expected exactly {sorted(expected_keys)}.")

    clip_worthiness_raw = payload["clip_worthiness"]
    primary_type = payload["primary_type"]
    reason = payload["reason"]
    boundary_adjustment_raw = payload["boundary_adjustment_seconds"]

    clip_worthiness = float(clip_worthiness_raw)
    if not 0.0 <= clip_worthiness <= 1.0:
        raise ValueError("clip_worthiness must be in [0, 1].")

    if not isinstance(primary_type, str) or primary_type not in ALLOWED_PRIMARY_TYPES:
        raise ValueError(f"primary_type must be one of {sorted(ALLOWED_PRIMARY_TYPES)}.")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")

    if isinstance(boundary_adjustment_raw, bool):
        raise ValueError("boundary_adjustment_seconds must be an integer.")

    boundary_adjustment_seconds = int(boundary_adjustment_raw)

    return {
        "clip_worthiness": clip_worthiness,
        "primary_type": primary_type,
        "reason": reason.strip(),
        "boundary_adjustment_seconds": boundary_adjustment_seconds,
    }


def _apply_boundary_adjustment(
    *,
    candidate_payload: dict[str, Any],
    rerank_payload: dict[str, Any],
    limit_seconds: int,
) -> dict[str, Any]:
    adjusted = dict(rerank_payload)

    adjustment = int(adjusted["boundary_adjustment_seconds"])
    cap = max(0, limit_seconds)
    adjustment = max(-cap, min(cap, adjustment))
    adjusted["boundary_adjustment_seconds"] = adjustment

    if "start_seconds" in candidate_payload:
        start = float(candidate_payload["start_seconds"])
        adjusted["adjusted_start_seconds"] = max(0.0, start + adjustment)

    if "end_seconds" in candidate_payload:
        end = float(candidate_payload["end_seconds"])
        adjusted["adjusted_end_seconds"] = max(0.0, end + adjustment)

    return adjusted


def _fallback_result(candidate_payload: dict[str, Any], *, reason: str) -> dict[str, Any]:
    base_score = float(candidate_payload.get("base_score", candidate_payload.get("score", 0.0)))
    clamped_score = max(0.0, min(1.0, base_score))
    fallback = {
        "clip_worthiness": clamped_score,
        "primary_type": "unknown",
        "reason": reason,
        "boundary_adjustment_seconds": 0,
    }
    return _apply_boundary_adjustment(
        candidate_payload=candidate_payload,
        rerank_payload=fallback,
        limit_seconds=DEFAULT_BOUNDARY_ADJUSTMENT_LIMIT_SECONDS,
    )
