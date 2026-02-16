from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FeatureWindow:
    """A time-bounded multimodal feature view used for scoring."""

    start_seconds: float
    end_seconds: float
    values: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateClip:
    """Stable candidate schema shared between scoring and export."""

    vod_id: str
    clip_id: str
    start_seconds: float
    duration_seconds: int
    score: float
    reason_tags: list[str]
    llm_summary: str | None = None


@dataclass(slots=True)
class PipelineContext:
    """Shared state references between pipeline stages."""

    vod_path: str
    cache_dir: str
    output_dir: str
