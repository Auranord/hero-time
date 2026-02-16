from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
ENV_PREFIX = "VOD_HIGHLIGHTS_"


class PipelineSettings(BaseModel):
    chunk_minutes: int = 10
    window_seconds: int = 30
    window_overlap_seconds: int = 15
    output_dir: Path = Path("data/outputs")
    cache_dir: Path = Path("data/cache")


class WeightSettings(BaseModel):
    loudness_spike: float = 0.2
    overlap_speech: float = 0.15
    speech_rate_burst: float = 0.1
    transcript_excitement: float = 0.2
    motion_peak: float = 0.2
    scene_change_rate: float = 0.15


class ScoringSettings(BaseModel):
    strategy: str = "hybrid"
    hybrid_alpha: float = 0.65


class LLMSettings(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    endpoint: str = "http://localhost:11434"
    timeout_seconds: int = 45


class LoggingSettings(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    weights: WeightSettings = Field(default_factory=WeightSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load typed settings from YAML with environment-variable overrides."""

    resolved_path = Path(
        config_path
        or os.getenv(f"{ENV_PREFIX}CONFIG")
        or DEFAULT_CONFIG_PATH
    )
    raw_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    data = Settings.model_validate(raw_config).model_dump(mode="python")

    for key, raw_value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue

        suffix = key[len(ENV_PREFIX) :]
        if suffix == "CONFIG":
            continue

        path = [part.lower() for part in suffix.split("__")]
        _apply_override(data, path, raw_value)

    return Settings.model_validate(data)


def _apply_override(data: dict[str, Any], path: list[str], raw_value: str) -> None:
    current: Any = data
    for segment in path[:-1]:
        if not isinstance(current, dict) or segment not in current:
            return
        current = current[segment]

    if not isinstance(current, dict):
        return

    final_key = path[-1]
    if final_key not in current:
        return

    current[final_key] = _coerce_value(raw_value, current[final_key])


def _coerce_value(raw_value: str, existing_value: Any) -> Any:
    if isinstance(existing_value, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(existing_value, int) and not isinstance(existing_value, bool):
        return int(raw_value)
    if isinstance(existing_value, float):
        return float(raw_value)
    if isinstance(existing_value, list | dict):
        return json.loads(raw_value)
    if isinstance(existing_value, Path):
        return Path(raw_value)
    return raw_value
