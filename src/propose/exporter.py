from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.models import CandidateClip


def export_candidates(clips: list[CandidateClip], output_path: str) -> Path:
    """Export candidate clips to JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(clip) for clip in clips]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
