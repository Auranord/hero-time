from __future__ import annotations

import csv
import json
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.models import CandidateClip


def export_candidates(clips: list[CandidateClip], output_path: str) -> Path:
    """Export candidate clips to JSON (default) or CSV, based on file extension."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        _write_csv(clips, path)
    else:
        _write_json(clips, path)

    return path


def export_final_outputs(
    clips: list[CandidateClip],
    output_dir: str | Path,
    *,
    basename: str = "candidates_final",
    vod_path: str | None = None,
    include_ffmpeg_commands: bool = True,
) -> dict[str, Path]:
    """Export final JSON/CSV contract files and a review manifest for quick triage."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    json_path = resolved_output_dir / f"{basename}.json"
    csv_path = resolved_output_dir / f"{basename}.csv"
    review_path = resolved_output_dir / f"{basename}_review.json"

    export_candidates(clips, str(json_path))
    export_candidates(clips, str(csv_path))

    review_manifest = generate_review_manifest(
        clips,
        vod_path=vod_path,
        include_ffmpeg_commands=include_ffmpeg_commands,
    )
    review_path.write_text(json.dumps(review_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "review": review_path,
    }


def generate_review_manifest(
    clips: list[CandidateClip],
    *,
    vod_path: str | None = None,
    include_ffmpeg_commands: bool = True,
) -> list[dict[str, Any]]:
    """Build a lightweight review manifest with confidence/reason summaries."""

    manifest: list[dict[str, Any]] = []
    for idx, clip in enumerate(clips, start=1):
        end_seconds = round(clip.start_seconds + clip.duration_seconds, 3)
        entry = {
            "index": idx,
            "vod_id": clip.vod_id,
            "clip_id": clip.clip_id,
            "start_seconds": clip.start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": clip.duration_seconds,
            "score": clip.score,
            "confidence": _confidence_label(clip.score),
            "reason_summary": _reason_summary(clip.reason_tags, clip.llm_summary),
        }
        if include_ffmpeg_commands and vod_path:
            entry["ffmpeg_command"] = build_ffmpeg_clip_command(
                vod_path=vod_path,
                clip=clip,
            )
        manifest.append(entry)

    return manifest


def build_ffmpeg_clip_command(
    *,
    vod_path: str,
    clip: CandidateClip,
    output_dir: str = "clips",
) -> str:
    """Generate a copy-paste ffmpeg command for a candidate clip snippet."""

    start = max(0.0, clip.start_seconds)
    output_name = f"{clip.clip_id}.mp4"
    output_path = f"{output_dir.rstrip('/')}/{output_name}"

    quoted_vod = shlex.quote(vod_path)
    quoted_output = shlex.quote(output_path)

    return (
        "ffmpeg "
        f"-ss {start:.3f} "
        f"-i {quoted_vod} "
        f"-t {int(clip.duration_seconds)} "
        "-c:v libx264 -preset veryfast -crf 18 "
        "-c:a aac -b:a 160k "
        f"{quoted_output}"
    )


def load_candidate_clips(path: str | Path) -> list[CandidateClip]:
    """Load candidate clips from exporter JSON contract for downstream review tooling."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Candidate contract must be a JSON array.")

    clips: list[CandidateClip] = []
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Candidate row {idx} must be an object.")
        clips.append(
            CandidateClip(
                vod_id=str(row["vod_id"]),
                clip_id=str(row["clip_id"]),
                start_seconds=float(row["start_seconds"]),
                duration_seconds=int(row["duration_seconds"]),
                score=float(row["score"]),
                reason_tags=[str(tag) for tag in row.get("reason_tags", [])],
                llm_summary=str(row["llm_summary"]) if row.get("llm_summary") is not None else None,
            )
        )

    return clips


def _write_json(clips: list[CandidateClip], path: Path) -> None:
    payload = [asdict(clip) for clip in clips]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(clips: list[CandidateClip], path: Path) -> None:
    fields = [
        "vod_id",
        "clip_id",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "score",
        "confidence",
        "reason_summary",
        "reason_tags",
        "llm_summary",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip in clips:
            end_seconds = round(clip.start_seconds + clip.duration_seconds, 3)
            writer.writerow(
                {
                    "vod_id": clip.vod_id,
                    "clip_id": clip.clip_id,
                    "start_seconds": f"{clip.start_seconds:.3f}",
                    "end_seconds": f"{end_seconds:.3f}",
                    "duration_seconds": clip.duration_seconds,
                    "score": f"{clip.score:.4f}",
                    "confidence": _confidence_label(clip.score),
                    "reason_summary": _reason_summary(clip.reason_tags, clip.llm_summary),
                    "reason_tags": "|".join(clip.reason_tags),
                    "llm_summary": clip.llm_summary or "",
                }
            )


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _reason_summary(reason_tags: list[str], llm_summary: str | None) -> str:
    if llm_summary and llm_summary.strip():
        return llm_summary.strip()
    if reason_tags:
        return ", ".join(reason_tags)
    return "no strong highlight signals"
