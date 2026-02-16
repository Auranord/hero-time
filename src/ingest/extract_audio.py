from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from src.ingest.probe import probe_media


def extract_audio_tracks(
    vod_path: str,
    cache_dir: str = "data/cache",
    target_sample_rate: int = 16000,
) -> dict[str, Any]:
    """Extract OBS audio tracks to WAV and persist extraction manifest."""

    metadata = probe_media(vod_path=vod_path, cache_dir=cache_dir)
    source_path = Path(metadata["vod_path"])
    ingest_dir = Path(metadata["cache_dir"])

    tracks: list[dict[str, Any]] = []
    for stream in metadata["streams"]:
        if stream["codec_type"] != "audio":
            continue

        stream_index = stream["index"]
        track_label = _stream_label(stream)
        output_path = ingest_dir / f"audio_track_{stream_index}_{track_label}.wav"

        was_cached = output_path.exists()
        if not was_cached:
            _run_ffmpeg_extract(
                source_path=source_path,
                stream_index=stream_index,
                output_path=output_path,
                target_sample_rate=target_sample_rate,
            )

        tracks.append(
            {
                "stream_index": stream_index,
                "label": track_label,
                "channels": stream.get("channels"),
                "sample_rate": target_sample_rate,
                "source_codec": stream.get("codec_name"),
                "path": str(output_path),
                "cached": was_cached,
            }
        )

    manifest = {
        "status": "ok",
        "vod_path": str(source_path),
        "cache_dir": str(ingest_dir),
        "track_count": len(tracks),
        "tracks": tracks,
    }

    manifest_path = ingest_dir / "audio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "metadata_path": metadata["metadata_path"],
        "ffprobe_raw_path": metadata["ffprobe_raw_path"],
    }


def _run_ffmpeg_extract(
    source_path: Path,
    stream_index: int,
    output_path: Path,
    target_sample_rate: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _stream_label(stream: dict[str, Any]) -> str:
    raw_label = (
        stream.get("tags", {}).get("title")
        or stream.get("tags", {}).get("handler_name")
        or f"stream_{stream.get('index')}"
    )
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw_label).strip("_")
    return sanitized or f"stream_{stream.get('index')}"
