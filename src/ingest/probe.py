from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def probe_media(vod_path: str, cache_dir: str = "data/cache") -> dict[str, Any]:
    """Probe media metadata via ffprobe and persist ingest artifacts."""

    source_path = Path(vod_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"VOD file not found: {source_path}")

    ingest_dir = Path(cache_dir).expanduser().resolve() / "ingest" / source_path.stem
    ingest_dir.mkdir(parents=True, exist_ok=True)

    raw_probe_path = ingest_dir / "ffprobe_raw.json"
    metadata_path = ingest_dir / "metadata.json"

    ffprobe_payload = _run_ffprobe(source_path)
    normalized_metadata = _normalize_probe_payload(source_path, ffprobe_payload)

    raw_probe_path.write_text(
        json.dumps(ffprobe_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(normalized_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        **normalized_metadata,
        "cache_dir": str(ingest_dir),
        "ffprobe_raw_path": str(raw_probe_path),
        "metadata_path": str(metadata_path),
    }


def _run_ffprobe(vod_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(vod_path),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe executable was not found. Install FFmpeg so ffprobe is available on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        details = f" ffprobe stderr: {stderr}" if stderr else ""
        raise RuntimeError(
            f"ffprobe failed to read media file: {vod_path}.{details}"
        ) from exc

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON output.") from exc


def _normalize_probe_payload(vod_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    stream_entries = payload.get("streams", [])
    format_entry = payload.get("format", {})

    streams = [_normalize_stream(stream) for stream in stream_entries]
    audio_streams = [stream for stream in streams if stream["codec_type"] == "audio"]

    return {
        "status": "ok",
        "vod_path": str(vod_path),
        "format": {
            "format_name": format_entry.get("format_name"),
            "format_long_name": format_entry.get("format_long_name"),
            "duration_seconds": _to_float(format_entry.get("duration")),
            "size_bytes": _to_int(format_entry.get("size")),
            "bit_rate": _to_int(format_entry.get("bit_rate")),
            "tags": format_entry.get("tags", {}),
        },
        "streams": streams,
        "audio_stream_count": len(audio_streams),
        "video_stream_count": sum(1 for stream in streams if stream["codec_type"] == "video"),
    }


def _normalize_stream(stream: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": stream.get("index"),
        "codec_type": stream.get("codec_type"),
        "codec_name": stream.get("codec_name"),
        "codec_long_name": stream.get("codec_long_name"),
        "sample_rate": _to_int(stream.get("sample_rate")),
        "channels": _to_int(stream.get("channels")),
        "width": _to_int(stream.get("width")),
        "height": _to_int(stream.get("height")),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration_seconds": _to_float(stream.get("duration")),
        "bit_rate": _to_int(stream.get("bit_rate")),
        "tags": stream.get("tags", {}),
    }


def _to_float(raw_value: Any) -> float | None:
    if raw_value in (None, "N/A", ""):
        return None
    return float(raw_value)


def _to_int(raw_value: Any) -> int | None:
    if raw_value in (None, "N/A", ""):
        return None
    return int(raw_value)
