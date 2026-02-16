from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from src.config import Settings, load_settings
from src.features.asr import run_asr
from src.features.diarization import run_diarization
from src.features.video_motion import analyze_video_motion
from src.ingest.extract_audio import extract_audio_tracks
from src.ingest.probe import probe_media
from src.logging_config import configure_logging

app = typer.Typer(help="Twitch VOD highlight pipeline (MVP scaffold).")
config_app = typer.Typer(help="Configuration commands.")
ingest_app = typer.Typer(help="Ingest commands.")
features_app = typer.Typer(help="Feature extraction commands.")

app.add_typer(config_app, name="config")
app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")

logger = logging.getLogger(__name__)


def _bootstrap(config_path: Path) -> Settings:
    settings = load_settings(config_path)
    configure_logging(settings.logging)
    logger.debug("Loaded runtime settings from %s", config_path)
    return settings


@config_app.command("show")
def show_config(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    )
) -> None:
    """Print resolved runtime configuration."""

    settings = _bootstrap(config_path)
    typer.echo(json.dumps(settings.model_dump(mode="json"), indent=2))


@app.command()
def plan(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    )
) -> None:
    """Backward-compatible alias for `config show`."""

    show_config(config_path)


@ingest_app.command("probe")
def probe(
    vod_path: str,
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Run ingest probe placeholder and print result JSON."""

    settings = _bootstrap(config_path)
    result = probe_media(vod_path=vod_path, cache_dir=str(settings.pipeline.cache_dir))
    logger.info("Probe completed for %s", vod_path)
    typer.echo(json.dumps(result, indent=2))


@ingest_app.command("extract-audio")
def extract_audio(
    vod_path: str,
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Extract all audio tracks from a VOD into ingest cache."""

    settings = _bootstrap(config_path)
    result = extract_audio_tracks(vod_path=vod_path, cache_dir=str(settings.pipeline.cache_dir))
    logger.info("Audio extraction completed for %s", vod_path)
    typer.echo(json.dumps(result, indent=2))


@features_app.command("asr")
def asr(
    audio_path: str,
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    ),
    language: str = typer.Option("de", help="Language code for transcription."),
    model_size: str = typer.Option("small", help="Faster-whisper model size/name."),
    chunk_seconds: int = typer.Option(120, help="Audio chunk size for transcription."),
) -> None:
    """Run chunked faster-whisper ASR and cache transcript artifacts."""

    settings = _bootstrap(config_path)
    result = run_asr(
        audio_path=audio_path,
        cache_dir=str(settings.pipeline.cache_dir),
        language=language,
        model_size=model_size,
        chunk_seconds=chunk_seconds,
    )
    logger.info("ASR completed for %s", audio_path)
    typer.echo(json.dumps(result, indent=2))


@features_app.command("diarize")
def diarize(
    audio_path: str,
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    ),
    transcript_path: Path | None = typer.Option(None, help="Optional ASR transcript.json for alignment."),
    hf_auth_token: str | None = typer.Option(None, help="Hugging Face auth token for pyannote model."),
    pipeline_model: str = typer.Option("pyannote/speaker-diarization-3.1", help="Pyannote pipeline model id."),
) -> None:
    """Run speaker diarization and compute overlap statistics."""

    settings = _bootstrap(config_path)
    result = run_diarization(
        audio_path=audio_path,
        cache_dir=str(settings.pipeline.cache_dir),
        transcript_path=str(transcript_path) if transcript_path else None,
        window_seconds=settings.pipeline.window_seconds,
        window_overlap_seconds=settings.pipeline.window_overlap_seconds,
        hf_auth_token=hf_auth_token,
        pipeline_model=pipeline_model,
    )
    logger.info("Diarization completed for %s", audio_path)
    typer.echo(json.dumps(result, indent=2))


@features_app.command("video-motion")
def video_motion(
    vod_path: str,
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    ),
    analysis_fps: float = typer.Option(2.0, help="Sampling FPS for motion analysis."),
) -> None:
    """Run low-FPS video motion/scene-change feature extraction."""

    settings = _bootstrap(config_path)
    result = analyze_video_motion(
        vod_path=vod_path,
        cache_dir=str(settings.pipeline.cache_dir),
        analysis_fps=analysis_fps,
        window_seconds=settings.pipeline.window_seconds,
        window_overlap_seconds=settings.pipeline.window_overlap_seconds,
    )
    logger.info("Video motion analysis completed for %s", vod_path)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
