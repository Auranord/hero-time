from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

import typer

from src.config import Settings, load_settings
from src.features.audio_events import detect_audio_events
from src.features.asr import run_asr
from src.features.diarization import run_diarization
from src.features.video_motion import analyze_video_motion
from src.ingest.extract_audio import extract_audio_tracks
from src.ingest.probe import probe_media
from src.logging_config import configure_logging
from src.propose.exporter import export_final_outputs, load_candidate_clips
from src.pipeline_candidate_builder import build_candidates_from_artifacts

app = typer.Typer(help="Twitch VOD highlight pipeline (MVP scaffold).")
config_app = typer.Typer(help="Configuration commands.")
ingest_app = typer.Typer(help="Ingest commands.")
features_app = typer.Typer(help="Feature extraction commands.")
propose_app = typer.Typer(help="Proposal output and review commands.")

app.add_typer(config_app, name="config")
app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(propose_app, name="propose")

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _run_with_progress(step_index: int, total_steps: int, label: str, work: Callable[[], T]) -> T:
    typer.echo(f"[{step_index}/{total_steps}] {label}...", err=True)
    started_at = perf_counter()
    try:
        result = work()
    except Exception:
        elapsed = perf_counter() - started_at
        typer.echo(f"[{step_index}/{total_steps}] {label} failed after {elapsed:.1f}s", err=True)
        raise
    elapsed = perf_counter() - started_at
    typer.echo(f"[{step_index}/{total_steps}] {label} done in {elapsed:.1f}s", err=True)
    return result


def _load_cached_payload(path: Path, label: str, step_index: int, total_steps: int) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load cached payload at %s (%s); recomputing step.", path, exc)
        return None

    typer.echo(f"[{step_index}/{total_steps}] {label} cached: {path}", err=True)
    payload.setdefault("cache_path", str(path))
    return payload


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
    device: str = typer.Option("auto", help="ASR device: auto, cpu, cuda, cuda:0, ..."),
    compute_type: str = typer.Option("default", help="faster-whisper compute type: default, float16, int8, ..."),
) -> None:
    """Run chunked faster-whisper ASR and cache transcript artifacts."""

    settings = _bootstrap(config_path)
    result = run_asr(
        audio_path=audio_path,
        cache_dir=str(settings.pipeline.cache_dir),
        language=language,
        model_size=model_size,
        chunk_seconds=chunk_seconds,
        device=device,
        compute_type=compute_type,
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
    device: str = typer.Option("auto", help="Diarization device: auto, cpu, cuda, cuda:0, ..."),
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
        device=device,
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
    processing_width: int = typer.Option(320, help="Resize video width before motion scoring (<=0 disables resize)."),
) -> None:
    """Run low-FPS video motion/scene-change feature extraction."""

    settings = _bootstrap(config_path)
    result = analyze_video_motion(
        vod_path=vod_path,
        cache_dir=str(settings.pipeline.cache_dir),
        analysis_fps=analysis_fps,
        processing_width=processing_width,
        window_seconds=settings.pipeline.window_seconds,
        window_overlap_seconds=settings.pipeline.window_overlap_seconds,
    )
    logger.info("Video motion analysis completed for %s", vod_path)
    typer.echo(json.dumps(result, indent=2))


@features_app.command("audio-events")
def audio_events(
    audio_path: str,
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Run lightweight audio-event extraction (loudness/reaction proxies)."""

    settings = _bootstrap(config_path)
    result = detect_audio_events(
        audio_path=audio_path,
        cache_dir=str(settings.pipeline.cache_dir),
        window_seconds=settings.pipeline.window_seconds,
        window_overlap_seconds=settings.pipeline.window_overlap_seconds,
    )
    logger.info("Audio event extraction completed for %s", audio_path)
    typer.echo(json.dumps(result, indent=2))


@propose_app.command("review")
def review_candidates(
    candidates_path: Path = typer.Argument(..., help="Path to candidate JSON contract."),
    output_dir: Path = typer.Option(Path("data/outputs"), "--output-dir", "-o", help="Directory for JSON/CSV/review outputs."),
    basename: str = typer.Option("candidates_final", help="Base filename for exported artifacts."),
    vod_path: str | None = typer.Option(None, help="Optional source VOD path for ffmpeg command generation."),
    include_ffmpeg_commands: bool = typer.Option(True, help="Include ffmpeg clip commands in review manifest when vod_path is provided."),
) -> None:
    """Export final JSON/CSV contract artifacts and review manifest."""

    clips = load_candidate_clips(candidates_path)
    exported = export_final_outputs(
        clips=clips,
        output_dir=output_dir,
        basename=basename,
        vod_path=vod_path,
        include_ffmpeg_commands=include_ffmpeg_commands,
    )
    typer.echo(
        json.dumps(
            {key: str(path) for key, path in exported.items()},
            indent=2,
        )
    )


@propose_app.command("build")
def build_candidates(
    vod_id: str,
    asr_path: Path = typer.Option(..., help="Path to ASR transcript.json artifact."),
    diarization_path: Path = typer.Option(..., help="Path to diarization.json artifact."),
    video_motion_path: Path = typer.Option(..., help="Path to video_motion.json artifact."),
    audio_events_path: Path = typer.Option(..., help="Path to audio_events.json artifact."),
    output_dir: Path = typer.Option(Path("data/outputs"), "--output-dir", "-o", help="Directory for JSON/CSV/review outputs."),
    basename: str = typer.Option("candidates_final", help="Base filename for exported artifacts."),
    top_k: int = typer.Option(20, help="Maximum number of candidate windows before merge/cooldown."),
    rerank_top_n: int = typer.Option(0, help="Optional top-N candidates to rerank with local LLM."),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Build final candidate clips from cached feature artifacts."""

    settings = _bootstrap(config_path)

    asr_payload = json.loads(asr_path.read_text(encoding="utf-8"))
    diarization_payload = json.loads(diarization_path.read_text(encoding="utf-8"))
    video_motion_payload = json.loads(video_motion_path.read_text(encoding="utf-8"))
    audio_events_payload = json.loads(audio_events_path.read_text(encoding="utf-8"))

    clips = build_candidates_from_artifacts(
        vod_id=vod_id,
        asr_payload=asr_payload,
        diarization_payload=diarization_payload,
        video_motion_payload=video_motion_payload,
        audio_events_payload=audio_events_payload,
        weights=settings.weights.model_dump(mode="python"),
        strategy=settings.scoring.strategy,
        hybrid_alpha=settings.scoring.hybrid_alpha,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        llm_endpoint=settings.llm.endpoint,
        llm_model=settings.llm.model,
        llm_timeout_seconds=settings.llm.timeout_seconds,
    )

    exported = export_final_outputs(
        clips=clips,
        output_dir=output_dir,
        basename=basename,
    )

    typer.echo(json.dumps({"clip_count": len(clips), **{k: str(v) for k, v in exported.items()}}, indent=2))


@app.command("run")
def run_pipeline(
    vod_path: str,
    vod_id: str | None = typer.Option(None, help="Optional VOD id. Defaults to source filename stem."),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        envvar="VOD_HIGHLIGHTS_CONFIG",
        help="Path to YAML configuration file.",
    ),
    language: str = typer.Option("de", help="Language code for ASR transcription."),
    model_size: str = typer.Option("small", help="Faster-whisper model size/name."),
    analysis_fps: float = typer.Option(2.0, help="Sampling FPS for video motion analysis."),
    motion_processing_width: int = typer.Option(320, help="Resize width before motion scoring (<=0 disables resize)."),
    top_k: int = typer.Option(20, help="Maximum number of candidate windows before merge/cooldown."),
    rerank_top_n: int = typer.Option(0, help="Optional top-N candidates to rerank with local LLM."),
    hf_auth_token: str | None = typer.Option(None, help="Optional Hugging Face auth token for diarization."),
    pipeline_model: str = typer.Option("pyannote/speaker-diarization-3.1", help="Pyannote pipeline model id."),
    allow_diarization_fallback: bool = typer.Option(
        True,
        help="If diarization fails, continue with a synthetic single-speaker timeline so pipeline still completes.",
    ),
    asr_device: str = typer.Option("auto", help="ASR device: auto, cpu, cuda, cuda:0, ..."),
    asr_compute_type: str = typer.Option("default", help="faster-whisper compute type: default, float16, int8, ..."),
    diarization_device: str = typer.Option("auto", help="Diarization device: auto, cpu, cuda, cuda:0, ..."),
    use_cache: bool = typer.Option(True, help="Reuse existing cached step artifacts when available."),
) -> None:
    """Run the complete pipeline from a VOD file using project-local default paths."""

    settings = _bootstrap(config_path)
    resolved_vod_path = Path(vod_path).expanduser().resolve()
    if not resolved_vod_path.exists():
        raise FileNotFoundError(f"VOD file not found: {resolved_vod_path}")

    resolved_vod_id = vod_id or resolved_vod_path.stem

    total_steps = 8

    cache_root = Path(settings.pipeline.cache_dir).expanduser().resolve()
    ingest_dir = cache_root / "ingest" / resolved_vod_path.stem

    try:
        probe_cache_path = ingest_dir / "metadata.json"
        probe_result = _load_cached_payload(probe_cache_path, "Probe media", 1, total_steps) if use_cache else None
        if probe_result is None:
            probe_result = _run_with_progress(
                1,
                total_steps,
                "Probe media",
                lambda: probe_media(vod_path=str(resolved_vod_path), cache_dir=str(settings.pipeline.cache_dir)),
            )

        extract_cache_path = ingest_dir / "audio_manifest.json"
        extract_result = _load_cached_payload(extract_cache_path, "Extract audio tracks", 2, total_steps) if use_cache else None
        if extract_result is None:
            extract_result = _run_with_progress(
                2,
                total_steps,
                "Extract audio tracks",
                lambda: extract_audio_tracks(vod_path=str(resolved_vod_path), cache_dir=str(settings.pipeline.cache_dir)),
            )

        audio_track_path = _select_default_audio_track(extract_result)
        audio_track_stem = Path(audio_track_path).stem

        asr_cache_path = cache_root / "features" / "asr" / audio_track_stem / "transcript.json"
        asr_result = _load_cached_payload(asr_cache_path, "Run ASR", 3, total_steps) if use_cache else None
        if asr_result is None:
            asr_result = _run_with_progress(
                3,
                total_steps,
                "Run ASR",
                lambda: run_asr(
                    audio_path=audio_track_path,
                    cache_dir=str(settings.pipeline.cache_dir),
                    language=language,
                    model_size=model_size,
                    chunk_seconds=int(settings.pipeline.chunk_minutes * 60),
                    device=asr_device,
                    compute_type=asr_compute_type,
                ),
            )

        diarization_token = hf_auth_token or os.getenv("HF_TOKEN")

        def _run_diarization_stage() -> dict[str, Any]:
            try:
                return run_diarization(
                    audio_path=audio_track_path,
                    cache_dir=str(settings.pipeline.cache_dir),
                    transcript_path=asr_result.get("transcript_path"),
                    window_seconds=settings.pipeline.window_seconds,
                    window_overlap_seconds=settings.pipeline.window_overlap_seconds,
                    hf_auth_token=diarization_token,
                    pipeline_model=pipeline_model,
                    device=diarization_device,
                )
            except Exception as exc:
                if not allow_diarization_fallback:
                    raise
                logger.warning("Diarization failed; continuing with fallback timeline: %s", exc)
                typer.echo("[4/8] Diarization fallback activated.", err=True)
                return _fallback_diarization_payload(
                    asr_result=asr_result,
                    window_seconds=settings.pipeline.window_seconds,
                    window_overlap_seconds=settings.pipeline.window_overlap_seconds,
                    cache_dir=settings.pipeline.cache_dir,
                    audio_track_path=audio_track_path,
                )

        diarization_cache_path = cache_root / "features" / "diarization" / audio_track_stem / "diarization.json"
        diarization_result = _load_cached_payload(diarization_cache_path, "Run speaker diarization", 4, total_steps) if use_cache else None
        if diarization_result is None and use_cache:
            fallback_cache = cache_root / "features" / "diarization" / audio_track_stem / "diarization_fallback.json"
            diarization_result = _load_cached_payload(fallback_cache, "Run speaker diarization", 4, total_steps)
        if diarization_result is None:
            diarization_result = _run_with_progress(4, total_steps, "Run speaker diarization", _run_diarization_stage)

        audio_events_cache_path = cache_root / "features" / "audio_events" / audio_track_stem / "audio_events.json"
        audio_events_result = _load_cached_payload(audio_events_cache_path, "Detect audio events", 5, total_steps) if use_cache else None
        if audio_events_result is None:
            audio_events_result = _run_with_progress(
                5,
                total_steps,
                "Detect audio events",
                lambda: detect_audio_events(
                    audio_path=audio_track_path,
                    cache_dir=str(settings.pipeline.cache_dir),
                    window_seconds=settings.pipeline.window_seconds,
                    window_overlap_seconds=settings.pipeline.window_overlap_seconds,
                ),
            )

        video_motion_cache_path = cache_root / "features" / "video_motion" / resolved_vod_path.stem / "video_motion.json"
        video_motion_result = _load_cached_payload(video_motion_cache_path, "Analyze video motion", 6, total_steps) if use_cache else None
        if video_motion_result is None:
            video_motion_result = _run_with_progress(
                6,
                total_steps,
                "Analyze video motion",
                lambda: analyze_video_motion(
                    vod_path=str(resolved_vod_path),
                    cache_dir=str(settings.pipeline.cache_dir),
                    analysis_fps=analysis_fps,
                    processing_width=motion_processing_width,
                    window_seconds=settings.pipeline.window_seconds,
                    window_overlap_seconds=settings.pipeline.window_overlap_seconds,
                ),
            )

        clips = _run_with_progress(
            7,
            total_steps,
            "Build highlight candidates",
            lambda: build_candidates_from_artifacts(
                vod_id=resolved_vod_id,
                asr_payload=asr_result,
                diarization_payload=diarization_result,
                video_motion_payload=video_motion_result,
                audio_events_payload=audio_events_result,
                weights=settings.weights.model_dump(mode="python"),
                strategy=settings.scoring.strategy,
                hybrid_alpha=settings.scoring.hybrid_alpha,
                top_k=top_k,
                rerank_top_n=rerank_top_n,
                llm_endpoint=settings.llm.endpoint,
                llm_model=settings.llm.model,
                llm_timeout_seconds=settings.llm.timeout_seconds,
            ),
        )

        exported = _run_with_progress(
            8,
            total_steps,
            "Export outputs",
            lambda: export_final_outputs(
                clips=clips,
                output_dir=settings.pipeline.output_dir,
                basename=f"{resolved_vod_id}_candidates",
                vod_path=str(resolved_vod_path),
                include_ffmpeg_commands=True,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "vod_id": resolved_vod_id,
                "vod_path": str(resolved_vod_path),
                "selected_audio_track": audio_track_path,
                "clip_count": len(clips),
                "probe_metadata_path": probe_result.get("metadata_path"),
                "artifacts": {
                    "asr": asr_result.get("transcript_path"),
                    "diarization": diarization_result.get("diarization_path"),
                    "audio_events": audio_events_result.get("audio_events_path"),
                    "video_motion": video_motion_result.get("video_motion_path"),
                },
                "outputs": {k: str(v) for k, v in exported.items()},
            },
            indent=2,
        )
    )


def _select_default_audio_track(extract_result: dict[str, Any]) -> str:
    tracks = extract_result.get("tracks", [])
    if not tracks:
        raise ValueError("No audio tracks found in extracted media manifest.")
    return str(tracks[0]["path"])


def _fallback_diarization_payload(
    *,
    asr_result: dict[str, Any],
    window_seconds: int,
    window_overlap_seconds: int,
    cache_dir: Path,
    audio_track_path: str,
) -> dict[str, Any]:
    duration_seconds = float(asr_result.get("duration_seconds", 0.0))
    step_seconds = max(window_seconds - window_overlap_seconds, 1)
    windows: list[dict[str, Any]] = []
    index = 0
    start = 0.0
    while start < duration_seconds:
        end = min(start + window_seconds, duration_seconds)
        windows.append(
            {
                "window_index": index,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "overlap_seconds": 0.0,
                "overlap_ratio": 0.0,
                "active_speakers": 1,
            }
        )
        start += float(step_seconds)
        index += 1

    fallback_dir = Path(cache_dir).expanduser().resolve() / "features" / "diarization" / Path(audio_track_path).stem
    fallback_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "fallback",
        "audio_path": str(Path(audio_track_path).expanduser().resolve()),
        "cache_dir": str(fallback_dir),
        "pipeline_model": "fallback-single-speaker",
        "speaker_count": 1,
        "speakers": ["speaker_00"],
        "duration_seconds": duration_seconds,
        "speaker_turn_count": 1 if duration_seconds > 0 else 0,
        "speaker_turns": [
            {
                "id": "turn_000000",
                "speaker": "speaker_00",
                "start_seconds": 0.0,
                "end_seconds": round(duration_seconds, 3),
                "duration_seconds": round(duration_seconds, 3),
            }
        ]
        if duration_seconds > 0
        else [],
        "aligned_transcript_segment_count": 0,
        "aligned_transcript_segments": [],
        "overlap": {
            "global_overlap_seconds": 0.0,
            "global_overlap_ratio": 0.0,
        },
        "window_seconds": window_seconds,
        "window_overlap_seconds": window_overlap_seconds,
        "window_overlap_stats": windows,
        "speaking_time_seconds": {"speaker_00": round(duration_seconds, 3)},
    }
    artifact_path = fallback_dir / "diarization_fallback.json"
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        **payload,
        "diarization_path": str(artifact_path),
    }


if __name__ == "__main__":
    app()
