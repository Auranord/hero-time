from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


class _DummyPipeline:
    cache_dir = Path("data/cache")
    chunk_minutes = 2
    window_seconds = 30
    window_overlap_seconds = 15
    output_dir = Path("data/output")


class _DummyWeights:
    def model_dump(self, mode: str = "python") -> dict[str, float]:
        return {"asr": 1.0}


class _DummyScoring:
    strategy = "weighted"
    hybrid_alpha = 0.5


class _DummyLlm:
    endpoint = "http://localhost:11434"
    model = "dummy"
    timeout_seconds = 30


class _DummySettings:
    pipeline = _DummyPipeline()
    weights = _DummyWeights()
    scoring = _DummyScoring()
    llm = _DummyLlm()


def test_run_command_prints_clean_error_without_traceback(tmp_path: Path, monkeypatch) -> None:
    vod_path = tmp_path / "sample.mkv"
    vod_path.write_bytes(b"data")

    monkeypatch.setattr(cli, "_bootstrap", lambda _: _DummySettings())
    monkeypatch.setattr(
        cli,
        "probe_media",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError("ffprobe is installed but failed to start because required shared libraries are missing")
        ),
    )

    result = CliRunner().invoke(cli.app, ["run", str(vod_path)])

    assert result.exit_code == 1
    assert "[1/8] Probe media..." in result.output
    assert "[1/8] Probe media failed" in result.output
    assert "Error: ffprobe is installed but failed to start" in result.output
    assert "Traceback" not in result.output


def test_run_command_shows_progress_for_all_stages(tmp_path: Path, monkeypatch) -> None:
    vod_path = tmp_path / "sample.mkv"
    vod_path.write_bytes(b"data")

    monkeypatch.setattr(cli, "_bootstrap", lambda _: _DummySettings())
    monkeypatch.setattr(cli, "probe_media", lambda **_: {"metadata_path": "/tmp/metadata.json"})
    monkeypatch.setattr(cli, "extract_audio_tracks", lambda **_: {"tracks": [{"path": "/tmp/audio.wav"}]})
    monkeypatch.setattr(cli, "run_asr", lambda **_: {"transcript_path": "/tmp/asr.json"})
    monkeypatch.setattr(cli, "run_diarization", lambda **_: {"diarization_path": "/tmp/diarization.json"})
    monkeypatch.setattr(cli, "detect_audio_events", lambda **_: {"audio_events_path": "/tmp/audio_events.json"})

    def _analyze_video_motion(**kwargs):
        return {"video_motion_path": "/tmp/video_motion.json"}

    monkeypatch.setattr(cli, "analyze_video_motion", _analyze_video_motion)
    monkeypatch.setattr(cli, "build_candidates_from_artifacts", lambda **_: [{"id": "clip-1"}])
    monkeypatch.setattr(
        cli,
        "export_final_outputs",
        lambda **_: {"json_path": Path("/tmp/out.json"), "csv_path": Path("/tmp/out.csv")},
    )

    result = CliRunner().invoke(cli.app, ["run", str(vod_path)])

    assert result.exit_code == 0
    assert "[1/8] Probe media..." in result.output
    assert "[8/8] Export outputs done" in result.output
    assert '"status": "ok"' in result.output



def test_run_command_passes_device_flags_to_asr_and_diarization(tmp_path: Path, monkeypatch) -> None:
    vod_path = tmp_path / "sample.mkv"
    vod_path.write_bytes(b"data")

    captured_asr: dict[str, str] = {}
    captured_diar: dict[str, str] = {}
    captured_motion: dict[str, int] = {}

    monkeypatch.setattr(cli, "_bootstrap", lambda _: _DummySettings())
    monkeypatch.setattr(cli, "probe_media", lambda **_: {"metadata_path": "/tmp/metadata.json"})
    monkeypatch.setattr(cli, "extract_audio_tracks", lambda **_: {"tracks": [{"path": "/tmp/audio.wav"}]})

    def _run_asr(**kwargs):
        captured_asr["device"] = kwargs["device"]
        captured_asr["compute_type"] = kwargs["compute_type"]
        return {"transcript_path": "/tmp/asr.json"}

    def _run_diarization(**kwargs):
        captured_diar["device"] = kwargs["device"]
        return {"diarization_path": "/tmp/diarization.json"}

    monkeypatch.setattr(cli, "run_asr", _run_asr)
    monkeypatch.setattr(cli, "run_diarization", _run_diarization)
    monkeypatch.setattr(cli, "detect_audio_events", lambda **_: {"audio_events_path": "/tmp/audio_events.json"})

    def _analyze_video_motion(**kwargs):
        captured_motion["processing_width"] = kwargs["processing_width"]
        return {"video_motion_path": "/tmp/video_motion.json"}

    monkeypatch.setattr(cli, "analyze_video_motion", _analyze_video_motion)
    monkeypatch.setattr(cli, "build_candidates_from_artifacts", lambda **_: [{"id": "clip-1"}])
    monkeypatch.setattr(
        cli,
        "export_final_outputs",
        lambda **_: {"json_path": Path("/tmp/out.json"), "csv_path": Path("/tmp/out.csv")},
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            str(vod_path),
            "--asr-device",
            "cuda",
            "--asr-compute-type",
            "float16",
            "--diarization-device",
            "cuda:0",
            "--motion-processing-width",
            "224",
        ],
    )

    assert result.exit_code == 0
    assert captured_asr == {"device": "cuda", "compute_type": "float16"}
    assert captured_diar == {"device": "cuda:0"}
    assert captured_motion == {"processing_width": 224}



def test_run_command_reuses_cached_step_artifacts(tmp_path: Path, monkeypatch) -> None:
    vod_path = tmp_path / "sample.mkv"
    vod_path.write_bytes(b"data")

    cache_root = tmp_path / "cache"
    ingest_dir = cache_root / "ingest" / vod_path.stem
    ingest_dir.mkdir(parents=True, exist_ok=True)

    (ingest_dir / "metadata.json").write_text('{"metadata_path": "cached"}', encoding="utf-8")
    (ingest_dir / "audio_manifest.json").write_text(
        '{"tracks": [{"path": "' + str(tmp_path / "audio.wav") + '"}]}',
        encoding="utf-8",
    )

    audio_stem = "audio"
    (cache_root / "features" / "asr" / audio_stem).mkdir(parents=True, exist_ok=True)
    (cache_root / "features" / "diarization" / audio_stem).mkdir(parents=True, exist_ok=True)
    (cache_root / "features" / "audio_events" / audio_stem).mkdir(parents=True, exist_ok=True)
    (cache_root / "features" / "video_motion" / vod_path.stem).mkdir(parents=True, exist_ok=True)

    (cache_root / "features" / "asr" / audio_stem / "transcript.json").write_text("{}", encoding="utf-8")
    (cache_root / "features" / "diarization" / audio_stem / "diarization.json").write_text("{}", encoding="utf-8")
    (cache_root / "features" / "audio_events" / audio_stem / "audio_events.json").write_text("{}", encoding="utf-8")
    (cache_root / "features" / "video_motion" / vod_path.stem / "video_motion.json").write_text("{}", encoding="utf-8")

    class _Pipeline:
        cache_dir = cache_root
        chunk_minutes = 2
        window_seconds = 30
        window_overlap_seconds = 15
        output_dir = tmp_path / "outputs"

    class _Settings(_DummySettings):
        pipeline = _Pipeline()

    monkeypatch.setattr(cli, "_bootstrap", lambda _: _Settings())

    monkeypatch.setattr(cli, "probe_media", lambda **_: (_ for _ in ()).throw(AssertionError("probe should be skipped")))
    monkeypatch.setattr(cli, "extract_audio_tracks", lambda **_: (_ for _ in ()).throw(AssertionError("extract should be skipped")))
    monkeypatch.setattr(cli, "run_asr", lambda **_: (_ for _ in ()).throw(AssertionError("asr should be skipped")))
    monkeypatch.setattr(cli, "run_diarization", lambda **_: (_ for _ in ()).throw(AssertionError("diarization should be skipped")))
    monkeypatch.setattr(cli, "detect_audio_events", lambda **_: (_ for _ in ()).throw(AssertionError("audio events should be skipped")))
    monkeypatch.setattr(cli, "analyze_video_motion", lambda **_: (_ for _ in ()).throw(AssertionError("video motion should be skipped")))
    monkeypatch.setattr(cli, "build_candidates_from_artifacts", lambda **_: [{"id": "clip-1"}])
    monkeypatch.setattr(
        cli,
        "export_final_outputs",
        lambda **_: {"json_path": Path("/tmp/out.json"), "csv_path": Path("/tmp/out.csv")},
    )

    result = CliRunner().invoke(cli.app, ["run", str(vod_path)])

    assert result.exit_code == 0
    assert "[1/8] Probe media cached" in result.output
    assert "[6/8] Analyze video motion cached" in result.output
