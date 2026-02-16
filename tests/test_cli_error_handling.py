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
    monkeypatch.setattr(cli, "analyze_video_motion", lambda **_: {"video_motion_path": "/tmp/video_motion.json"})
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
