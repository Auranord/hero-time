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


class _DummySettings:
    pipeline = _DummyPipeline()


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
    assert "Error: ffprobe is installed but failed to start" in result.output
    assert "Traceback" not in result.output
