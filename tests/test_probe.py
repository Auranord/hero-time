from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.ingest.probe import _run_ffprobe


def test_run_ffprobe_wraps_missing_binary_error(tmp_path: Path) -> None:
    vod_path = tmp_path / "sample.mkv"
    vod_path.write_bytes(b"data")

    def _raise_missing(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("ffprobe")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "run", _raise_missing)
        with pytest.raises(RuntimeError, match="ffprobe executable was not found"):
            _run_ffprobe(vod_path)


def test_run_ffprobe_wraps_called_process_error(tmp_path: Path) -> None:
    vod_path = tmp_path / "sample.mkv"
    vod_path.write_bytes(b"data")

    command = ["ffprobe", str(vod_path)]

    def _raise_process_error(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=127,
            cmd=command,
            output="",
            stderr="ffprobe: command not found",
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "run", _raise_process_error)
        with pytest.raises(RuntimeError, match="ffprobe failed to read media file"):
            _run_ffprobe(vod_path)
