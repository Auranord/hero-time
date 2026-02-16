from __future__ import annotations

import csv
import json

from src.models import CandidateClip
from src.propose.exporter import (
    build_ffmpeg_clip_command,
    export_candidates,
    export_final_outputs,
    generate_review_manifest,
    load_candidate_clips,
)


def _sample_clips() -> list[CandidateClip]:
    return [
        CandidateClip(
            vod_id="vod-1",
            clip_id="c_0001",
            start_seconds=12.5,
            duration_seconds=30,
            score=0.83,
            reason_tags=["signal:loudness_spike", "signal:motion_peak"],
            llm_summary="strong reaction payoff",
        ),
        CandidateClip(
            vod_id="vod-1",
            clip_id="c_0002",
            start_seconds=70.0,
            duration_seconds=25,
            score=0.55,
            reason_tags=["signal:overlap_speech"],
            llm_summary=None,
        ),
    ]


def test_export_candidates_json_contract_compatible(tmp_path) -> None:
    out = tmp_path / "candidates.json"
    export_candidates(_sample_clips(), str(out))

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload[0]["vod_id"] == "vod-1"
    assert payload[0]["clip_id"] == "c_0001"
    assert payload[0]["reason_tags"] == ["signal:loudness_spike", "signal:motion_peak"]


def test_export_candidates_csv_contains_confidence_and_reason_summary(tmp_path) -> None:
    out = tmp_path / "candidates.csv"
    export_candidates(_sample_clips(), str(out))

    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["confidence"] == "high"
    assert rows[0]["reason_summary"] == "strong reaction payoff"
    assert rows[1]["confidence"] == "low"
    assert rows[1]["reason_summary"] == "signal:overlap_speech"


def test_generate_review_manifest_with_ffmpeg_command() -> None:
    manifest = generate_review_manifest(_sample_clips(), vod_path="/tmp/vod.mkv", include_ffmpeg_commands=True)

    assert manifest[0]["confidence"] == "high"
    assert "ffmpeg_command" in manifest[0]
    assert "-ss 12.500" in manifest[0]["ffmpeg_command"]


def test_export_final_outputs_and_load_roundtrip(tmp_path) -> None:
    exported = export_final_outputs(
        _sample_clips(),
        output_dir=tmp_path,
        basename="final",
        vod_path="/tmp/vod.mkv",
        include_ffmpeg_commands=True,
    )

    assert exported["json"].exists()
    assert exported["csv"].exists()
    assert exported["review"].exists()

    loaded = load_candidate_clips(exported["json"])
    assert [clip.clip_id for clip in loaded] == ["c_0001", "c_0002"]


def test_build_ffmpeg_clip_command_has_expected_output_path() -> None:
    cmd = build_ffmpeg_clip_command(vod_path="/tmp/source vod.mkv", clip=_sample_clips()[0], output_dir="snippets")

    assert "snippets/c_0001.mp4" in cmd
    assert "ffmpeg -ss" in cmd
