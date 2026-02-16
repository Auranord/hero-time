# VOD Highlights (MVP Scaffold)

This repository is scaffolded from the `readme.md` MVP plan for a **local-first Twitch VOD highlight finder** focused on German streams and multi-speaker setups.

## What is included

- Project structure for all MVP pipeline stages.
- Stable data contracts for feature artifacts and clip proposals.
- Pluggable scorer architecture (`heuristic`, `llm`, `hybrid`).
- Local-LLM reranking interface designed for Ollama.

## Architecture (MVP)

The system is organized as a sequential offline pipeline:

1. **Ingest**
   - `probe.py`: metadata and stream probing (`ffprobe`).
   - `extract_audio.py`: extract relevant tracks from OBS `.mkv`.

2. **Feature extraction**
   - `asr.py`: German transcript segments with timestamps.
   - `diarization.py`: speaker turns + overlap metrics.
   - `audio_events.py`: loudness and reaction proxy events.
   - `video_motion.py`: low-cost motion / scene intensity proxies.

3. **Scoring and rerank**
   - `heuristic_score.py`: deterministic multimodal baseline.
   - `llm_rerank.py`: local Ollama scorer with strict JSON output.
   - `fuse.py`: strategy to blend heuristic + LLM scores.

4. **Proposal generation**
   - `segment_builder.py`: create and merge candidate windows.
   - `exporter.py`: JSON/CSV export to `data/outputs/`.

5. **CLI orchestration**
   - `cli.py`: single command entrypoint for staged processing.

## Milestone implementation strategy

- **Milestone 1**: populate ingest + baseline feature stages and heuristic ranking.
- **Milestone 2**: enable Ollama reranking by implementing `llm_rerank.py` against the existing scorer interface.
- **Milestone 3**: improve boundary alignment and add review tooling.

## Run (scaffold)

```bash
python -m src.cli config show
python -m src.cli ingest probe /path/to/vod.mkv
python -m src.cli ingest extract-audio /path/to/vod.mkv
python -m src.cli features asr /path/to/audio.wav
python -m src.cli features diarize /path/to/audio.wav --transcript-path /path/to/transcript.json
python -m src.cli features audio-events /path/to/audio.wav
python -m src.cli features video-motion /path/to/vod.mkv
python -m src.cli propose build vod_001 \
  --asr-path /path/to/transcript.json \
  --diarization-path /path/to/diarization.json \
  --video-motion-path /path/to/video_motion.json \
  --audio-events-path /path/to/audio_events.json
python -m src.cli propose review /path/to/candidates.json --vod-path /path/to/vod.mkv
```

## Prerequisites

- Python 3.11+
- `ffmpeg` and `ffprobe` available on `PATH`
- Python dependencies from `pyproject.toml`
- (Optional) Local Ollama runtime if you want LLM reranking
- (Optional) Hugging Face auth token for `pyannote.audio` speaker diarization model access

Install project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For local Ollama reranking, ensure Ollama is running and a compatible model is pulled (default: `qwen2.5:7b-instruct-q4_K_M`).

## Configuration

Default runtime configuration: `configs/default.yaml`.

Prompt template for local reranking: `prompts/rerank_prompt.txt`.


Environment overrides use the `VOD_HIGHLIGHTS_` prefix and nested keys via `__` (for example `VOD_HIGHLIGHTS_PIPELINE__CHUNK_MINUTES=5`).

Ingest artifacts are persisted under `data/cache/ingest/<vod_stem>/` (ffprobe raw JSON, normalized metadata, and extracted WAV tracks).

ASR artifacts are persisted under `data/cache/features/asr/<audio_stem>/` including chunk WAV files and `transcript.json` with timestamped segments and confidence fields.

Diarization artifacts are persisted under `data/cache/features/diarization/<audio_stem>/diarization.json` with speaker turns, transcript alignment, speaking-time totals, and overlap metrics per analysis window.

Audio event artifacts are persisted under `data/cache/features/audio_events/<audio_stem>/audio_events.json` with loudness spike events and per-window audio intensity features.

Video motion artifacts are persisted under `data/cache/features/video_motion/<vod_stem>/video_motion.json` with low-FPS motion peaks, scene-change detections, and per-window intensity features aligned to scoring windows.

Generated candidate exports are written to `data/outputs/` as JSON, CSV, and review manifest files.

## Manjaro setup guide

See `docs/manjaro_setup.md` for a distro-specific setup guide.
