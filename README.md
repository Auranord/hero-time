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
python -m src.cli plan
```

## Configuration

Default runtime configuration: `configs/default.yaml`.

Prompt template for local reranking: `prompts/rerank_prompt.txt`.
