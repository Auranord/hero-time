# Twitch VOD Highlight Finder (MVP Plan)

This document defines a practical MVP for automatically finding **1–3 minute highlight candidates** in long Twitch VODs (3–7+ hours), with a strong focus on:

- German language streams
- Multi-person voice setups (mic + Discord)
- Emotional/hype/funny moments (not transcript-only)
- Local-first AI, including local LLM usage in MVP

---

## 1) MVP Goals and Non-Goals

### Goals
- Input OBS `.mkv` VODs with multiple audio tracks.
- Output ranked clip candidates with:
  - `start_time`
  - `duration_seconds`
  - `score`
  - `reason_tags`
- Detect likely:
  - funny moments
  - hype reactions
  - gameplay highlights
- Include a local LLM in the pipeline for semantic reranking and explanation.

### Non-Goals (for MVP)
- Fully automated publishing to socials.
- Perfect game-specific event detection for every title.
- Real-time processing during live stream.

---

## 2) Recommended Stack

- **Language:** Python 3.11+
- **Environment:** `uv` (or conda) + virtual environment
- **Media handling:** FFmpeg / ffprobe
- **Data format:** JSON + Parquet
- **Core libs:**
  - `faster-whisper` (German ASR + timestamps)
  - `pyannote.audio` (speaker diarization)
  - `librosa` / `torchaudio` (audio features)
  - `opencv-python` (video motion/scene features)
  - `pandas`, `numpy`, `scikit-learn` (fusion + ranking)
- **Local LLM serving:**
  - **Ollama** (easy local model hosting)
  - optional alternative: `llama.cpp`

---

## 3) Hardware-Aware Design (16 GB VRAM, ~25 GB RAM)

- Prefer **sequential stages** over full parallel execution.
- Use chunked processing (e.g., 5–15 minute chunks).
- Cache intermediate artifacts to avoid re-running heavy models.
- Keep one heavy GPU model active at a time in MVP.

---

## 4) High-Level Pipeline

1. **Ingest**
   - Probe media metadata.
   - Extract relevant audio tracks (mic, discord, mix).
   - Optionally produce low-fps video proxy for CV analysis.

2. **Feature Extraction**
   - ASR transcript with timestamps (German).
   - Speaker diarization (who spoke when + overlap).
   - Audio event/emotion proxies:
     - loudness spikes
     - laughter-like segments
     - overlapping speech
     - spectral/tempo changes
   - Video intensity proxies:
     - motion peaks
     - scene-change frequency

3. **Candidate Generation**
   - Sliding windows (e.g., 20–40 sec with overlap).
   - Compute initial highlight score from multimodal features.
   - Select top windows and merge neighbors.

4. **LLM Reranking (Local)**
   - Send candidate summaries (not full raw stream) to local LLM.
   - Ask model to score for clip-worthiness and tag reason class
     (`funny`, `hype`, `gameplay`, `wholesome`, etc.).
   - Combine heuristic score + LLM semantic score.

5. **Clip Proposal Output**
   - Expand candidates to 60–180 seconds.
   - Align boundaries to sentence/speaker turns.
   - Export JSON/CSV with confidence + reasons.

---

## 5) Local LLM Integration in MVP (Required)

### Why use an LLM already in MVP?
Classic features detect intensity, but LLM helps with **semantic quality**:
- Is this moment understandable out of context?
- Is there a setup → payoff arc?
- Is the moment emotionally interesting vs random noise?

### Local deployment option
Use Ollama with a compact instruct model (e.g., 7B–14B class, quantized).

### LLM input format (per candidate)
Provide compact structured context, for example:
- transcript slice (timestamped)
- active speaker count and overlap stats
- audio event summary (loudness peak, laughter probability)
- motion/scene summary

### LLM tasks
- score `clip_worthiness` from 0–1
- classify primary type (`funny`, `hype`, `gameplay`, `story`, `cozy`)
- generate short human-readable reason
- optionally suggest +/- boundary adjustment in seconds

### Guardrails
- Never let LLM be the only scoring source.
- Use deterministic schema output (JSON schema validation).
- Keep prompts short and chunked to control memory/latency.

---

## 6) Detection Methods to Include in MVP

### A) Audio-driven emotion and reaction cues
- RMS/peak loudness deltas
- sudden pitch/energy changes
- overlapping speech ratio (excited group reaction)
- laughter/non-speech event detector (if available)

### B) Conversation dynamics
- turn-taking acceleration
- interruptions
- speech-rate bursts

### C) Transcript-level semantic cues
- surprise/excitement expressions in German
- short punchline-like exchanges
- win/lose/emotion phrases

### D) Video/gameplay intensity cues
- motion intensity peaks
- rapid scene/camera shifts

### E) Multimodal fusion
- weighted blend of A-D + LLM semantic rerank
- temporal smoothing + cooldown to avoid duplicate clips

---

## 7) Suggested Repo Structure

```text
vod-highlights/
  pyproject.toml
  README.md
  configs/
    default.yaml
  src/
    ingest/
      probe.py
      extract_audio.py
    features/
      asr.py
      diarization.py
      audio_events.py
      video_motion.py
    scoring/
      heuristic_score.py
      llm_rerank.py
      fuse.py
    propose/
      segment_builder.py
      exporter.py
    cli.py
  prompts/
    rerank_prompt.txt
  data/
    cache/
    outputs/
```

---

## 8) MVP Milestones

### Milestone 1: Baseline heuristic detector
- Ingest + ASR + diarization + loudness/motion scoring
- Generate top candidate windows
- Export timestamps and durations

### Milestone 2: Local LLM reranker (must-have)
- Add Ollama connector
- Add schema-constrained reranking output
- Blend heuristic and LLM scores

### Milestone 3: Quality and usability pass
- Improve clip boundary alignment
- Add reason tags + confidence reporting
- Add simple review script (play candidate snippets)

---

## 9) Output Contract (MVP)

Example JSON object:

```json
{
  "vod_id": "2026-02-16_stream01",
  "clip_id": "c_0042",
  "start_time": "01:42:18",
  "duration_seconds": 96,
  "score": 0.87,
  "reason_tags": ["hype", "overlap_speech", "motion_peak"],
  "llm_summary": "Team reaction after unexpected win; clear emotional spike and payoff."
}
```

---

## 10) Immediate Next Step

Implement Milestone 1 first, but design interfaces so Milestone 2 (local LLM reranking) plugs in without refactor:
- define stable candidate schema now
- persist intermediate features
- add scorer registry (`heuristic`, `llm`, `hybrid`)

This keeps the MVP practical while ensuring LLM integration is present early and useful.
