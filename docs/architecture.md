# Project Architecture Plan

## Purpose
Build an offline-first MVP pipeline that finds 1-3 minute highlight candidates from long Twitch VODs.

## Core principles
- Sequential, cache-heavy execution to fit 16 GB VRAM environments.
- Stable contracts between stages to avoid milestone refactors.
- Hybrid scoring where local LLM augments (but never replaces) deterministic features.

## Bounded contexts

### 1. Ingest domain
- Responsibilities: metadata probe, track selection, audio extraction.
- Interfaces: returns normalized media metadata and extracted track paths.

### 2. Feature domain
- Responsibilities: compute ASR, speaker, audio reaction, and motion features.
- Interfaces: outputs window-aligned feature frames persisted to cache.

### 3. Scoring domain
- Responsibilities: heuristic ranking, LLM reranking, score fusion.
- Interfaces: consumes feature windows and returns scored windows + reason tags.

### 4. Proposal domain
- Responsibilities: temporal merging, boundary alignment, serialization.
- Interfaces: emits stable clip schema for downstream review/publishing.

## Data model contracts
- `FeatureWindow`: start/end + feature map.
- `CandidateClip`: clip ID, time, duration, score, tags, optional LLM summary.

## Stage execution flow
1. Probe + extract tracks.
2. Build feature windows in chunks.
3. Score windows with heuristic model.
4. Rerank top-N with local LLM.
5. Fuse and emit final candidate clips.

## Extension path (post-MVP)
- Game-specific event detectors per title.
- Lightweight UI for manual review.
- Social export adapters.
