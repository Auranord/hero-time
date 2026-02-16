# Manjaro Linux Setup Guide

This project is tested as an offline-first Python pipeline with FFmpeg tooling and optional local LLM support.

## 1) Install system packages

```bash
sudo pacman -Syu
sudo pacman -S --needed python python-pip python-virtualenv ffmpeg base-devel
```

Optional but useful for compiling dependencies and debugging media metadata:

```bash
sudo pacman -S --needed git jq
```

## 2) Clone and create Python environment

```bash
git clone <your-repo-url> hero-time
cd hero-time
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For development tools:

```bash
pip install -e .[dev]
```

## 3) Optional: Ollama for local reranking

Install Ollama (AUR package name may vary by helper):

```bash
yay -S ollama
sudo systemctl enable --now ollama
ollama pull qwen2.5:7b-instruct-q4_K_M
```

Verify:

```bash
curl http://localhost:11434/api/tags
```

## 4) Optional: pyannote token

`pyannote.audio` speaker diarization may require a Hugging Face token with accepted model terms.

```bash
export HF_TOKEN="<your_hf_token>"
```

Then pass it to the CLI command:

```bash
python -m src.cli features diarize /path/to/audio.wav --hf-auth-token "$HF_TOKEN"
```

## 5) Quick smoke run

Single-command pipeline (recommended):

```bash
python -m src.cli run /path/to/vod.mkv
```

This uses project defaults from `configs/default.yaml` for cache/output paths.

Expanded staged commands (debug-friendly):

```bash
python -m src.cli config show
python -m src.cli ingest probe /path/to/vod.mkv
python -m src.cli ingest extract-audio /path/to/vod.mkv
python -m src.cli features asr /path/to/audio.wav
python -m src.cli features diarize /path/to/audio.wav --transcript-path /path/to/transcript.json --hf-auth-token "$HF_TOKEN"
python -m src.cli features audio-events /path/to/audio.wav
python -m src.cli features video-motion /path/to/vod.mkv
python -m src.cli propose build vod_001 \
  --asr-path /path/to/transcript.json \
  --diarization-path /path/to/diarization.json \
  --video-motion-path /path/to/video_motion.json \
  --audio-events-path /path/to/audio_events.json
```

## 6) Troubleshooting

- `ffprobe: command not found`: install `ffmpeg` package and restart shell.
- `ModuleNotFoundError`: ensure `.venv` is activated and `pip install -e .` was run.
- Diarization auth errors: verify HF token and accepted model terms.
- LLM rerank unavailable: either run Ollama or use `--rerank-top-n 0` for deterministic-only scoring.
