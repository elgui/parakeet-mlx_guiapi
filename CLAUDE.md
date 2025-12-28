# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parakeet-MLX GUI and API is a web interface and REST API wrapper for [parakeet-mlx](https://github.com/senstella/parakeet-mlx), which implements Nvidia's ASR models for Apple Silicon using MLX.

## Setup

The `parakeet-mlx` library is installed via pip (see `requirements.txt`). No sibling repo clone required.

## Commands

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start server (Flask API on port 5000, Gradio UI on port 5001)
python run.py

# With options
python run.py --host 127.0.0.1 --port 8000 --debug --model <model_name>

# CLI client for file transcription
python client.py audio.mp3 --output-format json
python client.py audio.mp3 --output-format srt --chunk-duration 120
python client.py audio.mp3 --segment 10-20 --output-file segment.wav

# Microphone recording with clipboard output
python client.py --mic --clipboard       # Record, transcribe, copy to clipboard
python client.py --mic                    # Record, transcribe, print to stdout
python client.py --mic --output-file out.txt  # Record and save to file
```

## Architecture

```
parakeet_mlx_guiapi/
├── api/routes.py          # Flask API endpoints (/api/transcribe, /api/segment, /api/models)
├── transcription/transcriber.py  # Core AudioTranscriber class wrapping parakeet_mlx
├── audio/processor.py     # Audio preprocessing utilities (resampling, mono conversion)
├── microphone/recorder.py # Live microphone recording (16kHz mono WAV)
├── ui/                    # Gradio web interface components
└── utils/
    ├── config.py          # Configuration management (env vars, config file)
    └── visualization.py   # Timeline and heatmap generation for transcripts
```

**Key flow:**
- `run.py` → initializes Flask app (`app.py`) and Gradio demo
- `app.py` → sets up Flask routes and creates Gradio interface
- API requests go through `api/routes.py` → `transcription/transcriber.py` → `parakeet_mlx.from_pretrained()`

**Entry points:**
- `run.py` / `parakeet-server` - Server startup
- `client.py` / `parakeet-client` - CLI client for API

## API Endpoints

- `POST /api/transcribe` - Transcribe audio file (supports json, txt, srt, vtt, csv output)
- `POST /api/segment` - Extract audio segment by time range
- `GET /api/models` - List available models

## Dependencies

Requires macOS with Apple Silicon (M1/M2/M3/M4) and ffmpeg installed (`brew install ffmpeg`).

Key libraries: parakeet-mlx, Flask, Gradio, pydub, pandas, matplotlib, sounddevice, pyperclip, rumps
