# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parakeet-MLX GUI and API is a web interface and REST API wrapper for [parakeet-mlx](https://github.com/senstella/parakeet-mlx), which implements Nvidia's ASR models for Apple Silicon using MLX. Features include:

- **Multi-provider architecture**: Local (Parakeet-MLX) and cloud (Deepgram Nova-2/Nova-3) transcription
- **Live transcription** via WebSocket with real-time speaker diarization
- **Cross-chunk speaker tracking** using speaker embeddings
- **macOS Menu Bar App** for quick voice-to-clipboard transcription

## Setup

```bash
# Install dependencies (use .venv, not venv)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option A: Run the server in the foreground (dev)
python run.py --host 127.0.0.1 --port 8080 --debug --model <model_name>

# Option B: Run the server as an always-on launchd user agent (recommended)
# See "Launchd Daemon" section below — the menu bar app expects this.
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gui.parakeet.plist

# Menu bar app (thin HTTP client of the daemon — installer auto-creates .venv if missing)
./install_menubar_app.sh
```

## Architecture (thin client + daemon)

The system has two cooperating components:

- **Launchd daemon** (`run.py` under `com.gui.parakeet`, port 8080) — owns all inference. Loads the parakeet-mlx model into MLX/Metal once; serves Flask REST + Gradio UI + WebSocket live transcription.
- **Menu bar app** (`menubar_app.py`, py2app `Parakeet.app`) — stateless HTTP client of the daemon. Captures audio, POSTs WAV bytes to `/api/transcribe`, drops returned text on the clipboard. No local model loading. Reads `~/.parakeet_mlx_guiapi.json` for provider/model/options state, sends per-request.

Single inference engine, single model load. The menu bar's `Server` submenu controls the daemon via `launchctl bootstrap` / `bootout` / `kickstart` — never spawns a competing `run.py`.

## Architecture

```
parakeet_mlx_guiapi/
├── providers/                    # STT provider abstraction layer
│   ├── base.py                   # Abstract interfaces (STTProvider, TranscriptionResult)
│   ├── parakeet/provider.py      # Local MLX transcription
│   └── deepgram/provider.py      # Cloud transcription (Nova-2/Nova-3)
├── live/                         # Real-time streaming
│   ├── session.py                # LiveTranscriptionSession with speaker tracking
│   └── websocket_handler.py      # WebSocket endpoint /ws/live-transcribe
├── diarization/
│   └── diarizer.py               # Speaker diarization (pyannote.audio)
├── transcription/
│   └── transcriber.py            # Core AudioTranscriber (parakeet_mlx wrapper)
├── audio/processor.py            # Resampling, mono conversion, segmentation
├── microphone/recorder.py        # 16kHz mono WAV recording
├── api/routes.py                 # Flask REST endpoints
├── ui/                           # Gradio interface
└── utils/
    ├── config.py                 # Config (~/.parakeet_mlx_guiapi.json)
    └── visualization.py          # Timeline/heatmap generation

menubar_app.py                    # macOS menu bar application
templates/live_transcription.html # Live transcription web UI
```

## Provider System

### Provider Types
- **Parakeet (Local)**: MLX-accelerated on Apple Silicon, supports diarization via pyannote
- **Deepgram (Cloud)**: REST API, Nova-2/Nova-3 models, built-in diarization

### Deepgram Models (latest: Nova-3)
```python
# Nova-3 (latest, best accuracy)
"nova-3", "nova-3-meeting", "nova-3-phonecall", "nova-3-voicemail", "nova-3-finance", "nova-3-medical"

# Nova-2 (proven)
"nova-2", "nova-2-meeting", "nova-2-phonecall", "nova-2-voicemail", "nova-2-finance", "nova-2-medical"
```

### Deepgram Configurable Options
```python
{
    "smart_format": True,    # Auto-capitalize, format numbers
    "punctuate": True,       # Add punctuation
    "paragraphs": True,      # Group into paragraphs
    "utterances": True,      # Break by utterances (for diarization)
    "profanity_filter": False,
    "numerals": False,       # Convert "one" to "1"
}
```

### Parakeet Models
| Model | WER | Speed | Languages | Best For |
|-------|-----|-------|-----------|----------|
| `parakeet-tdt-0.6b-v3` | 6.34% | Fast | 25 languages | **Recommended - Multilingual** |
| `parakeet-tdt-1.1b` | ~5.5% | Slow | EN only | Best English accuracy |
| `parakeet-tdt_ctc-1.1b` | ~5.8% | Medium | EN only | Long audio (up to 11hr) |
| `parakeet-tdt_ctc-110m` | ~12% | Instant | EN only | Ultra lightweight (220MB) |

## Live Transcription

### WebSocket Endpoint
`ws://localhost:8080/ws/live-transcribe`

### Message Protocol
```javascript
// Client → Server
{type: "config", enable_diarization: true, provider: "deepgram", model: "nova-3"}
{type: "audio_chunk", data: "<base64 WAV>", chunk_start: 0.0}
{type: "export", format: "txt"|"srt"}
{type: "clear"}

// Server → Client
{type: "connected", session_id, provider, diarization_enabled, ...}
{type: "transcription", messages: [{speaker, text, start_time, end_time, color}, ...]}
{type: "status", message, debug}
{type: "export_result", content, filename}
```

### Speaker Diarization Features
- **Cross-chunk speaker tracking**: Uses SpeechBrain ECAPA-VoxCeleb embeddings
- **Local diarization fallback**: When Deepgram fails on short chunks, pyannote takes over
- **Speaker color assignment**: 8-color palette for visual distinction
- **Configurable similarity threshold**: Default 0.45 for speaker matching

### Key Files
- `session.py:_apply_cross_chunk_speaker_tracking()` - Matches speakers across chunks
- `session.py:_apply_local_diarization()` - Fallback when cloud fails
- `session.py:_needs_local_diarization()` - Detects failed cloud diarization

## REST API Endpoints

**Base URL**: `http://localhost:8080/api`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/transcribe` | POST | Transcribe audio file (json, txt, srt, vtt, csv). Optional per-request form fields: `provider` (`parakeet`\|`deepgram`), `model`, `deepgram_options` (JSON string), `enable_diarization` (`true`\|`false`). All optional; missing fields fall back to config in `~/.parakeet_mlx_guiapi.json`. |
| `/api/segment` | POST | Extract audio segment by time range |
| `/api/models` | GET | List available models |

## Web Interfaces

| URL | Purpose |
|-----|---------|
| `http://localhost:8080/live` | Live transcription with WebSocket |
| `http://localhost:8081/` | Gradio file transcription UI |

## Configuration

**Config file**: `~/.parakeet_mlx_guiapi.json`

```json
{
    "model_name": "mlx-community/parakeet-tdt-0.6b-v3",
    "stt_provider": "deepgram",
    "deepgram_model": "nova-3",
    "deepgram_api_key": "<key>",
    "deepgram_options": {"smart_format": true, "punctuate": true},
    "diarization_enabled": true,
    "huggingface_token": "<token>",
    "default_chunk_duration": 120
}
```

**Environment Variables**:
- `DEEPGRAM_API_KEY` - Deepgram API key
- `HUGGINGFACE_TOKEN` / `HF_TOKEN` - For pyannote diarization models

## CLI Client

```bash
# File transcription
python client.py audio.mp3 --output-format json
python client.py audio.mp3 --output-format srt --chunk-duration 120

# Microphone recording
python client.py --mic --clipboard    # Record → transcribe → clipboard
python client.py --mic                 # Record → transcribe → stdout

# Extract segment
python client.py audio.mp3 --segment 10-20 --output-file segment.wav
```

## Menu Bar App (`menubar_app.py`)

macOS menu bar application for voice-to-clipboard transcription. **Thin HTTP client** of the launchd daemon — no local inference, no `parakeet_mlx` import in this process. Launches in <1s; first transcription only takes daemon-load time.

**Features**:
- Provider switching (Parakeet/Deepgram) — sent per-request to daemon
- Model selection per provider — sent per-request to daemon
- Deepgram options toggle — sent per-request as JSON form field
- Parakeet options (chunk duration, language)
- Speaker diarization toggle — daemon handles diarization
- Daemon lifecycle control: `Server > Start/Stop/Restart` runs `launchctl bootstrap/bootout/kickstart` against `com.gui.parakeet`
- Daemon health indicator (●/○) — polled every 30s via `GET /api/models`
- Transcription history (last 10)

**Dependency**: the launchd daemon must be running on `localhost:8080`. If it's down, the menu bar shows `Daemon: ○ offline` and recording is disabled with a notification pointing to `Server > Start`.

## Launchd Daemon (`com.gui.parakeet`)

Always-on user agent serving the Flask REST + Gradio UI + WebSocket endpoints on port 8080.

**Plist**: `~/Library/LaunchAgents/com.gui.parakeet.plist`. Required `EnvironmentVariables`:

| Var | Value | Why |
|-----|-------|-----|
| `PATH` | `/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin` | launchd's default PATH excludes Homebrew; pydub needs `ffprobe`/`ffmpeg` to decode mp3/m4a/etc |
| `MPLBACKEND` | `Agg` | matplotlib defaults to MacOS GUI backend which crashes on worker threads when generating visualization PNGs |
| `HF_HOME` | `/Volumes/models/huggingface` | Pre-downloaded model cache; daemon never re-downloads if present |
| `HUGGINGFACE_HUB_CACHE` | `/Volumes/models/huggingface/hub` | Same cache path used by `parakeet_mlx.from_pretrained` |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Pyannote diarization on Apple Silicon — fall back to CPU when MPS op unsupported |

**Service control**:
```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gui.parakeet.plist  # start
launchctl bootout    gui/$(id -u) ~/Library/LaunchAgents/com.gui.parakeet.plist  # stop
launchctl kickstart -k gui/$(id -u)/com.gui.parakeet                              # restart
launchctl print     gui/$(id -u)/com.gui.parakeet                                 # status
```

**Logs**: `stdout.log` and `stderr.log` at the repo root (paths set by plist `StandardOutPath` / `StandardErrorPath`). Both gitignored.

## Testing

```bash
# Test streaming injection (simulates browser WebSocket)
python test_streaming_injection.py

# Uses: static/test/2ppl-FR.mp3 (9.8s, 2 speakers)
# Streams in 8s chunks with 500ms delay
```

## Dependencies

**Required**: macOS with Apple Silicon (M1/M2/M3/M4), ffmpeg (`brew install ffmpeg`)

**Key packages**:
- `parakeet-mlx` - Core ASR model
- `pyannote.audio >= 3.1.0` - Speaker diarization (requires HF token)
- `speechbrain` - Speaker embeddings for cross-chunk tracking
- `flask`, `flask-sock` - REST API and WebSocket
- `gradio` - File transcription UI
- `rumps` - macOS menu bar
- `sounddevice` - Microphone recording

## Key Implementation Details

### Cross-Chunk Speaker Tracking
1. Extract speaker embedding from audio segment using SpeechBrain
2. Compare with known speaker embeddings (cosine similarity)
3. Match to existing speaker if similarity > threshold (0.45)
4. Create new global speaker ID if no match
5. Update running average of speaker embedding

### Local Diarization Fallback (for Deepgram)
1. Check if all segments have same speaker (cloud diarization failed)
2. Run pyannote diarization on the audio chunk
3. Merge transcription text with diarization speaker labels
4. Continue with cross-chunk tracking

### Text Cleaning
- Removes `<unk>` tokens from Parakeet output
- Normalizes whitespace
- Skips empty segments
