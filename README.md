# Parakeet-MLX GUI and API 🦜✨

A comprehensive GUI and REST API for [parakeet-mlx](https://github.com/senstella/parakeet-mlx), Nvidia's ASR (Automatic Speech Recognition) models optimized for Apple Silicon using MLX.

## Features 🚀

- Web GUI for easy transcription of audio files (Gradio interface) 🌐
- REST API endpoints for seamless integration with other applications 🔌
- Support for multiple output formats (TXT, SRT, VTT, JSON, CSV) 📄
- Word-level timestamp highlighting in subtitle formats ✨
- Chunking of long audio files for better memory management 🧠
- Visualization of transcription results with timeline and heatmap 📊
- Audio segment extraction and playback 🎧
- **Live microphone recording** with direct transcription 🎤
- **Live streaming transcription** over WebSocket at `/live`, with real-time speaker labels ⚡
- **Speaker diarization** - identify who said what in multi-speaker audio 🗣️
- **Clipboard integration** for quick copy of transcription results 📋
- **Menu bar app** for one-click voice-to-clipboard with model switching and history (macOS) 🖥️
- **Always-on launchd daemon** that owns the model, so the menu bar app starts instantly 🔁
- **Multiple transcription providers** — Parakeet-MLX (local), Deepgram (cloud), and `openai_audio` for any OpenAI-compatible endpoint, including the companion Local Model Server project (a separate repo) that transcribes via a multimodal LLM (gemma-4-12b-qat through llama.cpp) 🧠
- **25 languages supported** including English, French, Spanish, German, and more 🌍
- Comprehensive CLI client with pip-installable commands 💻

> **Integrating another app?** `API.md` is the full local API contract — every endpoint,
> every form field, response shapes, and error strings.

## Prerequisites ✅

- macOS with Apple Silicon (M1/M2/M3/M4) 🍎
- Python 3.8 or higher 🐍
- ffmpeg installed 🛠️

**Note:** This project is optimized for Apple Silicon. All ML inference runs locally:
- **Transcription:** Uses MLX (Apple's ML framework) - GPU accelerated
- **Diarization:** Uses PyTorch CPU for stability - no CUDA needed

`parakeet-mlx` itself is a normal pip dependency (pinned in `requirements.txt`) — you do
**not** need to clone it. If a sibling checkout exists at `../parakeet-mlx`, `run.py` and
`app.py` prepend it to `sys.path` so you can develop against a local copy of the library;
that path is optional and skipped when absent.

## Quick Start 🚀

> **Architecture note**: The menu bar app is a **thin HTTP client** of a long-running server. Both options below start that server — Option A also installs the menu bar UI on top. The server (daemon) owns all inference; the menu bar only captures audio and POSTs it to `/api/transcribe`.

### Option A: Menu Bar App (Recommended for Daily Use)

```bash
# 1. Install ffmpeg (if not already installed)
brew install ffmpeg

# 2. Clone and enter the repository
git clone https://github.com/elgui/parakeet-mlx_guiapi.git
cd parakeet-mlx_guiapi

# 3. Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 4. Install the menu bar app
./install_menubar_app.sh
```

This installs **Parakeet.app** to `/Applications`. Launch it from:
- **Spotlight**: Cmd+Space, type "Parakeet"
- **Menu Bar**: Click the 🎤 icon
- **Terminal**: `open /Applications/Parakeet.app`

### Option B: Web GUI + API Server

```bash
# After steps 1-3 above, start the server:
python run.py
```

- **Web GUI (Gradio)**: http://localhost:8081
- **Live transcription**: http://localhost:8080/live
- **REST API**: http://localhost:8080/api/

The first run will download the model (~1.2GB).

### Option C: Always-On Daemon (what the menu bar app expects)

Instead of running `run.py` in a terminal, install it as a launchd user agent so it
starts at login and restarts on crash:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gui.parakeet.plist  # start
launchctl bootout    gui/$(id -u) ~/Library/LaunchAgents/com.gui.parakeet.plist  # stop
launchctl kickstart -k gui/$(id -u)/com.gui.parakeet                             # restart
launchctl print      gui/$(id -u)/com.gui.parakeet                               # status
```

The plist must set these environment variables, or things break in non-obvious ways:

| Var | Value | Why |
|-----|-------|-----|
| `PATH` | `/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin` | launchd's default `PATH` has no Homebrew; pydub needs `ffprobe`/`ffmpeg` to decode mp3/m4a |
| `MPLBACKEND` | `Agg` | matplotlib's macOS GUI backend crashes on worker threads when rendering the visualization PNGs |
| `HF_HOME` | your HuggingFace cache root | Stops the daemon re-downloading models |
| `HUGGINGFACE_HUB_CACHE` | `$HF_HOME/hub` | Same cache path `parakeet_mlx.from_pretrained` reads |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Pyannote diarization falls back to CPU where an MPS op is missing |

Logs land in `stdout.log` / `stderr.log` at the repo root (both gitignored).

## Usage ▶️

### Starting the Server 🚀

Run the server with:

```bash
python run.py
```

Or with custom options:

```bash
python run.py --host 127.0.0.1 --port 8000 --debug --model <model_name>
```

Server options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port for the Flask API (default: 8080)
- `--debug`: Enable debug mode
- `--model`: Specify the ASR model to use

The server will start and be accessible at:
- Gradio Web GUI: http://localhost:8081 (port + 1) 🌐
- REST API: http://localhost:8080/api/ 🔌

Note: The Gradio UI runs on port+1 from the specified port (default: 8081).

### Web GUI 🖥️

1. Open your browser and navigate to http://localhost:8081 (Gradio interface)
2. Upload an audio file using the interface ⬆️
3. Configure transcription options:
   - Output Format: json, txt, srt, or vtt 📄
   - Highlight Words: Enable word-level timestamps in SRT/VTT ✨
   - Chunk Duration: Duration in seconds for chunking long audio (0 to disable) ⏱️
   - Overlap Duration: Overlap duration in seconds when using chunking 🔄
4. Click "Transcribe" and view the results 🎉

### REST API 🤖

The following API endpoints are available:

#### Transcribe Audio 🎤➡️📄

```
POST /api/transcribe
```

Parameters (multipart/form-data). Only `file` is required — everything else falls back to
`~/.parakeet_mlx_guiapi.json`:

| Field | Default | Notes |
|-------|---------|-------|
| `file` | — | **Required.** Audio/video file ⬆️ |
| `output_format` | `json` | `json` \| `txt` \| `srt` \| `vtt` \| `csv` 📄 |
| `highlight_words` | `false` | Word-level timestamps in SRT/VTT ✨ |
| `chunk_duration` | `120` | Seconds; `0` disables chunking ⏱️ |
| `overlap_duration` | `15` | Seconds of overlap between chunks (parakeet path) 🔄 |
| `provider` | config | `parakeet` \| `deepgram` \| `openai_audio` 🔀 |
| `model` | config | Override the model for the chosen provider 🧠 |
| `enable_diarization` | config | `true` \| `false`; not supported by `openai_audio` 🗣️ |
| `deepgram_options` | — | JSON string; only meaningful with `provider=deepgram` ⚙️ |

Response:
- For JSON format: JSON object with transcription results and visualizations 📊
- For other formats: File download with the appropriate content type ⬇️

> **Gotcha:** the `json` response embeds two base64 PNGs (`visualization`, `heatmap`) that
> bloat the payload. For app integration prefer `output_format=txt`, or parse the JSON and
> drop those two fields. See `API.md`.

Example cURL request:
```bash
curl -X POST -F "file=@audio.mp3" -F "output_format=json" http://localhost:8080/api/transcribe

# Force a specific provider and model for one request
curl -X POST -F "file=@audio.mp3" -F "output_format=txt" \
  -F "provider=deepgram" -F "model=nova-3" \
  -F "enable_diarization=true" \
  -F 'deepgram_options={"smart_format":true,"punctuate":true}' \
  http://localhost:8080/api/transcribe
```

#### Get Audio Segment ✂️🎧

```
POST /api/segment
```

Parameters (multipart/form-data):
- `file`: The audio file to extract segment from (required) ⬆️
- `start_time`: Start time in seconds (required) ⏱️
- `end_time`: End time in seconds (required) ⏱️

Response:
- WAV audio file containing the requested segment 🎧

Example cURL request:
```bash
curl -X POST -F "file=@audio.mp3" -F "start_time=10" -F "end_time=20" http://localhost:8080/api/segment -o segment.wav
```

#### Get Available Models 🧠

```
GET /api/models
```

Response:
- JSON array of available model names 📜

Example cURL request:
```bash
curl http://localhost:8080/api/models
```

### Live Transcription (WebSocket) ⚡

Open **http://localhost:8080/live** for the streaming UI, or drive the socket directly at
`ws://localhost:8080/ws/live-transcribe`.

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

Speakers are tracked **across** chunks using SpeechBrain ECAPA-VoxCeleb embeddings, so
`SPEAKER_00` in chunk 1 stays `SPEAKER_00` in chunk 9. When cloud diarization fails on a
short chunk, pyannote takes over locally. `test_streaming_injection.py` is a working
client that replays `static/test/2ppl-FR.mp3` through the socket.

### CLI Client 💻

After installation, you can use the CLI client for file transcription and microphone recording:

#### File Transcription
```bash
# Basic transcription (outputs JSON)
python client.py audio.mp3

# Specify output format
python client.py audio.mp3 --output-format srt

# With chunking for long audio files
python client.py audio.mp3 --output-format json --chunk-duration 120

# Extract a specific segment
python client.py audio.mp3 --segment 10-20 --output-file segment.wav

# Generate visualization
python client.py audio.mp3 --output-format json --visualize
```

#### Microphone Recording 🎤
```bash
# Record from microphone and transcribe (press Enter to stop)
python client.py --mic

# Record and copy transcription to clipboard
python client.py --mic --clipboard

# Record and save to file
python client.py --mic --output-file transcription.txt
```

#### CLI Options Reference

| Option | Description |
|--------|-------------|
| `--mic` | Record from microphone instead of using a file |
| `--clipboard` | Copy transcription result to clipboard |
| `--api-url` | Base URL for the API (default: http://localhost:8080/api) |
| `--output-format` | Output format: json, txt, srt, vtt, csv (default: json) |
| `--highlight-words` | Enable word-level timestamps in SRT/VTT |
| `--chunk-duration` | Chunking duration in seconds (default: 120, 0 to disable) |
| `--overlap-duration` | Overlap duration in seconds (default: 15) |
| `--output-file` | Output file path |
| `--segment` | Extract segment (format: start_time-end_time) |
| `--visualize` | Generate visualization (JSON output only) |

#### Installable Commands

After installing with pip (`pip install -e .`), these commands are available **when the virtual environment is activated**:

```bash
source .venv/bin/activate  # Activate first!
parakeet-server   # Start the web GUI + API server
parakeet-client   # Run the CLI client
parakeet-menubar  # Launch the menu bar app (dev mode)
```

**Tip:** For daily use without activating venv, install the Parakeet.app instead (see below).

### Menu Bar App (Voice-to-Clipboard) 📋

A macOS menu bar app for quick, seamless voice transcription. Click to record, click to stop - transcription is automatically copied to your clipboard.

#### One-Line Install (Recommended) 🚀

```bash
./install_menubar_app.sh
```

This script will:
1. Build `Parakeet.app` (alias mode - fast build)
2. Install it to `/Applications`
3. Optionally add it to Login Items (start at boot)
4. Launch the app

**First Launch:**
- The menu bar app itself launches instantly — no model loading in this process
- It probes the daemon (`GET /api/models` on `localhost:8080`); if the daemon is down, the status shows `Daemon: ○ offline` and offers `Server > Start`
- On first transcription request, the daemon loads the model (~10-15s) from the HuggingFace cache at `/Volumes/models/huggingface/hub/` (or wherever `HF_HOME` points)
- If the model isn't cached, the daemon will fetch it on first request (~1.2GB for the default TDT 0.6B v3 Multilingual)

After installation, find **Parakeet** in:
- Your **menu bar** (🎤 icon in the top-right)
- **Spotlight** (Cmd+Space, type "Parakeet")
- **Applications** folder

#### How It Works
1. A microphone icon (🎤) appears in your macOS menu bar
2. **Click** the icon to start recording (icon shows 🔴 with timer)
3. **Click again** to stop recording
4. The app transcribes your audio and **automatically copies to clipboard**
5. A notification shows a preview of the transcription

#### Menu Bar Features

| Feature | Description |
|---------|-------------|
| **Provider Switching** | Parakeet (local), Deepgram (cloud), or `openai_audio`; for `openai_audio` the app discovers models from the gateway and offers **Set Server URL…** |
| **Model Selection** | Switch between models organized by category; selection is sent per-request to the daemon |
| **Daemon Health** | Status item shows `Daemon: ● ready` or `Daemon: ○ offline`; polled every 30s |
| **Recording Timer** | See elapsed time while recording (🔴 0:15) |
| **Transcription History** | Access last 10 transcriptions, click to copy again |
| **Speaker Diarization** | Toggle on/off; the daemon handles model availability and inference |
| **Server submenu** | `Start` / `Stop` / `Restart` call `launchctl bootstrap` / `bootout` / `kickstart` against `com.gui.parakeet` |
| **Settings** | Configure chunk duration, auto-copy, notifications |
| **Status Display** | Daemon liveness + current provider |

#### Available Models

| Model | Languages | WER | Speed | Size | Best For |
|-------|-----------|-----|-------|------|----------|
| ⭐ **TDT 0.6B v3 Multilingual** | EN, FR, ES, DE + 21 more | 6.34% | Fast | ~1.2GB | **Recommended** - General use |
| TDT 0.6B v2 English | English | 6.5% | Fast | ~1.2GB | English-only, accurate |
| TDT 1.1B English | English | ~5.5% | Slower | ~2.2GB | Meetings, interviews |
| CTC 0.6B English | English | ~7% | Fastest | ~1.2GB | Quick notes, real-time |
| CTC 1.1B English | English | ~6% | Very Fast | ~2.2GB | Long audio, speed priority |
| TDT+CTC 1.1B English | English | ~5.8% | Medium | ~2.2GB | Podcasts, 11hr support |
| TDT+CTC 110M Tiny | English | ~12% | Instant | ~220MB | Ultra-fast loading |

**Supported Languages (v3 Multilingual):**
🇬🇧 English, 🇫🇷 French, 🇪🇸 Spanish, 🇩🇪 German, 🇮🇹 Italian, 🇵🇹 Portuguese, 🇳🇱 Dutch, 🇵🇱 Polish, 🇷🇺 Russian, 🇺🇦 Ukrainian, 🇨🇿 Czech, 🇸🇰 Slovak, 🇧🇬 Bulgarian, 🇭🇷 Croatian, 🇩🇰 Danish, 🇪🇪 Estonian, 🇫🇮 Finnish, 🇬🇷 Greek, 🇭🇺 Hungarian, 🇱🇻 Latvian, 🇱🇹 Lithuanian, 🇲🇹 Maltese, 🇷🇴 Romanian, 🇸🇮 Slovenian, 🇸🇪 Swedish

To change models: Click menu bar icon → **Model** → Select category → Select model

#### Speaker Diarization (Who Said What) 🗣️

The app supports **speaker diarization** - identifying WHO is speaking in multi-speaker recordings. Works with any transcription model.

**Example output with diarization enabled:**
```
SPEAKER_00: Hello, how are you today?

SPEAKER_01: I'm doing great, thanks for asking! How about you?

SPEAKER_00: Pretty good. Let me tell you about our project...
```

**In-App Setup (Recommended):**

The app includes a guided setup wizard:

1. Go to **Settings → Speaker Diarization → Quick Setup**
2. The wizard will:
   - Install pyannote.audio if needed (progress shown in Terminal)
   - Open HuggingFace to accept model license and create token
   - Let you paste the token directly in the app (saved to config)
3. Enable diarization when setup completes

**What you need:**
- A free [HuggingFace account](https://huggingface.co)
- Accept the [pyannote model license](https://huggingface.co/pyannote/speaker-diarization-3.1)
- A **Read** access token (not Write) - the wizard guides you through this

**Apple Silicon Compatibility:**
- Uses PyTorch CPU for maximum stability (no CUDA needed)
- Diarization runs entirely locally after setup
- First use downloads ~1GB model (progress shown)
- Adds ~10-30s processing time depending on audio length

**Manual Setup (Alternative):**

If you prefer command-line setup:
```bash
# Install pyannote
pip install pyannote.audio>=3.1.0

# Save token to config file
echo '{"huggingface_token": "hf_your_token_here"}' > ~/.parakeet_mlx_guiapi.json
```

#### Manual Installation

If you prefer to install manually:

```bash
# 1. Install thin-client deps (model libs live on the daemon, not the menu bar)
pip install py2app rumps pyobjc-framework-Cocoa pyperclip sounddevice requests scipy numpy

# 2. Build the app (alias mode for faster build)
python setup_app.py py2app --alias

# 3. Copy to Applications
cp -R dist/Parakeet.app /Applications/

# 4. Make sure the launchd daemon is loaded (the menu bar requires it)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gui.parakeet.plist

# 5. Launch
open /Applications/Parakeet.app
```

Note: The app is alias mode — it symlinks back to your `.venv`. Moving the `.venv` breaks `/Applications/Parakeet.app`.

#### Start at Login

To have Parakeet start automatically when you log in:

1. Open **System Settings** (or System Preferences on older macOS)
2. Go to **General → Login Items** (or Users & Groups → Login Items)
3. Click **+** and select `/Applications/Parakeet.app`

Or via Terminal:
```bash
osascript -e 'tell application "System Events" to make login item at end with properties {path:"/Applications/Parakeet.app", hidden:false}'
```

#### Running from Terminal (Development)

For development or testing without building the app:

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Option 1: Run directly
python menubar_app.py

# Option 2: Use the installed command (requires pip install -e .)
parakeet-menubar
```

**Note:** These commands require the virtual environment to be activated. For daily use, install the Parakeet.app which works without activation.

#### Troubleshooting

**"Parakeet" can't be opened because Apple cannot check it for malicious software:**
1. Open **System Settings → Privacy & Security**
2. Scroll down to find the message about Parakeet
3. Click **Open Anyway**

**Microphone permission:**
- The first time you record, macOS will ask for microphone permission
- If denied, go to **System Settings → Privacy & Security → Microphone** and enable Parakeet

## License 📜

No license has been chosen for this project yet, so default copyright applies. Note that
the upstream [parakeet-mlx](https://github.com/senstella/parakeet-mlx) library and the
Nvidia model weights carry their own licenses — check those before redistributing.

## Contributing 👋

Pull requests are welcome! Feel free to contribute bug fixes or new features. We appreciate your contributions! 🙏

## Acknowledgments 🙌

- This project uses [parakeet-mlx](https://github.com/senstella/parakeet-mlx) as its core library
- Thanks to [Nvidia](https://www.nvidia.com/) for training these powerful models
- Thanks to [MLX](https://github.com/ml-explore/mlx) for providing the incredible framework
- Special thanks to [Sam Witteveen](https://github.com/samwit) for his inspirational code and his insightful [YouTube channel](https://www.youtube.com/@samwitteveenai)
