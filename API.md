# Parakeet STT — Local API Contract

The STT server runs as a launchd user agent (`com.gui.parakeet`) and is **already
reachable by any app on this machine**. This document is the integration contract.

## Connection

| | |
|---|---|
| **Base URL** | `http://localhost:8080` (also `http://<LAN-IP>:8080` — see Security note) |
| **Bind** | `0.0.0.0:8080` (REST + WebSocket), `0.0.0.0:8081` (Gradio UI) |
| **CORS** | Enabled for all origins (`CORS(app)`) — browser apps can call it directly |
| **Auth** | None (local trust model) |
| **Model** | `mlx-community/parakeet-tdt-0.6b-v3` (Apple Silicon / MLX) |
| **Lifecycle** | Auto-starts at login (`RunAtLoad`), auto-restarts on crash (`KeepAlive`) |
| **Service file** | `~/Library/LaunchAgents/com.gui.parakeet.plist` → runs `~/dev/parakeet-mlx_guiapi/run.py` |

Restart/stop the service:
```bash
launchctl kickstart -k gui/$(id -u)/com.gui.parakeet   # restart
launchctl bootout gui/$(id -u)/com.gui.parakeet         # stop
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gui.parakeet.plist  # start
```

---

## `POST /api/transcribe` — transcribe an audio/video file

`multipart/form-data`. Only `file` is required.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `file` | file | — | **Required.** Audio/video file. |
| `output_format` | string | `json` | `json` \| `txt` \| `srt` \| `vtt` \| `csv` |
| `chunk_duration` | float (s) | `120` | Long-audio chunking. `0` disables chunking. |
| `overlap_duration` | float (s) | `15` | Overlap between chunks (parakeet path). |
| `highlight_words` | bool | `false` | `true`/`false`. |
| `provider` | string | config | `parakeet` (local), `deepgram` (cloud, needs key), or `openai_audio` (local LLM via the Local Model Server). Omit for default. |
| `model` | string | config | Override model name. |
| `enable_diarization` | bool | config | `true`/`false`. Speaker labels. (Not supported by `openai_audio`.) |
| `deepgram_options` | JSON string | — | Only when `provider=deepgram`. |

**Fastest integration — plain text:**
```bash
curl -s -X POST http://localhost:8080/api/transcribe \
  -F "file=@/path/to/audio.wav" \
  -F "output_format=txt"
# → raw transcript text (text/plain)
```

**JSON (default) response shape:**
```jsonc
{
  "text": "full transcript ...",
  "segments": [
    {
      "Start (s)": 0.0,
      "End (s)": 3.2,
      "Segment": "I have nothing.",
      "Duration": 3.2,
      "Tokens": [],
      "Speaker": "Speaker 1"   // present only when diarization produced speakers
    }
  ],
  "visualization": "<base64 PNG>",  // ⚠ large — ignore for programmatic use
  "heatmap": "<base64 PNG>"         // ⚠ large — ignore for programmatic use
}
```

> **Gotcha:** the default `json` format embeds two base64 PNGs (`visualization`,
> `heatmap`) that bloat the payload. For app integration prefer `output_format=txt`
> (raw string), `srt`/`vtt` (subtitle file download), or `csv`, or parse `json` and
> drop the image fields.

`srt` / `vtt` / `csv` return a **file download** (`Content-Disposition: attachment`),
not inline text — write the response body to a file.

**Errors** (`application/json`):
| Status | Body |
|--------|------|
| 400 | `{"error": "No file part"}` — `file` field missing |
| 400 | `{"error": "No selected file"}` — empty filename |
| 400 | `{"error": "provider must be one of ['deepgram', 'openai_audio', 'parakeet']"}` |
| 400 | `{"error": "enable_diarization must be 'true' or 'false'"}` |
| 400 | `{"error": "deepgram_options must be valid JSON"}` |
| 500 | `{"error": "<exception message>"}` |

---

## `GET /api/models` — list available model(s)

```bash
curl -s http://localhost:8080/api/models
# → ["mlx-community/parakeet-tdt-0.6b-v3"]
```

---

## `POST /api/segment` — extract an audio segment as WAV

`multipart/form-data`. Returns a `.wav` file download.

| Field | Type | Notes |
|-------|------|-------|
| `file` | file | **Required.** |
| `start_time` | float (s) | Default `0`. |
| `end_time` | float (s) | Must be `> start_time` else `400 {"error":"Invalid time range"}`. |

```bash
curl -s -X POST http://localhost:8080/api/segment \
  -F "file=@/path/to/audio.wav" -F "start_time=5" -F "end_time=12" \
  -o segment.wav
```

---

## Live (streaming) transcription — WebSocket

| | |
|---|---|
| **WS endpoint** | `ws://localhost:8080/ws/live-transcribe` |
| **Demo page** | `http://localhost:8080/live` |

Client streams base64-encoded audio chunks; server returns transcribed segments as
JSON messages. See `parakeet_mlx_guiapi/live/websocket_handler.py` and the working
example in `test_ws_live.py` for the exact message protocol.

---

## Other UIs

- **Gradio interface:** `http://localhost:8081`
- **Landing page:** `http://localhost:8080/`

---

## Security note (read before exposing beyond this Mac)

The service binds `0.0.0.0`, so it is reachable from **any device on your LAN**, not
just this machine, and there is **no authentication**. For a strictly local-only
posture, change the plist `--host` from `0.0.0.0` to `127.0.0.1` and reload the
agent. Leave it at `0.0.0.0` only if you intend other devices on the network to use it.
