# Roadmap

What is shipped and what is still open. Architecture lives in `CLAUDE.md`, the API
contract in `API.md`, and setup in `README.md` — this file is only the roadmap.

_Last reconciled against the code: 2026-08-17._

## Shipped

- **Core transcription** — `AudioTranscriber` over `parakeet_mlx`, with chunking and
  overlap for long audio.
- **Provider abstraction** — `providers/base.py` defines `STTProvider` /
  `TranscriptionResult`; three implementations: `parakeet` (local MLX), `deepgram`
  (Nova-2/Nova-3 cloud), `openai_audio` (any OpenAI-compatible `/v1/audio/transcriptions`,
  including the companion Local Model Server).
- **Per-request provider switching** — `/api/transcribe` accepts `provider`, `model`,
  `enable_diarization`, and `deepgram_options` as form fields, cached per resolved config.
- **Speaker diarization** — pyannote.audio, with a local fallback when cloud diarization
  collapses every segment onto one speaker.
- **Cross-chunk speaker tracking** — SpeechBrain ECAPA-VoxCeleb embeddings, cosine
  matching at a 0.45 similarity threshold, running average per global speaker.
- **Live / real-time transcription** — WebSocket at `/ws/live-transcribe` plus the `/live`
  UI, with streaming speaker labels and txt/srt export.
- **Multiple model support** — 7 Parakeet models plus 12 Deepgram models, selectable from
  the menu bar or per request.
- **REST API** — `/api/transcribe`, `/api/segment`, `/api/models`; json/txt/srt/vtt/csv.
- **Gradio UI** — file transcription on port+1, with timeline and heatmap visualizations.
- **CLI client** — file transcription, microphone capture, segment extraction, clipboard.
- **macOS menu bar app** — thin HTTP client of the daemon; provider/model switching,
  history, daemon health indicator, `launchctl` lifecycle control.
- **Launchd daemon** — `com.gui.parakeet` owns the single model load; menu bar and any
  other client talk to it over HTTP.
- **Unit tests** — `tests/test_transcription.py`, `tests/test_diarization.py`,
  `tests/test_menubar_recording.py`.
- **CI** — `.github/workflows/ci.yml`: pytest on macOS + Ubuntu across Python 3.10–3.12,
  plus Ruff lint and an informational mypy pass.
- **Documentation** — `README.md`, `API.md`, `CLAUDE.md`, this file.

## Open

- [ ] **Batch processing** — transcribe a directory of files in one call.
- [ ] **Result caching** — skip re-transcribing an unchanged file (hash the audio).
- [ ] **API authentication** — the daemon binds `0.0.0.0:8080` with no auth. Either bind
      `127.0.0.1` in the plist or add a token; see the security note in `API.md`.
- [ ] **Landing-page port bug** — `app.py:40` hardcodes `http://localhost:5001` for the
      Gradio iframe, but `run.py` puts Gradio on `port + 1` (8081 in practice). The `/`
      landing page therefore iframes a dead port; go to `:8081` directly.
- [ ] **`/api/models` is single-entry** — it returns only the configured model
      (`routes.py:333`), not the catalog the menu bar knows about.
- [ ] **Diarization on `openai_audio`** — unsupported; the provider returns one segment
      with no timing.
- [ ] **Examples directory** — no worked end-to-end examples beyond the test scripts.

## Not planned

- **Docker / Linux support.** MLX is Metal-backed and has no Linux build, so the local
  transcription path is Apple-Silicon-only. The old `Dockerfile` and `docker-compose.yml`
  were removed on 2026-08-17 — they had invalid `COPY` paths and could never have built.
  Note the nuance if anyone revisits this: the `deepgram` and `openai_audio` providers are
  pure network calls and don't need MLX, so a container serving only the cloud providers is
  technically viable. That would be a product decision, not a fix — don't re-add a
  Dockerfile that implies the local path works in it.
