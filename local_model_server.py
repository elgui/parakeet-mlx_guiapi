"""
Local Model Server — a self-owned, OpenAI-compatible gateway for local LLMs.

WHY THIS EXISTS
---------------
LM Studio and Ollama can load multimodal LLMs but their HTTP APIs refuse audio input
(verified 2026-06: both reject audio content as "must be text or image_url"). Yet the
models themselves can hear — e.g. gemma-4-12b-qat, whose encoder-free design projects raw
audio straight into the token space, transcribes audio fine when driven through llama.cpp
directly. The capability lives in the engine, not the GUI.

So this gateway owns the contract. It sits in front of an engine (llama.cpp's
`llama-server` running the gemma-4-qat omni GGUF + audio projector on :8124) and exposes
one OpenAI-compatible endpoint that does EVERYTHING:

  - /v1/models, /v1/chat/completions, /v1/completions  -> transparently proxied to the
    engine (text + images + audio chat, streaming supported)
  - /v1/audio/transcriptions                            -> ADAPTED: wraps audio into a
    gemma chat call (thinking disabled) and returns the OpenAI transcription shape, so the
    Parakeet daemon's provider — and any OpenAI client — gets clean transcription.

RUN
---
    python local_model_server.py            # 127.0.0.1:8123, backend :8124
    uvicorn local_model_server:app --port 8123

ENV
---
    LMS_HOST / LMS_PORT       gateway bind (default 127.0.0.1:8123)
    LMS_BACKEND_URL           engine OpenAI base (default http://127.0.0.1:8124/v1)
    LMS_DEFAULT_MODEL         model id used when a request omits one
"""

from __future__ import annotations

import os
import base64
import logging

import httpx
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local_model_server")

HOST = os.environ.get("LMS_HOST", "127.0.0.1")
PORT = int(os.environ.get("LMS_PORT", "8123"))
BACKEND_URL = os.environ.get("LMS_BACKEND_URL", "http://127.0.0.1:8124/v1").rstrip("/")
DEFAULT_MODEL = os.environ.get("LMS_DEFAULT_MODEL", "gemma-4-12B-it-QAT-Q4_0.gguf")

# Reused async client. Long timeout: audio + 12B inference on Metal can take a while.
_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))

app = FastAPI(title="Local Model Server", version="1.0.0")

# Map common audio extensions to the `format` llama.cpp expects in input_audio.
_AUDIO_FMT = {
    ".wav": "wav", ".mp3": "mp3", ".m4a": "m4a", ".flac": "flac",
    ".ogg": "ogg", ".opus": "opus", ".webm": "webm",
}


@app.on_event("shutdown")
async def _shutdown():
    await _client.aclose()


@app.get("/health")
async def health():
    """Gateway status + whether the engine backend is reachable, and its models."""
    backend_ok, models = False, []
    try:
        r = await _client.get(f"{BACKEND_URL}/models", timeout=3.0)
        if r.status_code == 200:
            backend_ok = True
            models = [m.get("id") for m in r.json().get("data", [])]
    except Exception as exc:
        logger.debug("backend probe failed: %s", exc)
    return {
        "status": "ok",
        "backend_url": BACKEND_URL,
        "backend_reachable": backend_ok,
        "default_model": DEFAULT_MODEL,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Transparent proxy for the chat/completions/models surface (text+image+audio)
# ---------------------------------------------------------------------------
async def _proxy(request: Request, subpath: str):
    body = await request.body()
    url = f"{BACKEND_URL}/{subpath}"
    # Honor streaming (SSE) when the client asked for it.
    is_stream = b'"stream"' in body and b'"stream":true' in body.replace(b" ", b"")
    headers = {"content-type": "application/json"}
    if is_stream:
        async def gen():
            async with _client.stream("POST", url, content=body, headers=headers) as r:
                async for chunk in r.aiter_raw():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")
    try:
        r = await _client.post(url, content=body, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"engine unreachable at {BACKEND_URL}: {exc}")
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy(request, "chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy(request, "completions")


@app.get("/v1/models")
async def models():
    try:
        r = await _client.get(f"{BACKEND_URL}/models", timeout=5.0)
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"engine unreachable at {BACKEND_URL}: {exc}")


# ---------------------------------------------------------------------------
# Transcription adapter: OpenAI /v1/audio/transcriptions -> gemma audio chat
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    audio_bytes = await file.read()
    ext = os.path.splitext(file.filename or "audio.wav")[1].lower()
    fmt = _AUDIO_FMT.get(ext, "wav")
    b64 = base64.b64encode(audio_bytes).decode()

    instruction = prompt or (
        "Transcribe this audio verbatim. Output ONLY the transcription text, "
        "with no commentary, labels, or explanation."
    )
    if language:
        instruction += f" The audio language is {language}."

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
        ]}],
        "temperature": temperature,
        "max_tokens": 4096,
        # gemma-4 is a reasoning model; disable thinking so `content` holds the transcript
        # instead of it landing in reasoning_content and exhausting the token budget.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        r = await _client.post(f"{BACKEND_URL}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")
        logger.error("engine transcription error: %s %s", exc, detail)
        raise HTTPException(status_code=502, detail=f"engine transcription error: {exc} {detail}")

    msg = data.get("choices", [{}])[0].get("message", {}) or {}
    text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()

    fmt_out = (response_format or "json").lower()
    if fmt_out == "text":
        return PlainTextResponse(text)
    if fmt_out == "verbose_json":
        return JSONResponse({"task": "transcribe", "language": language, "text": text, "segments": []})
    return JSONResponse({"text": text})


if __name__ == "__main__":
    import uvicorn
    logger.info("Local Model Server on http://%s:%d (backend=%s)", HOST, PORT, BACKEND_URL)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
