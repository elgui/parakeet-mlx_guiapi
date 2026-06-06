"""
OpenAI-compatible audio transcription provider.

Uses the OpenAI `POST {base_url}/audio/transcriptions` contract (multipart upload),
which is what local audio servers converge on. Pairs with the companion Local Model
Server project (~/dev/local-model-server) but works against any OpenAI-compatible
STT endpoint.

Notes / limitations:
- LLM/Whisper-style transcription returns plain text. Without `verbose_json` segment
  timing we surface a single segment spanning the clip. No speaker diarization.
"""

import os
import logging
import tempfile
from typing import Optional

import requests

from ..base import (
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8123/v1"
DEFAULT_MODEL = "stub"


class OpenAITranscriptionProvider(STTProvider):
    """STT provider for OpenAI-compatible /audio/transcriptions servers."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        **kwargs,
    ):
        """
        Args:
            base_url: OpenAI-compatible base, e.g. "http://localhost:8123/v1".
            model: Model id the server should use.
            api_key: Optional bearer token (local servers usually need none).
            timeout: Request timeout in seconds (LLM transcription can be slow).
        """
        self.base_url = (base_url or os.environ.get("LMS_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("LMS_API_KEY")
        self.timeout = timeout
        logger.info("OpenAITranscriptionProvider base_url=%s model=%s", self.base_url, self.model)

    @property
    def name(self) -> str:
        return f"Local LLM ({self.model})"

    @property
    def supports_diarization(self) -> bool:
        return False

    @property
    def supports_streaming(self) -> bool:
        return False

    def is_available(self) -> tuple[bool, str]:
        """Ping the server's model list with a short timeout."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=3.0, headers=self._headers())
            if resp.status_code == 200:
                return True, f"Reachable at {self.base_url}"
            return False, f"Server returned {resp.status_code} at {self.base_url}"
        except requests.exceptions.RequestException as exc:
            return False, f"Not reachable at {self.base_url}: {exc}"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def transcribe(
        self,
        audio_path: str,
        enable_diarization: bool = True,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe an audio file via the OpenAI-compatible endpoint."""
        url = f"{self.base_url}/audio/transcriptions"
        # verbose_json yields duration + per-segment timing on capable servers
        # (e.g. whisper); minimal servers ignore it and still return {"text": ...},
        # which the parsing below handles.
        data = {"model": self.model, "response_format": "verbose_json"}
        if language:
            data["language"] = language

        try:
            with open(audio_path, "rb") as fh:
                files = {"file": (os.path.basename(audio_path), fh, "application/octet-stream")}
                resp = requests.post(
                    url, data=data, files=files, headers=self._headers(), timeout=self.timeout
                )
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", "")
            logger.error("Transcription server error: %s %s", exc, detail)
            raise RuntimeError(f"Transcription server error: {exc} {detail}")

        text = (payload.get("text") or "").strip()
        duration = payload.get("duration")

        segments: list[TranscriptionSegment] = []
        if payload.get("segments"):
            for seg in payload["segments"]:
                segments.append(TranscriptionSegment(
                    text=(seg.get("text") or "").strip(),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    speaker=None,
                ))
        elif text:
            segments.append(TranscriptionSegment(
                text=text, start=0.0, end=float(duration or 0.0), speaker=None
            ))

        return TranscriptionResult(
            segments=segments,
            full_text=text,
            language=payload.get("language") or language,
            duration=duration,
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        enable_diarization: bool = True,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe raw audio bytes by staging to a temp file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(audio_bytes)
            temp_path = fh.name
        try:
            return self.transcribe(
                temp_path,
                enable_diarization=enable_diarization,
                language=language,
                **kwargs,
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
