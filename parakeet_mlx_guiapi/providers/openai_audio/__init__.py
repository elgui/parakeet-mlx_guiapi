"""
OpenAI-compatible audio transcription provider.

Talks to any server exposing OpenAI's POST /v1/audio/transcriptions — our own
Local Model Server gateway (local_model_server.py), faster-whisper-server / speaches,
llama.cpp's llama-server, LocalAI, vLLM, or OpenAI itself. The endpoint, model, and key
are configurable, so the same provider points at a local LLM today and anything
OpenAI-compatible tomorrow.
"""

from .provider import OpenAITranscriptionProvider

__all__ = ["OpenAITranscriptionProvider"]
