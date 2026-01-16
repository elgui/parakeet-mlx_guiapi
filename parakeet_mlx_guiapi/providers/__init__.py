"""
Provider abstraction layer for STT and diarization services.

This module provides a clean interface for switching between different
speech-to-text and diarization providers.
"""

from .base import (
    ProviderType,
    STTProvider,
    DiarizationProvider,
    TranscriptionSegment,
    TranscriptionResult,
    get_provider,
)

__all__ = [
    "ProviderType",
    "STTProvider",
    "DiarizationProvider",
    "TranscriptionSegment",
    "TranscriptionResult",
    "get_provider",
]
