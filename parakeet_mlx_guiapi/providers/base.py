"""
Base classes for STT and diarization providers.

This module defines the abstract interfaces that all providers must implement,
enabling clean separation of concerns and easy provider switching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class ProviderType(Enum):
    """Available provider types."""
    PARAKEET = "parakeet"  # Local MLX-based transcription + pyannote diarization
    DEEPGRAM = "deepgram"  # Cloud-based STT with built-in diarization


@dataclass
class TranscriptionSegment:
    """A single transcription segment with speaker info."""
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker or "Unknown",
            "confidence": self.confidence
        }


@dataclass
class TranscriptionResult:
    """Result from transcription with optional diarization."""
    segments: List[TranscriptionSegment]
    full_text: str
    language: Optional[str] = None
    duration: Optional[float] = None

    @property
    def speakers(self) -> List[str]:
        """Get unique speakers in this result."""
        return list(set(seg.speaker for seg in self.segments if seg.speaker))


class STTProvider(ABC):
    """Abstract base class for Speech-to-Text providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        pass

    @property
    @abstractmethod
    def supports_diarization(self) -> bool:
        """Whether this provider supports speaker diarization."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports real-time streaming."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        enable_diarization: bool = True,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            enable_diarization: Whether to enable speaker diarization
            language: Optional language code (e.g., "en", "fr")
            **kwargs: Provider-specific options

        Returns:
            TranscriptionResult with segments and optional speaker labels
        """
        pass

    @abstractmethod
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        enable_diarization: bool = True,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio from bytes (e.g., WAV data).

        Args:
            audio_bytes: Raw audio bytes (typically WAV format)
            enable_diarization: Whether to enable speaker diarization
            language: Optional language code
            **kwargs: Provider-specific options

        Returns:
            TranscriptionResult with segments and optional speaker labels
        """
        pass

    def is_available(self) -> tuple[bool, str]:
        """
        Check if this provider is available and configured.

        Returns:
            Tuple of (is_available, message)
        """
        return True, "Provider available"


class DiarizationProvider(ABC):
    """Abstract base class for standalone diarization providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        pass

    @abstractmethod
    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[TranscriptionSegment]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers

        Returns:
            List of segments with speaker labels (no text)
        """
        pass

    def is_available(self) -> tuple[bool, str]:
        """Check if this provider is available."""
        return True, "Provider available"


def get_provider(provider_type: ProviderType, **config) -> STTProvider:
    """
    Factory function to get a provider instance.

    Args:
        provider_type: Which provider to use
        **config: Provider-specific configuration

    Returns:
        Configured STTProvider instance
    """
    if provider_type == ProviderType.PARAKEET:
        from .parakeet import ParakeetProvider
        return ParakeetProvider(**config)
    elif provider_type == ProviderType.DEEPGRAM:
        from .deepgram import DeepgramProvider
        return DeepgramProvider(**config)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
