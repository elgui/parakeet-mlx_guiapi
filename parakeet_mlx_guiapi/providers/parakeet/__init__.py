"""
Parakeet-MLX provider for local speech-to-text.

Uses parakeet-mlx for transcription and pyannote.audio for diarization.
Runs entirely on-device using Apple Silicon MLX acceleration.
"""

from .provider import ParakeetProvider

__all__ = ["ParakeetProvider"]
