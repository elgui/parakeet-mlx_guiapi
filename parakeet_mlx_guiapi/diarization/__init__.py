"""
Speaker diarization module for Parakeet-MLX GUI and API.

This module provides speaker diarization using pyannote.audio.
"""

from .diarizer import SpeakerDiarizer, DiarizationResult

__all__ = ["SpeakerDiarizer", "DiarizationResult"]
