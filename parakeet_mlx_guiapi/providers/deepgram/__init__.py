"""
Deepgram provider for cloud-based speech-to-text with built-in diarization.

Uses Deepgram's Nova-2 model for high-accuracy transcription with
native speaker diarization support.
"""

from .provider import DeepgramProvider

__all__ = ["DeepgramProvider"]
