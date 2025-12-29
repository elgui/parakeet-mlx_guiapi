"""
Speaker diarization module for Parakeet-MLX GUI and API.

This module provides speaker diarization using pyannote.audio.
"""

# CRITICAL: Patch pyannote's Audio class to use soundfile backend BEFORE import
# This fixes the "AudioDecoder is not defined" error in torchaudio 2.8+
# (torchcodec is not installed, but torchaudio tries to use it by default)
try:
    from pyannote.audio.core.io import Audio as _PyannoteAudio
    _original_audio_init = _PyannoteAudio.__init__

    def _patched_audio_init(self, sample_rate=None, mono=None, backend=None):
        # Always force soundfile backend
        _original_audio_init(self, sample_rate=sample_rate, mono=mono, backend="soundfile")

    _PyannoteAudio.__init__ = _patched_audio_init
except Exception:
    pass  # pyannote not installed or patch failed - will handle later

from .diarizer import SpeakerDiarizer, DiarizationResult

__all__ = ["SpeakerDiarizer", "DiarizationResult"]
