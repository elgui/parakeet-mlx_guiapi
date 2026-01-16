"""
Parakeet-MLX provider implementation.

Wraps the existing parakeet-mlx transcriber and pyannote diarizer
into the unified provider interface.
"""

import os
import tempfile
import logging
from typing import Optional, List

from ..base import (
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)


class ParakeetProvider(STTProvider):
    """
    Local STT provider using parakeet-mlx and pyannote.audio.

    Features:
    - Runs entirely on-device (no cloud API needed)
    - Uses MLX for Apple Silicon acceleration
    - Supports speaker diarization via pyannote.audio
    """

    def __init__(
        self,
        model_name: str = "mlx-community/parakeet-tdt-0.6b-v3",
        hf_token: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Parakeet provider.

        Args:
            model_name: HuggingFace model identifier
            hf_token: HuggingFace token for diarization models
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self._transcriber = None
        self._diarizer = None
        self._diarizer_available = None

    @property
    def name(self) -> str:
        return "Parakeet-MLX (Local)"

    @property
    def supports_diarization(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return False  # Parakeet processes complete audio chunks

    @property
    def transcriber(self):
        """Lazy-load the transcriber."""
        if self._transcriber is None:
            from parakeet_mlx_guiapi.api.routes import get_transcriber
            self._transcriber = get_transcriber()
        return self._transcriber

    @property
    def diarizer(self):
        """Lazy-load the diarizer."""
        if self._diarizer is None and self._diarizer_available is None:
            try:
                from parakeet_mlx_guiapi.diarization import SpeakerDiarizer
                available, msg = SpeakerDiarizer.is_available()
                self._diarizer_available = available
                if available:
                    self._diarizer = SpeakerDiarizer(hf_token=self.hf_token)
            except Exception as e:
                logger.warning(f"Diarization not available: {e}")
                self._diarizer_available = False
        return self._diarizer

    def is_available(self) -> tuple[bool, str]:
        """Check if parakeet-mlx is available."""
        try:
            import parakeet_mlx
            return True, "Parakeet-MLX available"
        except ImportError:
            return False, "parakeet-mlx not installed"

    def transcribe(
        self,
        audio_path: str,
        enable_diarization: bool = True,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe an audio file."""
        # Get chunk_duration from kwargs or use default
        chunk_duration = kwargs.get("chunk_duration", 30)

        # Run transcription
        df, full_text = self.transcriber.transcribe(
            audio_path,
            chunk_duration=chunk_duration
        )

        if df.empty:
            return TranscriptionResult(segments=[], full_text="")

        # Convert DataFrame to segments
        segments = []
        for _, row in df.iterrows():
            seg = TranscriptionSegment(
                text=row.get("Segment", row.get("text", "")),
                start=row.get("Start (s)", row.get("start", 0)),
                end=row.get("End (s)", row.get("end", 0)),
                speaker=None
            )
            segments.append(seg)

        # Apply diarization if enabled and available
        if enable_diarization and self.diarizer:
            try:
                diar_result = self.diarizer.diarize(audio_path)
                # Merge diarization with transcription
                for seg in segments:
                    mid_time = (seg.start + seg.end) / 2
                    speaker = diar_result.get_speaker_at_time(mid_time)
                    if speaker is None:
                        speaker = diar_result._find_closest_speaker(mid_time)
                    seg.speaker = speaker or "Unknown"
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
                for seg in segments:
                    seg.speaker = "Speaker"

        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language=language
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        enable_diarization: bool = True,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio from bytes."""
        # Save to temp file and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            return self.transcribe(
                temp_path,
                enable_diarization=enable_diarization,
                language=language,
                **kwargs
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
