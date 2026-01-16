"""
Live transcription session management.

This module provides the LiveTranscriptionSession class for managing
real-time transcription sessions with speaker diarization.
"""

import os
import uuid
import base64
import tempfile
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

from parakeet_mlx_guiapi.providers import (
    ProviderType,
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
    get_provider,
)
from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer

# Set up logging
logger = logging.getLogger('live_session')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[SESSION %(asctime)s] %(levelname)s: %(message)s', '%H:%M:%S'))
    logger.addHandler(handler)


# Speaker colors palette (8 distinct light colors for readability)
SPEAKER_COLORS = [
    "#E3F2FD",  # Light blue
    "#FFF3E0",  # Light orange
    "#E8F5E9",  # Light green
    "#FCE4EC",  # Light pink
    "#F3E5F5",  # Light purple
    "#FFFDE7",  # Light yellow
    "#E0F7FA",  # Light cyan
    "#FBE9E7",  # Light deep orange
]


@dataclass
class TranscriptionMessage:
    """A single transcription message with speaker info."""
    speaker: str
    speaker_id: str
    text: str
    start_time: float
    end_time: float
    color: str
    timestamp: datetime = field(default_factory=datetime.now)


def clean_transcription_text(text: str) -> str:
    """Clean transcription text by removing <unk> tokens and normalizing whitespace."""
    import re
    # Remove <unk> tokens
    cleaned = re.sub(r'<unk>', '', text)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


class LiveTranscriptionSession:
    """
    Manages a live transcription session.

    Handles audio chunk processing, speaker identification,
    color assignment, and message history.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        enable_diarization: bool = True,
        provider_type: ProviderType = ProviderType.PARAKEET,
        provider_config: Optional[Dict] = None
    ):
        """
        Initialize a new live transcription session.

        Args:
            session_id: Optional session ID (generated if not provided)
            enable_diarization: Whether to enable speaker diarization
            provider_type: Which STT provider to use (PARAKEET or DEEPGRAM)
            provider_config: Provider-specific configuration (e.g., API keys)
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.enable_diarization = enable_diarization
        self.provider_type = provider_type
        self.provider_config = provider_config or {}

        # Speaker tracking
        self._speaker_color_map: Dict[str, str] = {}
        self._next_speaker_num = 1

        # Message history
        self.messages: List[TranscriptionMessage] = []

        # Cumulative time offset (for multiple chunks)
        self._time_offset: float = 0.0

        # Lazy-loaded provider
        self._provider: Optional[STTProvider] = None

        # Cross-chunk speaker tracking using embeddings (for Parakeet provider)
        self._embedding_model = None
        self._speaker_embeddings: Dict[str, np.ndarray] = {}  # global_id -> embedding
        self._next_global_speaker_id = 0
        self._embedding_similarity_threshold = 0.45

        # For tracking diarization info
        self._last_diarization_info = None

        # Legacy compatibility
        self._diarization_available = None

        # Local pyannote diarizer for within-chunk speaker detection
        # Used when cloud providers (Deepgram) fail to detect speakers in short chunks
        self._local_diarizer: Optional[SpeakerDiarizer] = None
        self._local_diarization_available: Optional[bool] = None

    @property
    def provider(self) -> STTProvider:
        """Get the STT provider instance (lazy loading)."""
        if self._provider is None:
            logger.info(f"Initializing provider: {self.provider_type.value}")
            self._provider = get_provider(self.provider_type, **self.provider_config)
            logger.info(f"Provider initialized: {self._provider.name}")
        return self._provider

    @property
    def diarizer(self):
        """Legacy property for backward compatibility."""
        if self.provider_type == ProviderType.PARAKEET:
            # Return the Parakeet provider's diarizer if available
            if hasattr(self.provider, 'diarizer'):
                return self.provider.diarizer
        # Deepgram has built-in diarization, no separate diarizer
        return None if not self.enable_diarization else True

    @property
    def local_diarizer(self) -> Optional[SpeakerDiarizer]:
        """Get the local pyannote diarizer (lazy loading).

        Used for within-chunk speaker detection when cloud providers
        (like Deepgram) fail to detect speakers in short audio chunks.
        """
        if self._local_diarization_available is None:
            is_available, msg = SpeakerDiarizer.is_available()
            self._local_diarization_available = is_available
            if is_available:
                logger.info("Local pyannote diarization available")
            else:
                logger.warning(f"Local diarization not available: {msg}")

        if self._local_diarization_available and self._local_diarizer is None:
            try:
                logger.info("Loading local pyannote diarizer for within-chunk speaker detection...")
                self._local_diarizer = SpeakerDiarizer()
                # Trigger initialization
                self._local_diarizer._ensure_initialized()
                logger.info("Local diarizer loaded")
            except Exception as e:
                logger.warning(f"Could not load local diarizer: {e}")
                self._local_diarization_available = False

        return self._local_diarizer

    @property
    def embedding_model(self):
        """Get the speaker embedding model (lazy loading) for cross-chunk speaker tracking."""
        if self._embedding_model is None:
            try:
                from speechbrain.inference import EncoderClassifier
                logger.info("Loading speaker embedding model for cross-chunk tracking...")
                self._embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="/tmp/speechbrain_model"
                )
                logger.info("Speaker embedding model loaded")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self._embedding_model = False
        return self._embedding_model if self._embedding_model is not False else None

    def _extract_speaker_embedding(self, audio_path: str, start: float, end: float) -> Optional[np.ndarray]:
        """Extract speaker embedding for a time segment."""
        if not self.embedding_model:
            return None

        try:
            import torchaudio
            signal, fs = torchaudio.load(audio_path, backend="soundfile")
            start_sample = int(start * fs)
            end_sample = int(end * fs)

            if start_sample >= signal.shape[1] or end_sample <= start_sample:
                return None

            segment = signal[:, start_sample:end_sample]
            if segment.shape[1] < fs * 0.5:
                return None

            embedding = self.embedding_model.encode_batch(segment)
            return embedding.squeeze().numpy()
        except Exception as e:
            logger.warning(f"Could not extract embedding: {e}")
            return None

    def _match_speaker_to_known(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Match an embedding to known speakers."""
        if not self._speaker_embeddings:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for global_id, known_embedding in self._speaker_embeddings.items():
            similarity = np.dot(embedding, known_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = global_id

        if best_similarity >= self._embedding_similarity_threshold:
            return best_match, best_similarity
        return None, best_similarity

    def _get_or_create_global_speaker_id(self, embedding: Optional[np.ndarray]) -> str:
        """Get an existing global speaker ID or create a new one."""
        if embedding is not None:
            matched_id, similarity = self._match_speaker_to_known(embedding)
            if matched_id:
                logger.info(f"✓ MATCHED to {matched_id} (similarity: {similarity:.3f})")
                self._speaker_embeddings[matched_id] = (
                    self._speaker_embeddings[matched_id] * 0.8 + embedding * 0.2
                )
                return matched_id

        global_id = f"GLOBAL_SPEAKER_{self._next_global_speaker_id:02d}"
        self._next_global_speaker_id += 1

        if embedding is not None:
            self._speaker_embeddings[global_id] = embedding
            logger.info(f"Created NEW speaker {global_id}")
        return global_id

    def get_speaker_color(self, speaker_id: str) -> str:
        """Get a consistent color for a speaker."""
        if speaker_id not in self._speaker_color_map:
            color_index = len(self._speaker_color_map) % len(SPEAKER_COLORS)
            self._speaker_color_map[speaker_id] = SPEAKER_COLORS[color_index]
        return self._speaker_color_map[speaker_id]

    def get_speaker_name(self, speaker_id: str) -> str:
        """Get a human-readable name for a speaker."""
        if speaker_id not in self._speaker_color_map:
            self.get_speaker_color(speaker_id)

        # Convert GLOBAL_SPEAKER_XX or SPEAKER_XX to "Speaker N"
        for prefix in ["GLOBAL_SPEAKER_", "SPEAKER_"]:
            if speaker_id.startswith(prefix):
                try:
                    num = int(speaker_id.split("_")[-1]) + 1
                    return f"Speaker {num}"
                except (IndexError, ValueError):
                    pass

        speakers = list(self._speaker_color_map.keys())
        if speaker_id in speakers:
            return f"Speaker {speakers.index(speaker_id) + 1}"
        return speaker_id

    def process_audio_chunk(self, audio_base64: str, chunk_start_time: float = 0.0) -> List[Dict]:
        """
        Process an audio chunk and return transcription messages.

        Args:
            audio_base64: Base64-encoded WAV audio data
            chunk_start_time: Start time of this chunk in the overall session

        Returns:
            List of message dictionaries with speaker info and colors
        """
        logger.info(f"=== Processing audio chunk at {chunk_start_time:.1f}s ({self.provider_type.value}) ===")

        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        logger.debug(f"Decoded {len(audio_bytes)} bytes of audio data")

        # Save to temporary file (needed for some processing)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            import time
            t0 = time.time()

            # Use the provider for transcription
            logger.info(f">>> Calling {self.provider.name} transcribe...")
            result = self.provider.transcribe_bytes(
                audio_bytes,
                enable_diarization=self.enable_diarization
            )
            t_total = time.time() - t0
            logger.info(f"<<< Transcription took {t_total:.1f}s: {len(result.segments)} segments")

            if not result.segments:
                logger.warning("No transcription results!")
                return []

            # Handle diarization
            if self.enable_diarization:
                # Check if cloud provider's diarization failed (all same speaker)
                if self._needs_local_diarization(result):
                    result = self._apply_local_diarization(result, temp_path)

                # Apply cross-chunk speaker tracking using embeddings
                result = self._apply_cross_chunk_speaker_tracking(result, temp_path)

            # Convert provider result to messages
            messages = []
            for seg in result.segments:
                # Clean text and skip empty segments
                cleaned_text = clean_transcription_text(seg.text)
                if not cleaned_text:
                    logger.debug(f"Skipping empty segment after cleaning: '{seg.text[:50]}...'")
                    continue

                speaker_id = seg.speaker or "SPEAKER_00"
                adjusted_start = chunk_start_time + seg.start
                adjusted_end = chunk_start_time + seg.end

                msg = TranscriptionMessage(
                    speaker=self.get_speaker_name(speaker_id),
                    speaker_id=speaker_id,
                    text=cleaned_text,
                    start_time=adjusted_start,
                    end_time=adjusted_end,
                    color=self.get_speaker_color(speaker_id)
                )
                self.messages.append(msg)

                messages.append({
                    'speaker': msg.speaker,
                    'speaker_id': msg.speaker_id,
                    'text': msg.text,
                    'start_time': msg.start_time,
                    'end_time': msg.end_time,
                    'color': msg.color
                })

            self._last_diarization_info = {
                'ran': True,
                'provider': self.provider_type.value,
                'segments': len(result.segments),
                'speakers': result.speakers
            }

            return messages

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _apply_cross_chunk_speaker_tracking(
        self,
        result: TranscriptionResult,
        audio_path: str
    ) -> TranscriptionResult:
        """Apply cross-chunk speaker tracking using speaker embeddings.

        This works for all providers by extracting speaker embeddings from
        audio segments and matching them to previously seen speakers.
        """
        # Group segments by speaker
        speaker_segments = {}
        for seg in result.segments:
            speaker = seg.speaker or "Unknown"
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)

        # Map local speaker IDs to global IDs
        local_to_global = {}
        for local_speaker, segs in speaker_segments.items():
            # Find longest segment for embedding
            longest = max(segs, key=lambda s: s.end - s.start)
            embedding = self._extract_speaker_embedding(
                audio_path, longest.start, longest.end
            )
            global_id = self._get_or_create_global_speaker_id(embedding)
            local_to_global[local_speaker] = global_id

        # Update segments with global IDs
        for seg in result.segments:
            if seg.speaker in local_to_global:
                seg.speaker = local_to_global[seg.speaker]

        return result

    def _apply_local_diarization(
        self,
        result: TranscriptionResult,
        audio_path: str
    ) -> TranscriptionResult:
        """Apply local pyannote diarization to transcription results.

        Used when cloud providers (Deepgram) fail to detect multiple speakers
        within a single audio chunk. This replaces the provider's speaker
        labels with local diarization results.

        Args:
            result: Transcription result from provider (with text and timestamps)
            audio_path: Path to the audio file

        Returns:
            Updated TranscriptionResult with accurate speaker labels
        """
        if not self.local_diarizer:
            logger.debug("Local diarizer not available, skipping local diarization")
            return result

        try:
            import time
            t0 = time.time()
            logger.info("Running local pyannote diarization for within-chunk speaker detection...")

            # Run diarization
            diarization_result = self.local_diarizer.diarize(audio_path)
            t_diarize = time.time() - t0
            logger.info(f"Local diarization: {diarization_result.num_speakers} speakers in {t_diarize:.1f}s")

            if diarization_result.num_speakers <= 1:
                logger.debug("Local diarization found <=1 speaker, no change needed")
                return result

            # Merge transcription with diarization
            # For each segment, find the speaker based on midpoint time
            updated_segments = []
            for seg in result.segments:
                mid_time = (seg.start + seg.end) / 2
                speaker = diarization_result.get_speaker_at_time(mid_time)
                if speaker is None:
                    speaker = diarization_result._find_closest_speaker(mid_time)

                updated_segments.append(TranscriptionSegment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    speaker=speaker or seg.speaker,
                    confidence=seg.confidence
                ))

            # Log the speaker distribution
            local_speakers = set(s.speaker for s in updated_segments)
            logger.info(f"Local diarization detected {len(local_speakers)} distinct speakers in chunk")

            return TranscriptionResult(
                segments=updated_segments,
                full_text=result.full_text,
                language=result.language,
                duration=result.duration
            )

        except Exception as e:
            logger.warning(f"Local diarization failed: {e}")
            return result

    def _needs_local_diarization(self, result: TranscriptionResult) -> bool:
        """Check if local diarization is needed for this result.

        Returns True if:
        - Using a cloud provider (Deepgram)
        - Diarization is enabled
        - All segments have the same speaker (cloud diarization failed)
        - There are multiple segments (potential for different speakers)
        """
        if self.provider_type == ProviderType.PARAKEET:
            return False  # Parakeet uses its own local diarization

        if not self.enable_diarization:
            return False

        if len(result.segments) <= 1:
            return False

        # Check if all segments have the same speaker
        speakers = set(seg.speaker for seg in result.segments)
        if len(speakers) > 1:
            return False  # Cloud diarization worked

        logger.info(f"Cloud provider returned all segments with same speaker, will try local diarization")
        return True

    def export_txt(self) -> str:
        """Export conversation as plain text with speaker labels."""
        lines = []
        current_speaker = None
        current_texts = []

        for msg in self.messages:
            if msg.speaker != current_speaker:
                if current_texts:
                    lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                    lines.append("")
                current_speaker = msg.speaker
                current_texts = [msg.text]
            else:
                current_texts.append(msg.text)

        if current_texts:
            lines.append(f"{current_speaker}: {' '.join(current_texts)}")
        return '\n'.join(lines)

    def export_srt(self) -> str:
        """Export conversation as SRT subtitle format."""
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        lines = []
        for i, msg in enumerate(self.messages, 1):
            lines.append(str(i))
            lines.append(f"{format_time(msg.start_time)} --> {format_time(msg.end_time)}")
            lines.append(f"[{msg.speaker}] {msg.text}")
            lines.append("")
        return '\n'.join(lines)

    def clear(self):
        """Clear the session history."""
        self.messages.clear()
        self._speaker_color_map.clear()
        self._time_offset = 0.0
        self._speaker_embeddings.clear()
        self._next_global_speaker_id = 0
