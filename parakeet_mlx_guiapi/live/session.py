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


class LiveTranscriptionSession:
    """
    Manages a live transcription session.

    Handles audio chunk processing, speaker identification,
    color assignment, and message history.
    """

    def __init__(self, session_id: Optional[str] = None, enable_diarization: bool = True):
        """
        Initialize a new live transcription session.

        Args:
            session_id: Optional session ID (generated if not provided)
            enable_diarization: Whether to enable speaker diarization
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.enable_diarization = enable_diarization

        # Speaker tracking
        self._speaker_color_map: Dict[str, str] = {}
        self._next_speaker_num = 1

        # Message history
        self.messages: List[TranscriptionMessage] = []

        # Cumulative time offset (for multiple chunks)
        self._time_offset: float = 0.0

        # Lazy-loaded components
        self._transcriber = None
        self._diarizer = None
        self._diarization_available: Optional[bool] = None

        # Cross-chunk speaker tracking using embeddings
        self._embedding_model = None
        self._speaker_embeddings: Dict[str, np.ndarray] = {}  # global_id -> embedding
        self._next_global_speaker_id = 0
        self._embedding_similarity_threshold = 0.45  # Cosine similarity threshold (lowered for short chunks)

    @property
    def transcriber(self):
        """Get the transcriber instance (lazy loading, reuses singleton)."""
        if self._transcriber is None:
            from parakeet_mlx_guiapi.api.routes import get_transcriber
            self._transcriber = get_transcriber()
        return self._transcriber

    @property
    def diarizer(self):
        """Get the diarizer instance if available (lazy loading)."""
        if self._diarizer is None and self.enable_diarization:
            if self._diarization_available is None:
                # Check if diarization is available
                try:
                    from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer
                    available, msg = SpeakerDiarizer.is_available()
                    logger.info(f"Diarization availability check: available={available}, msg={msg}")
                    self._diarization_available = available
                    if available:
                        logger.info("Initializing SpeakerDiarizer...")
                        self._diarizer = SpeakerDiarizer()
                        logger.info("SpeakerDiarizer initialized successfully")
                    else:
                        logger.warning(f"Diarization not available: {msg}")
                except Exception as e:
                    logger.error(f"Diarization init failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    self._diarization_available = False
        return self._diarizer

    @property
    def embedding_model(self):
        """Get the speaker embedding model (lazy loading)."""
        if self._embedding_model is None:
            try:
                from speechbrain.inference import EncoderClassifier
                logger.info("Loading speaker embedding model...")
                self._embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="/tmp/speechbrain_model"
                )
                logger.info("Speaker embedding model loaded")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self._embedding_model = False  # Mark as unavailable
        return self._embedding_model if self._embedding_model is not False else None

    def _extract_speaker_embedding(self, audio_path: str, start: float, end: float) -> Optional[np.ndarray]:
        """Extract speaker embedding for a time segment."""
        if not self.embedding_model:
            return None

        try:
            import torchaudio

            # Load audio segment (explicitly use soundfile backend to avoid torchcodec issues)
            signal, fs = torchaudio.load(audio_path, backend="soundfile")

            # Convert time to samples
            start_sample = int(start * fs)
            end_sample = int(end * fs)

            # Ensure valid range
            if start_sample >= signal.shape[1] or end_sample <= start_sample:
                return None

            segment = signal[:, start_sample:end_sample]

            # Need at least 0.5 seconds for good embedding
            if segment.shape[1] < fs * 0.5:
                return None

            # Extract embedding
            embedding = self.embedding_model.encode_batch(segment)
            return embedding.squeeze().numpy()

        except Exception as e:
            logger.warning(f"Could not extract embedding: {e}")
            return None

    def _match_speaker_to_known(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match an embedding to known speakers.

        Returns:
            Tuple of (matched_global_id or None, similarity_score)
        """
        if not self._speaker_embeddings:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for global_id, known_embedding in self._speaker_embeddings.items():
            # Cosine similarity
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
        """
        Get an existing global speaker ID if embedding matches, or create a new one.
        """
        if embedding is not None:
            matched_id, similarity = self._match_speaker_to_known(embedding)
            logger.info(f"Embedding match: best_match={matched_id}, similarity={similarity:.3f}, threshold={self._embedding_similarity_threshold}")
            if matched_id:
                logger.info(f"✓ MATCHED to existing speaker {matched_id} (similarity: {similarity:.3f})")
                # Update embedding with running average for better matching
                self._speaker_embeddings[matched_id] = (
                    self._speaker_embeddings[matched_id] * 0.8 + embedding * 0.2
                )
                return matched_id
            else:
                logger.info(f"✗ NO MATCH (best similarity {similarity:.3f} < threshold {self._embedding_similarity_threshold})")

        # Create new speaker
        global_id = f"GLOBAL_SPEAKER_{self._next_global_speaker_id:02d}"
        self._next_global_speaker_id += 1

        if embedding is not None:
            self._speaker_embeddings[global_id] = embedding
            logger.info(f"Created NEW speaker {global_id} (total: {len(self._speaker_embeddings)} speakers)")
        else:
            logger.warning(f"Created speaker {global_id} WITHOUT embedding (total: {len(self._speaker_embeddings)} speakers)")

        return global_id

    def get_speaker_color(self, speaker_id: str) -> str:
        """
        Get a consistent color for a speaker.

        Args:
            speaker_id: The speaker identifier (e.g., "SPEAKER_00")

        Returns:
            Hex color code for the speaker
        """
        if speaker_id not in self._speaker_color_map:
            color_index = len(self._speaker_color_map) % len(SPEAKER_COLORS)
            self._speaker_color_map[speaker_id] = SPEAKER_COLORS[color_index]
        return self._speaker_color_map[speaker_id]

    def get_speaker_name(self, speaker_id: str) -> str:
        """
        Get a human-readable name for a speaker.

        Args:
            speaker_id: The speaker identifier

        Returns:
            Human-readable name (e.g., "Speaker 1")
        """
        # Track speaker numbers consistently
        if speaker_id not in self._speaker_color_map:
            self.get_speaker_color(speaker_id)  # Ensure color is assigned

        # Convert GLOBAL_SPEAKER_00 -> Speaker 1, GLOBAL_SPEAKER_01 -> Speaker 2, etc.
        if speaker_id.startswith("GLOBAL_SPEAKER_"):
            try:
                num = int(speaker_id.split("_")[2]) + 1
                return f"Speaker {num}"
            except (IndexError, ValueError):
                pass

        # Convert SPEAKER_00 -> Speaker 1, SPEAKER_01 -> Speaker 2, etc.
        if speaker_id.startswith("SPEAKER_"):
            try:
                num = int(speaker_id.split("_")[1]) + 1
                return f"Speaker {num}"
            except (IndexError, ValueError):
                pass

        # Count position in our map
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
        logger.info(f"=== Processing audio chunk at {chunk_start_time:.1f}s ===")

        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        logger.debug(f"Decoded {len(audio_bytes)} bytes of audio data")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        logger.debug(f"Saved to temp file: {temp_path}")

        try:
            import time

            # Transcribe with chunking enabled for proper diarization merge
            # Each chunk gets its own timestamp range for speaker assignment
            logger.info(">>> Calling transcriber.transcribe()...")
            t0 = time.time()
            df, full_text = self.transcriber.transcribe(
                temp_path,
                chunk_duration=30  # Enable chunking for multi-speaker support
            )
            t_transcribe = time.time() - t0
            logger.info(f"<<< Transcription took {t_transcribe:.1f}s: {len(df)} rows, text: '{full_text[:100] if full_text else '(empty)'}...'")

            if df.empty:
                logger.warning("DataFrame is empty - no transcription results!")
                return []

            # Convert DataFrame to list of dicts
            segments = df.to_dict(orient='records')
            logger.debug(f"Converted to {len(segments)} segments")

            # Apply diarization if available
            logger.info(f"Diarization check: enable_diarization={self.enable_diarization}, diarizer={self.diarizer is not None}, _diarization_available={self._diarization_available}")
            self._last_diarization_info = None  # Track what happened
            local_to_global_speaker = {}  # Map per-chunk SPEAKER_XX to global IDs

            if self.diarizer:
                try:
                    logger.info(">>> Running speaker diarization...")
                    t1 = time.time()
                    diarization_result = self.diarizer.diarize(temp_path)
                    t_diarize = time.time() - t1
                    num_diar_segments = len(diarization_result.segments)
                    num_diar_speakers = diarization_result.num_speakers
                    logger.info(f"<<< Diarization raw result: {num_diar_speakers} speaker(s), {num_diar_segments} segments in {t_diarize:.1f}s")

                    # Cross-chunk speaker tracking using embeddings
                    # Group diarization segments by speaker to get time ranges
                    speaker_time_ranges = {}
                    for diar_seg in diarization_result.segments:
                        speaker = diar_seg.speaker
                        if speaker not in speaker_time_ranges:
                            speaker_time_ranges[speaker] = []
                        speaker_time_ranges[speaker].append((diar_seg.start, diar_seg.end))

                    # Extract embeddings and map to global speakers
                    logger.info(f">>> Extracting embeddings for {len(speaker_time_ranges)} local speaker(s)")
                    for local_speaker, time_ranges in speaker_time_ranges.items():
                        # Use the longest segment for embedding extraction
                        longest = max(time_ranges, key=lambda x: x[1] - x[0])
                        start, end = longest

                        # Ensure minimum duration
                        duration = end - start
                        if duration < 1.0:
                            # Try to expand by combining adjacent segments
                            all_start = min(t[0] for t in time_ranges)
                            all_end = max(t[1] for t in time_ranges)
                            start, end = all_start, all_end
                            logger.debug(f"Expanded segment for {local_speaker}: {start:.1f}-{end:.1f}s")

                        embedding = self._extract_speaker_embedding(temp_path, start, end)
                        global_id = self._get_or_create_global_speaker_id(embedding)
                        local_to_global_speaker[local_speaker] = global_id
                        logger.info(f"Speaker mapping: {local_speaker} -> {global_id} (embedding: {'extracted' if embedding is not None else 'FAILED'})")

                    segments = diarization_result.merge_with_transcription(segments)

                    # Replace local speaker IDs with global IDs
                    for seg in segments:
                        local_speaker = seg.get('speaker', 'SPEAKER_00')
                        if local_speaker in local_to_global_speaker:
                            seg['speaker'] = local_to_global_speaker[local_speaker]
                        else:
                            # Fallback: create new global speaker
                            global_id = self._get_or_create_global_speaker_id(None)
                            seg['speaker'] = global_id

                    unique_speakers = set(seg.get('speaker', 'unknown') for seg in segments)
                    logger.info(f"<<< After merge: {len(unique_speakers)} unique speaker(s): {unique_speakers}")

                    self._last_diarization_info = {
                        'ran': True,
                        'time': t_diarize,
                        'raw_speakers': num_diar_speakers,
                        'raw_segments': num_diar_segments,
                        'merged_speakers': list(unique_speakers),
                        'speaker_mapping': local_to_global_speaker
                    }
                except Exception as e:
                    logger.error(f"Diarization failed: {e}, using default speaker")
                    import traceback
                    logger.error(traceback.format_exc())
                    default_global = self._get_or_create_global_speaker_id(None)
                    for seg in segments:
                        seg['speaker'] = default_global
                    self._last_diarization_info = {'ran': False, 'error': str(e)}
            else:
                # No diarization - assign default speaker
                logger.warning(f"Diarization SKIPPED: enable_diarization={self.enable_diarization}, _diarization_available={self._diarization_available}")
                default_global = self._get_or_create_global_speaker_id(None)
                for seg in segments:
                    seg['speaker'] = default_global
                self._last_diarization_info = {'ran': False, 'reason': 'diarizer_not_available'}

            # Create messages
            messages = []
            for seg in segments:
                speaker_id = seg.get('speaker', 'SPEAKER_00')

                # Get start/end times (handle different column names)
                start = seg.get('start', seg.get('Start (s)', 0))
                end = seg.get('end', seg.get('End (s)', 0))
                text = seg.get('text', seg.get('Segment', ''))

                # Adjust times based on session offset
                adjusted_start = chunk_start_time + start
                adjusted_end = chunk_start_time + end

                msg = TranscriptionMessage(
                    speaker=self.get_speaker_name(speaker_id),
                    speaker_id=speaker_id,
                    text=text,
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

            return messages

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def export_txt(self) -> str:
        """
        Export conversation as plain text with speaker labels.

        Returns:
            Formatted text transcript
        """
        lines = []
        current_speaker = None
        current_texts = []

        for msg in self.messages:
            if msg.speaker != current_speaker:
                if current_texts:
                    lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                    lines.append("")  # Empty line between speakers
                current_speaker = msg.speaker
                current_texts = [msg.text]
            else:
                current_texts.append(msg.text)

        # Don't forget last speaker
        if current_texts:
            lines.append(f"{current_speaker}: {' '.join(current_texts)}")

        return '\n'.join(lines)

    def export_srt(self) -> str:
        """
        Export conversation as SRT subtitle format.

        Returns:
            SRT formatted transcript
        """
        def format_time(seconds: float) -> str:
            """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
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
            lines.append("")  # Empty line between entries

        return '\n'.join(lines)

    def clear(self):
        """Clear the session history."""
        self.messages.clear()
        self._speaker_color_map.clear()
        self._time_offset = 0.0
        # Reset cross-chunk speaker tracking
        self._speaker_embeddings.clear()
        self._next_global_speaker_id = 0
