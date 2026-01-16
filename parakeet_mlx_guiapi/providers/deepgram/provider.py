"""
Deepgram provider implementation.

Uses Deepgram's REST API for speech-to-text with built-in diarization.
"""

import os
import json
import logging
from typing import Optional
import requests

from ..base import (
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)

# Deepgram API endpoint
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"


class DeepgramProvider(STTProvider):
    """
    Cloud-based STT provider using Deepgram's API.

    Features:
    - High-accuracy Nova-3 and Nova-2 models
    - Built-in speaker diarization
    - Multi-language support
    - Fast processing
    - Configurable formatting options
    """

    # Available Deepgram models
    MODELS = {
        # Nova-3 models (latest, best accuracy)
        "nova-3": "Nova-3 (General)",
        "nova-3-general": "Nova-3 General",
        "nova-3-meeting": "Nova-3 Meeting",
        "nova-3-phonecall": "Nova-3 Phone",
        "nova-3-voicemail": "Nova-3 Voicemail",
        "nova-3-finance": "Nova-3 Finance",
        "nova-3-medical": "Nova-3 Medical",
        # Nova-2 models (still excellent)
        "nova-2": "Nova-2 (General)",
        "nova-2-meeting": "Nova-2 Meeting",
        "nova-2-phonecall": "Nova-2 Phone",
        "nova-2-voicemail": "Nova-2 Voicemail",
        "nova-2-finance": "Nova-2 Finance",
        "nova-2-medical": "Nova-2 Medical",
    }

    # Default options
    DEFAULT_OPTIONS = {
        "smart_format": True,   # Auto-capitalize, format numbers, etc.
        "punctuate": True,      # Add punctuation
        "paragraphs": True,     # Group into paragraphs
        "utterances": True,     # Break by utterances (needed for diarization)
        "profanity_filter": False,  # Filter profanity
        "numerals": False,      # Convert numbers to digits (1 instead of "one")
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-3",
        options: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize the Deepgram provider.

        Args:
            api_key: Deepgram API key (or set DEEPGRAM_API_KEY env var)
            model: Deepgram model to use (default: nova-3)
            options: Dict of options (smart_format, punctuate, etc.)
        """
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        self.model = model
        # Merge default options with provided options
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        logger.info(f"DeepgramProvider initialized with model: {model}")
        logger.info(f"Deepgram options: {self.options}")

    @property
    def name(self) -> str:
        model_name = self.MODELS.get(self.model, self.model)
        return f"Deepgram {model_name}"

    @property
    def supports_diarization(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True  # Deepgram supports WebSocket streaming

    def is_available(self) -> tuple[bool, str]:
        """Check if Deepgram is configured."""
        if not self.api_key:
            return False, "Deepgram API key not configured"
        return True, "Deepgram available"

    def transcribe(
        self,
        audio_path: str,
        enable_diarization: bool = True,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe an audio file using Deepgram API."""
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return self.transcribe_bytes(
            audio_bytes,
            enable_diarization=enable_diarization,
            language=language,
            **kwargs
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        enable_diarization: bool = True,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio bytes using Deepgram API.

        Args:
            audio_bytes: Raw audio data (WAV, MP3, etc.)
            enable_diarization: Enable speaker diarization
            language: Language code (e.g., "en", "fr") or None for auto-detect
        """
        if not self.api_key:
            raise ValueError("Deepgram API key not configured")

        # Build query parameters using configurable options
        params = {
            "model": self.model,
        }

        # Apply configurable options
        if self.options.get("smart_format"):
            params["smart_format"] = "true"
        if self.options.get("punctuate"):
            params["punctuate"] = "true"
        if self.options.get("paragraphs"):
            params["paragraphs"] = "true"
        if self.options.get("utterances"):
            params["utterances"] = "true"
        if self.options.get("profanity_filter"):
            params["profanity_filter"] = "true"
        if self.options.get("numerals"):
            params["numerals"] = "true"

        if enable_diarization:
            params["diarize"] = "true"
            # Request word-level timestamps for better speaker tracking
            params["words"] = "true"

        if language:
            params["language"] = language
        else:
            params["detect_language"] = "true"

        # Make API request
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }

        logger.info(f"Calling Deepgram API with params: {params}")

        try:
            response = requests.post(
                DEEPGRAM_API_URL,
                params=params,
                headers=headers,
                data=audio_bytes,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Deepgram API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise RuntimeError(f"Deepgram API error: {e}")

        # Debug: log raw response structure
        logger.debug(f"Deepgram raw response keys: {result.keys()}")
        if "results" in result:
            logger.debug(f"Results keys: {result['results'].keys()}")

            # Check word-level speakers
            channels = result["results"].get("channels", [])
            if channels:
                words = channels[0].get("alternatives", [{}])[0].get("words", [])
                if words:
                    speakers_in_words = set(w.get("speaker", -1) for w in words)
                    logger.info(f"Deepgram word-level speakers: {speakers_in_words}")

            if "utterances" in result["results"]:
                utterances = result["results"]["utterances"]
                logger.info(f"Deepgram returned {len(utterances)} utterances:")
                for i, utt in enumerate(utterances):
                    speaker = utt.get("speaker", "?")
                    text = utt.get("transcript", "")[:50]
                    logger.info(f"  Utterance {i}: speaker={speaker}, text='{text}...'")

        # Parse response
        return self._parse_response(result, enable_diarization)

    def _parse_response(
        self,
        result: dict,
        enable_diarization: bool
    ) -> TranscriptionResult:
        """Parse Deepgram API response into TranscriptionResult."""
        segments = []
        full_text_parts = []

        # Get the first channel's results
        channels = result.get("results", {}).get("channels", [])
        if not channels:
            return TranscriptionResult(segments=[], full_text="")

        channel = channels[0]
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            return TranscriptionResult(segments=[], full_text="")

        alternative = alternatives[0]

        # Get detected language
        detected_language = result.get("results", {}).get("channels", [{}])[0].get(
            "detected_language", None
        )

        # If diarization is enabled, use utterances for speaker info
        if enable_diarization and "utterances" in result.get("results", {}):
            utterances = result["results"]["utterances"]
            for utt in utterances:
                speaker_id = utt.get("speaker", 0)
                seg = TranscriptionSegment(
                    text=utt.get("transcript", ""),
                    start=utt.get("start", 0),
                    end=utt.get("end", 0),
                    speaker=f"SPEAKER_{speaker_id:02d}",
                    confidence=utt.get("confidence")
                )
                segments.append(seg)
                full_text_parts.append(seg.text)
        else:
            # Use words to build segments
            words = alternative.get("words", [])
            if words:
                # Group words into segments by speaker or by gaps
                current_segment_words = []
                current_speaker = None
                segment_start = None

                for word in words:
                    word_speaker = word.get("speaker", 0) if enable_diarization else 0

                    if current_speaker is None:
                        current_speaker = word_speaker
                        segment_start = word.get("start", 0)

                    # New speaker or significant gap = new segment
                    if word_speaker != current_speaker:
                        if current_segment_words:
                            seg = TranscriptionSegment(
                                text=" ".join(w.get("word", "") for w in current_segment_words),
                                start=segment_start,
                                end=current_segment_words[-1].get("end", 0),
                                speaker=f"SPEAKER_{current_speaker:02d}" if enable_diarization else None,
                                confidence=sum(w.get("confidence", 0) for w in current_segment_words) / len(current_segment_words)
                            )
                            segments.append(seg)
                            full_text_parts.append(seg.text)

                        current_segment_words = [word]
                        current_speaker = word_speaker
                        segment_start = word.get("start", 0)
                    else:
                        current_segment_words.append(word)

                # Don't forget the last segment
                if current_segment_words:
                    seg = TranscriptionSegment(
                        text=" ".join(w.get("word", "") for w in current_segment_words),
                        start=segment_start,
                        end=current_segment_words[-1].get("end", 0),
                        speaker=f"SPEAKER_{current_speaker:02d}" if enable_diarization else None,
                        confidence=sum(w.get("confidence", 0) for w in current_segment_words) / len(current_segment_words)
                    )
                    segments.append(seg)
                    full_text_parts.append(seg.text)
            else:
                # Fallback: use transcript directly
                transcript = alternative.get("transcript", "")
                if transcript:
                    segments.append(TranscriptionSegment(
                        text=transcript,
                        start=0,
                        end=result.get("metadata", {}).get("duration", 0),
                        speaker="SPEAKER_00" if enable_diarization else None
                    ))
                    full_text_parts.append(transcript)

        # Get duration from metadata
        duration = result.get("metadata", {}).get("duration")

        return TranscriptionResult(
            segments=segments,
            full_text=" ".join(full_text_parts),
            language=detected_language,
            duration=duration
        )
