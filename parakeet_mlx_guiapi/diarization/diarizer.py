"""
Speaker diarization using pyannote.audio.

This module provides speaker identification ("who spoke when") for audio files.
Requires a HuggingFace token with access to pyannote models.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a specific speaker."""
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""
    segments: List[SpeakerSegment]
    num_speakers: int

    def get_speaker_at_time(self, time: float) -> Optional[str]:
        """Get the speaker at a specific time."""
        for seg in self.segments:
            if seg.start <= time <= seg.end:
                return seg.speaker
        return None

    def merge_with_transcription(self, transcription_segments: List[dict]) -> List[dict]:
        """
        Merge diarization with transcription segments.

        Each transcription segment should have 'start', 'end', and 'text' keys.
        Returns segments with added 'speaker' key.
        """
        result = []
        for seg in transcription_segments:
            start = seg.get("start", seg.get("Start (s)", 0))
            end = seg.get("end", seg.get("End (s)", 0))
            mid_time = (start + end) / 2

            speaker = self.get_speaker_at_time(mid_time)
            if speaker is None:
                # Try to find closest speaker
                speaker = self._find_closest_speaker(mid_time)

            result.append({
                **seg,
                "speaker": speaker or "Unknown"
            })
        return result

    def _find_closest_speaker(self, time: float) -> Optional[str]:
        """Find the closest speaker to a given time."""
        if not self.segments:
            return None

        closest = min(self.segments,
                     key=lambda s: min(abs(s.start - time), abs(s.end - time)))
        return closest.speaker

    def format_transcript(self, transcription_segments: List[dict]) -> str:
        """
        Format transcription with speaker labels.

        Returns formatted text like:
        Speaker 1: Hello, how are you?
        Speaker 2: I'm doing great, thanks!
        """
        merged = self.merge_with_transcription(transcription_segments)

        lines = []
        current_speaker = None
        current_text = []

        for seg in merged:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", seg.get("Segment", ""))

            if speaker != current_speaker:
                if current_text:
                    lines.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = speaker
                current_text = [text]
            else:
                current_text.append(text)

        # Don't forget the last speaker
        if current_text:
            lines.append(f"{current_speaker}: {' '.join(current_text)}")

        return "\n\n".join(lines)

    def format_transcript_markdown(self, transcription_segments: List[dict]) -> str:
        """
        Format transcription with speaker labels using clean markdown.

        Returns formatted text with clear speaker differentiation:
        ---
        **Speaker 1**
        Hello, how are you?

        ---
        **Speaker 2**
        I'm doing great, thanks!
        """
        merged = self.merge_with_transcription(transcription_segments)

        lines = []
        current_speaker = None
        current_text = []

        for seg in merged:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", seg.get("Segment", ""))

            if speaker != current_speaker:
                if current_text:
                    # Format previous speaker's text
                    speaker_label = current_speaker.replace("SPEAKER_", "Speaker ")
                    lines.append(f"---\n**{speaker_label}**\n{' '.join(current_text)}")
                current_speaker = speaker
                current_text = [text]
            else:
                current_text.append(text)

        # Don't forget the last speaker
        if current_text:
            speaker_label = current_speaker.replace("SPEAKER_", "Speaker ")
            lines.append(f"---\n**{speaker_label}**\n{' '.join(current_text)}")

        return "\n\n".join(lines)


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio.

    Requires:
    - pyannote.audio >= 3.1.0
    - HuggingFace token with access to pyannote models

    To get access:
    1. Create a HuggingFace account at https://huggingface.co
    2. Accept the user agreements at:
       - https://huggingface.co/pyannote/speaker-diarization-3.1
       - https://huggingface.co/pyannote/segmentation-3.0
    3. Create an access token at https://huggingface.co/settings/tokens
    4. Set the token via:
       - Environment variable: HUGGINGFACE_TOKEN
       - Or pass to __init__: SpeakerDiarizer(hf_token="your_token")
    """

    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        """
        Initialize the diarizer.

        Args:
            hf_token: HuggingFace access token. If None, reads from
                      config file or HUGGINGFACE_TOKEN/HF_TOKEN environment variable.
            device: Device to use - "auto", "mps", "cpu".
                    Note: MPS support is experimental.
        """
        self.hf_token = hf_token or self._get_token_from_config() or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        self.device = device
        self.pipeline = None
        self._initialized = False

    @staticmethod
    def _get_token_from_config() -> Optional[str]:
        """Try to read HuggingFace token from config file."""
        try:
            import json
            from pathlib import Path
            config_path = Path.home() / ".parakeet_mlx_guiapi.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    return config.get("huggingface_token")
        except Exception:
            pass
        return None

    def _ensure_initialized(self):
        """Lazy initialization of the pipeline."""
        if self._initialized:
            return

        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required for speaker diarization.\n"
                "Set HUGGINGFACE_TOKEN environment variable or pass hf_token parameter.\n"
                "Get your token at: https://huggingface.co/settings/tokens\n"
                "Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        try:
            from pyannote.audio import Pipeline
            import torch
            # Note: pyannote Audio class is patched in __init__.py to use soundfile backend
        except ImportError:
            raise ImportError(
                "pyannote.audio is required for speaker diarization.\n"
                "Install with: pip install pyannote.audio>=3.1.0"
            )

        # Fix for PyTorch 2.6+ weights_only default change
        # pyannote checkpoints contain custom classes that need to be allowlisted
        safe_classes = []
        try:
            from torch.torch_version import TorchVersion
            safe_classes.append(TorchVersion)
        except Exception:
            pass
        try:
            from pyannote.audio.core.task import Specifications, Problem, Resolution
            safe_classes.extend([Specifications, Problem, Resolution])
        except Exception:
            pass
        try:
            from pyannote.audio.core.model import Introspection
            safe_classes.append(Introspection)
        except Exception:
            pass
        if safe_classes:
            torch.serialization.add_safe_globals(safe_classes)

        print("Loading speaker diarization model...")

        # Set HF_TOKEN env var - this is the most reliable way to authenticate
        # as the parameter name changes between pyannote/huggingface_hub versions
        os.environ["HF_TOKEN"] = self.hf_token

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )

        # Set device - prefer CPU on Apple Silicon for stability
        # MPS support in pyannote is experimental and can cause issues
        if self.device == "auto":
            # Check for Apple Silicon
            import platform
            is_apple_silicon = (
                platform.system() == "Darwin" and
                platform.machine() == "arm64"
            )

            if is_apple_silicon:
                # Use CPU for reliability on Apple Silicon
                # MPS has known issues with some pyannote operations
                print("Apple Silicon detected - using CPU for stability")
                self.pipeline.to(torch.device("cpu"))
                self._device_used = "cpu"
            elif torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                self._device_used = "cuda"
            else:
                self.pipeline.to(torch.device("cpu"))
                self._device_used = "cpu"
        elif self.device == "mps":
            # User explicitly requested MPS
            if torch.backends.mps.is_available():
                warnings.warn(
                    "Using MPS (Apple Silicon GPU). "
                    "This is experimental - if you encounter issues, use device='cpu'"
                )
                self.pipeline.to(torch.device("mps"))
                self._device_used = "mps"
            else:
                print("MPS not available, falling back to CPU")
                self.pipeline.to(torch.device("cpu"))
                self._device_used = "cpu"
        else:
            self.pipeline.to(torch.device("cpu"))
            self._device_used = "cpu"

        self._initialized = True
        print(f"Speaker diarization model loaded (device: {self._device_used})")

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum expected number of speakers
            max_speakers: Maximum expected number of speakers

        Returns:
            DiarizationResult with speaker segments
        """
        self._ensure_initialized()

        print(f"Diarizing audio: {audio_path}")

        # Run diarization
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        diarization = self.pipeline(audio_path, **kwargs)

        # Convert to our format
        segments = []
        speakers = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end
            ))
            speakers.add(speaker)

        # Sort by start time
        segments.sort(key=lambda s: s.start)

        print(f"Found {len(speakers)} speakers in {len(segments)} segments")

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers)
        )

    @staticmethod
    def is_available() -> Tuple[bool, str]:
        """
        Check if diarization is available.

        Returns:
            Tuple of (is_available, message)
        """
        # Check pyannote
        try:
            import pyannote.audio
            pyannote_ok = True
        except ImportError:
            return False, "pyannote.audio not installed. Run: pip install pyannote.audio>=3.1.0"

        # Check token (config file or env)
        token = SpeakerDiarizer._get_token_from_config() or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            return False, (
                "HuggingFace token not set.\n"
                "Use the Quick Setup in Settings > Speaker Diarization"
            )

        return True, "Diarization available"
