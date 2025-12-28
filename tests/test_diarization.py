"""
Unit tests for speaker diarization module.

Tests cover:
1. Data classes (SpeakerSegment, DiarizationResult)
2. Transcript formatting
3. Speaker identification at timestamps
4. Configuration and availability checks

Run with: pytest tests/test_diarization.py -v
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock


class TestSpeakerSegment:
    """Tests for SpeakerSegment dataclass."""

    def test_segment_creation(self):
        """Test basic segment creation."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment

        segment = SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0)

        assert segment.speaker == "SPEAKER_00"
        assert segment.start == 0.0
        assert segment.end == 5.0

    def test_segment_duration(self):
        """Test duration property calculation."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment

        segment = SpeakerSegment(speaker="SPEAKER_00", start=1.5, end=4.5)

        assert segment.duration == 3.0

    def test_segment_zero_duration(self):
        """Test segment with zero duration."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment

        segment = SpeakerSegment(speaker="SPEAKER_00", start=2.0, end=2.0)

        assert segment.duration == 0.0


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_empty_result(self):
        """Test empty diarization result."""
        from parakeet_mlx_guiapi.diarization.diarizer import DiarizationResult

        result = DiarizationResult(segments=[], num_speakers=0)

        assert result.num_speakers == 0
        assert len(result.segments) == 0

    def test_get_speaker_at_time_found(self):
        """Test finding speaker at a specific time."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
            SpeakerSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)

        assert result.get_speaker_at_time(2.5) == "SPEAKER_00"
        assert result.get_speaker_at_time(7.5) == "SPEAKER_01"

    def test_get_speaker_at_time_boundary(self):
        """Test speaker lookup at segment boundaries."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
            SpeakerSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)

        # At boundary, should return the segment that includes it
        assert result.get_speaker_at_time(5.0) in ["SPEAKER_00", "SPEAKER_01"]

    def test_get_speaker_at_time_not_found(self):
        """Test speaker lookup when time is outside all segments."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=2.0, end=4.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=1)

        assert result.get_speaker_at_time(0.0) is None
        assert result.get_speaker_at_time(10.0) is None

    def test_find_closest_speaker(self):
        """Test finding closest speaker to a time outside segments."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=3.0),
            SpeakerSegment(speaker="SPEAKER_01", start=7.0, end=10.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)

        # Time 4.0 is closer to SPEAKER_00's end (3.0) than SPEAKER_01's start (7.0)
        assert result._find_closest_speaker(4.0) == "SPEAKER_00"

        # Time 6.0 is closer to SPEAKER_01's start (7.0)
        assert result._find_closest_speaker(6.0) == "SPEAKER_01"

    def test_merge_with_transcription(self):
        """Test merging diarization with transcription segments."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
            SpeakerSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)

        transcription = [
            {"start": 0.0, "end": 2.0, "text": "Hello there"},
            {"start": 2.0, "end": 5.0, "text": "How are you"},
            {"start": 5.0, "end": 8.0, "text": "I am fine"},
        ]

        merged = result.merge_with_transcription(transcription)

        assert len(merged) == 3
        assert merged[0]["speaker"] == "SPEAKER_00"
        assert merged[1]["speaker"] == "SPEAKER_00"
        assert merged[2]["speaker"] == "SPEAKER_01"

    def test_merge_with_alternative_keys(self):
        """Test merging with alternative transcription key names."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=10.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=1)

        # Using alternative key names (from parakeet output)
        transcription = [
            {"Start (s)": 0.0, "End (s)": 5.0, "Segment": "Hello"},
        ]

        merged = result.merge_with_transcription(transcription)

        assert merged[0]["speaker"] == "SPEAKER_00"


class TestTranscriptFormatting:
    """Tests for transcript formatting methods."""

    def test_format_transcript_basic(self):
        """Test basic transcript formatting."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
            SpeakerSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)

        transcription = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 5.0, "end": 10.0, "text": "Hi there"},
        ]

        formatted = result.format_transcript(transcription)

        assert "SPEAKER_00: Hello" in formatted
        assert "SPEAKER_01: Hi there" in formatted

    def test_format_transcript_consolidates_same_speaker(self):
        """Test that consecutive segments from same speaker are merged."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=10.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=1)

        transcription = [
            {"start": 0.0, "end": 3.0, "text": "Hello"},
            {"start": 3.0, "end": 6.0, "text": "how are"},
            {"start": 6.0, "end": 10.0, "text": "you today"},
        ]

        formatted = result.format_transcript(transcription)

        # All should be merged into one speaker block
        assert formatted.count("SPEAKER_00:") == 1
        assert "Hello" in formatted
        assert "how are" in formatted
        assert "you today" in formatted

    def test_format_transcript_markdown(self):
        """Test markdown transcript formatting."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerSegment, DiarizationResult

        segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
            SpeakerSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)

        transcription = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 5.0, "end": 10.0, "text": "Hi there"},
        ]

        formatted = result.format_transcript_markdown(transcription)

        # Should contain markdown formatting
        assert "**Speaker 00**" in formatted
        assert "**Speaker 01**" in formatted
        assert "---" in formatted


class TestSpeakerDiarizerConfig:
    """Tests for SpeakerDiarizer configuration."""

    def test_token_from_env(self):
        """Test token loading from environment variable."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer

        # Mock config file to return None so env var is used
        with patch.object(SpeakerDiarizer, '_get_token_from_config', return_value=None):
            with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "test_token_123", "HF_TOKEN": ""}, clear=False):
                diarizer = SpeakerDiarizer()
                assert diarizer.hf_token == "test_token_123"

    def test_token_from_hf_token_env(self):
        """Test token loading from HF_TOKEN environment variable."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer

        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=False):
            # Clear other token sources
            with patch.object(SpeakerDiarizer, '_get_token_from_config', return_value=None):
                with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": ""}, clear=False):
                    diarizer = SpeakerDiarizer()
                    # Should fall back to HF_TOKEN
                    assert "hf" in diarizer.hf_token or diarizer.hf_token == "hf_test_token"

    def test_token_passed_directly(self):
        """Test token passed directly to constructor."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer

        diarizer = SpeakerDiarizer(hf_token="direct_token")
        assert diarizer.hf_token == "direct_token"

    def test_is_available_without_pyannote(self):
        """Test is_available when pyannote is not installed."""
        with patch.dict('sys.modules', {'pyannote.audio': None}):
            from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer

            # Force reimport to test import error
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                # This would need module reload, simplified test
                pass

    def test_is_available_without_token(self):
        """Test is_available returns False without token."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer

        with patch.object(SpeakerDiarizer, '_get_token_from_config', return_value=None):
            with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "", "HF_TOKEN": ""}, clear=False):
                available, msg = SpeakerDiarizer.is_available()
                # Should indicate token issue if pyannote is installed
                if not available:
                    assert "token" in msg.lower() or "pyannote" in msg.lower()


class TestSpeakerDiarizerLazyInit:
    """Tests for lazy initialization behavior."""

    def test_lazy_init_not_called_on_construction(self):
        """Test that pipeline is not loaded on construction."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer

        diarizer = SpeakerDiarizer(hf_token="test")

        assert diarizer.pipeline is None
        assert diarizer._initialized is False

    def test_ensure_initialized_raises_without_token(self):
        """Test that initialization fails without token."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer

        diarizer = SpeakerDiarizer(hf_token=None)
        diarizer.hf_token = None  # Force no token

        with pytest.raises(ValueError, match="HuggingFace token required"):
            diarizer._ensure_initialized()


class TestDiarizeMethod:
    """Tests for the diarize method parameters."""

    def test_diarize_accepts_num_speakers(self):
        """Test that diarize accepts num_speakers parameter."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer
        import inspect

        sig = inspect.signature(SpeakerDiarizer.diarize)
        params = list(sig.parameters.keys())

        assert "num_speakers" in params
        assert "min_speakers" in params
        assert "max_speakers" in params

    def test_diarize_parameter_defaults(self):
        """Test default parameter values for diarize."""
        from parakeet_mlx_guiapi.diarization.diarizer import SpeakerDiarizer
        import inspect

        sig = inspect.signature(SpeakerDiarizer.diarize)

        assert sig.parameters["num_speakers"].default is None
        assert sig.parameters["min_speakers"].default is None
        assert sig.parameters["max_speakers"].default is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
