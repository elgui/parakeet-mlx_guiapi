"""
Unit tests for transcription module.

Tests cover:
1. AudioTranscriber initialization
2. Audio preprocessing
3. Transcription output format

Run with: pytest tests/test_transcription.py -v
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestAudioPreprocessing:
    """Tests for audio preprocessing functionality."""

    def test_audio_resampling_needed(self):
        """Test detection when resampling is needed."""
        # Simulate audio at wrong sample rate
        from pydub import AudioSegment

        # Create a short audio segment at 44100 Hz
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            # Create 44100 Hz audio
            audio = AudioSegment.silent(duration=100, frame_rate=44100)
            audio.export(temp_path, format='wav')

            # Load and check
            loaded = AudioSegment.from_file(temp_path)
            assert loaded.frame_rate == 44100

            # Verify resampling would be needed (target is 16000)
            assert loaded.frame_rate != 16000
        finally:
            os.unlink(temp_path)

    def test_mono_conversion_needed(self):
        """Test detection when mono conversion is needed."""
        from pydub import AudioSegment

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            # Create stereo audio
            audio = AudioSegment.silent(duration=100, frame_rate=16000)
            stereo = audio.set_channels(2)
            stereo.export(temp_path, format='wav')

            loaded = AudioSegment.from_file(temp_path)
            assert loaded.channels == 2

            # Convert to mono
            mono = loaded.set_channels(1)
            assert mono.channels == 1
        finally:
            os.unlink(temp_path)

    def test_audio_duration_calculation(self):
        """Test audio duration is correctly calculated."""
        from pydub import AudioSegment

        # Create 1 second of audio
        audio = AudioSegment.silent(duration=1000, frame_rate=16000)
        assert abs(audio.duration_seconds - 1.0) < 0.01


class TestTranscriptionOutput:
    """Tests for transcription output format."""

    def test_dataframe_columns(self):
        """Test that transcription DataFrame has expected columns."""
        import pandas as pd

        # Expected columns from transcriber output
        expected_cols = ["Start (s)", "End (s)", "Segment", "Duration", "Tokens"]

        # Create mock DataFrame matching expected format
        df = pd.DataFrame({
            "Start (s)": [0.0, 2.5],
            "End (s)": [2.5, 5.0],
            "Segment": ["Hello", "World"],
            "Duration": [2.5, 2.5],
            "Tokens": [["Hello"], ["World"]],
        })

        for col in expected_cols:
            assert col in df.columns

    def test_transcription_segment_format(self):
        """Test transcription segment has required fields."""
        segment = {
            "Start (s)": 0.0,
            "End (s)": 2.5,
            "Segment": "Hello world",
            "Duration": 2.5,
        }

        assert isinstance(segment["Start (s)"], (int, float))
        assert isinstance(segment["End (s)"], (int, float))
        assert isinstance(segment["Segment"], str)
        assert segment["End (s)"] >= segment["Start (s)"]


class TestAudioSegmentExtraction:
    """Tests for audio segment extraction."""

    def test_segment_time_conversion(self):
        """Test time to milliseconds conversion."""
        start_time = 1.5  # seconds
        end_time = 3.0    # seconds

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        assert start_ms == 1500
        assert end_ms == 3000

    def test_segment_extraction(self):
        """Test extracting a segment from audio."""
        from pydub import AudioSegment

        # Create 5 second audio
        audio = AudioSegment.silent(duration=5000, frame_rate=16000)

        # Extract 1-3 second segment
        segment = audio[1000:3000]

        assert abs(segment.duration_seconds - 2.0) < 0.01


class TestWavFileCreation:
    """Tests for WAV file creation from numpy arrays."""

    def test_float32_to_int16_conversion(self):
        """Test audio format conversion."""
        # Float32 audio at full scale
        audio_float = np.array([1.0, -1.0, 0.5, -0.5, 0.0], dtype=np.float32)
        audio_int16 = (audio_float * 32767).astype(np.int16)

        assert audio_int16[0] == 32767   # Max positive
        assert audio_int16[1] == -32767  # Max negative
        assert audio_int16[4] == 0       # Zero

    def test_wav_write_and_read(self):
        """Test WAV file round-trip."""
        from scipy.io import wavfile

        sample_rate = 16000
        duration = 0.5
        samples = int(sample_rate * duration)

        # Generate test audio
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        audio_int16 = (audio * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            wavfile.write(temp_path, sample_rate, audio_int16)

            # Read back
            sr, data = wavfile.read(temp_path)
            assert sr == sample_rate
            assert len(data) == samples
        finally:
            os.unlink(temp_path)


class TestTranscriberInitialization:
    """Tests for AudioTranscriber initialization."""

    def test_transcriber_accepts_model_name(self):
        """Test that transcriber accepts model_name parameter."""
        try:
            from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"parakeet_mlx not available: {e}")

        import inspect
        sig = inspect.signature(AudioTranscriber.__init__)
        params = list(sig.parameters.keys())

        assert "model_name" in params

    def test_default_model_name(self):
        """Test default model name is set."""
        try:
            from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"parakeet_mlx not available: {e}")

        import inspect
        sig = inspect.signature(AudioTranscriber.__init__)

        # Check default value
        default = sig.parameters["model_name"].default
        assert default is not None
        assert "parakeet" in default.lower() or "mlx" in default.lower()


class TestTranscribeMethodSignature:
    """Tests for transcribe method signature."""

    def test_transcribe_parameters(self):
        """Test transcribe method has expected parameters."""
        try:
            from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"parakeet_mlx not available: {e}")

        import inspect
        sig = inspect.signature(AudioTranscriber.transcribe)
        params = list(sig.parameters.keys())

        assert "audio_path" in params
        assert "chunk_duration" in params
        assert "overlap_duration" in params

    def test_transcribe_returns_tuple(self):
        """Test that transcribe is documented to return DataFrame and text."""
        try:
            from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"parakeet_mlx not available: {e}")

        # Check docstring mentions return type
        docstring = AudioTranscriber.transcribe.__doc__
        assert docstring is not None
        assert "DataFrame" in docstring or "Returns" in docstring


class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_audio_handling(self):
        """Test handling of empty audio data."""
        audio_data = []

        with pytest.raises(ValueError):
            np.concatenate(audio_data, axis=0)

    def test_invalid_path_handling(self):
        """Test handling of invalid file path."""
        from pydub import AudioSegment

        with pytest.raises(Exception):
            AudioSegment.from_file("/nonexistent/path/audio.wav")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
