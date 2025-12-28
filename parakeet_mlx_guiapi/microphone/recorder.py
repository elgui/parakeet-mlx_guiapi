"""
Microphone recording module for Parakeet-MLX GUI and API.

This module provides the MicrophoneRecorder class for recording audio from the microphone.
"""

import os
import tempfile
import threading
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


class MicrophoneRecorder:
    """
    Records audio from the microphone.

    Audio is recorded at 16kHz mono to match the transcriber's expected format.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the microphone recorder.

        Parameters:
        - sample_rate: Sample rate in Hz (default: 16000 to match transcriber)
        - channels: Number of audio channels (default: 1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._frames = []

    def _get_default_device_info(self) -> dict:
        """Get information about the default input device."""
        try:
            device_info = sd.query_devices(kind='input')
            return device_info
        except Exception as e:
            raise RuntimeError(f"No microphone found: {e}")

    def record_until_keypress(self) -> str:
        """
        Record audio from the microphone until Enter is pressed.

        Returns:
        - Path to the recorded WAV file
        """
        # Check for microphone
        device_info = self._get_default_device_info()
        print(f"Using microphone: {device_info['name']}")

        self._frames = []
        self._recording = True

        # Start recording in a separate thread
        def callback(indata, frames, time, status):
            if status:
                print(f"Recording status: {status}")
            if self._recording:
                self._frames.append(indata.copy())

        print("Recording... Press Enter to stop.")

        # Start the stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=callback
        ):
            # Wait for Enter key
            input()

        self._recording = False
        print("Recording stopped.")

        if not self._frames:
            raise RuntimeError("No audio recorded")

        # Concatenate all frames
        audio_data = np.concatenate(self._frames, axis=0)

        # Convert to int16 for WAV file
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            prefix='parakeet_mic_'
        )
        temp_path = temp_file.name
        temp_file.close()

        # Write WAV file
        wavfile.write(temp_path, self.sample_rate, audio_int16)

        duration = len(audio_data) / self.sample_rate
        print(f"Recorded {duration:.2f} seconds of audio")
        print(f"Saved to: {temp_path}")

        return temp_path

    def record_for_duration(self, duration_seconds: float) -> str:
        """
        Record audio from the microphone for a specific duration.

        Parameters:
        - duration_seconds: Duration to record in seconds

        Returns:
        - Path to the recorded WAV file
        """
        # Check for microphone
        device_info = self._get_default_device_info()
        print(f"Using microphone: {device_info['name']}")

        print(f"Recording for {duration_seconds} seconds...")

        # Record audio
        audio_data = sd.rec(
            int(duration_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()

        print("Recording stopped.")

        # Convert to int16 for WAV file
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            prefix='parakeet_mic_'
        )
        temp_path = temp_file.name
        temp_file.close()

        # Write WAV file
        wavfile.write(temp_path, self.sample_rate, audio_int16)

        print(f"Recorded {duration_seconds:.2f} seconds of audio")
        print(f"Saved to: {temp_path}")

        return temp_path
