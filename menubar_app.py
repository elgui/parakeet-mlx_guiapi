#!/usr/bin/env python3
"""
Parakeet Menu Bar App - Voice to Clipboard

A macOS menu bar app for quick voice transcription.
Click to start recording, click again to stop and copy to clipboard.
"""

import os
import sys
import threading
import tempfile
import time

import rumps
import pyperclip

# Add paths for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parakeet_path = os.path.join(parent_dir, 'parakeet-mlx')
sys.path.insert(0, current_dir)
sys.path.insert(0, parakeet_path)

from parakeet_mlx_guiapi.microphone import MicrophoneRecorder
from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber


class ParakeetMenuBarApp(rumps.App):
    """Menu bar app for voice-to-clipboard transcription."""

    # Icons (using emoji for simplicity - can be replaced with actual icons)
    ICON_IDLE = "🎤"
    ICON_RECORDING = "🔴"
    ICON_PROCESSING = "⏳"

    def __init__(self):
        super().__init__(
            name="Parakeet",
            title=self.ICON_IDLE,
            quit_button="Quit"
        )

        self.recording = False
        self.processing = False
        self.recorder = None
        self.transcriber = None
        self._recording_thread = None
        self._audio_data = []
        self._temp_file = None

        # Menu items
        self.record_button = rumps.MenuItem(
            "Start Recording",
            callback=self.toggle_recording
        )
        self.menu = [
            self.record_button,
            None,  # Separator
            rumps.MenuItem("About", callback=self.show_about),
        ]

        # Lazy-load transcriber in background
        threading.Thread(target=self._init_transcriber, daemon=True).start()

    def _init_transcriber(self):
        """Initialize transcriber in background (model loading can be slow)."""
        try:
            self.transcriber = AudioTranscriber()
            rumps.notification(
                title="Parakeet Ready",
                subtitle="",
                message="Voice transcription ready. Click the mic icon to record.",
                sound=False
            )
        except Exception as e:
            rumps.notification(
                title="Parakeet Error",
                subtitle="Failed to load model",
                message=str(e)[:100],
                sound=True
            )

    def toggle_recording(self, _):
        """Toggle recording state."""
        if self.processing:
            rumps.notification(
                title="Parakeet",
                subtitle="",
                message="Still processing previous recording...",
                sound=False
            )
            return

        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording from microphone."""
        try:
            import sounddevice as sd
            import numpy as np

            self.recording = True
            self.title = self.ICON_RECORDING
            self.record_button.title = "Stop Recording"
            self._audio_data = []

            # Recording parameters
            self.sample_rate = 16000
            self.channels = 1

            def audio_callback(indata, frames, time_info, status):
                if self.recording:
                    self._audio_data.append(indata.copy())

            # Start recording stream
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=audio_callback
            )
            self._stream.start()

            rumps.notification(
                title="Recording Started",
                subtitle="",
                message="Click the icon again to stop and transcribe.",
                sound=False
            )

        except Exception as e:
            self.recording = False
            self.title = self.ICON_IDLE
            self.record_button.title = "Start Recording"
            rumps.notification(
                title="Recording Error",
                subtitle="",
                message=str(e)[:100],
                sound=True
            )

    def stop_recording(self):
        """Stop recording and start transcription."""
        import numpy as np
        from scipy.io import wavfile

        self.recording = False
        self._stream.stop()
        self._stream.close()

        if not self._audio_data:
            self.title = self.ICON_IDLE
            self.record_button.title = "Start Recording"
            rumps.notification(
                title="No Audio",
                subtitle="",
                message="No audio was recorded.",
                sound=True
            )
            return

        # Update UI for processing
        self.processing = True
        self.title = self.ICON_PROCESSING
        self.record_button.title = "Processing..."

        # Process in background thread
        threading.Thread(target=self._process_audio, daemon=True).start()

    def _process_audio(self):
        """Process recorded audio and transcribe."""
        import numpy as np
        from scipy.io import wavfile

        try:
            # Concatenate audio data
            audio_data = np.concatenate(self._audio_data, axis=0)
            duration = len(audio_data) / self.sample_rate

            # Convert to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False,
                prefix='parakeet_menubar_'
            )
            temp_path = temp_file.name
            temp_file.close()
            wavfile.write(temp_path, self.sample_rate, audio_int16)

            # Wait for transcriber if not ready
            while self.transcriber is None:
                time.sleep(0.1)

            # Transcribe
            df, full_text = self.transcriber.transcribe(temp_path)

            # Clean up temp file
            os.remove(temp_path)

            if full_text:
                # Copy to clipboard
                pyperclip.copy(full_text)

                # Show notification with preview
                preview = full_text[:80] + "..." if len(full_text) > 80 else full_text
                rumps.notification(
                    title="Copied to Clipboard",
                    subtitle=f"{duration:.1f}s of audio",
                    message=preview,
                    sound=True
                )
            else:
                rumps.notification(
                    title="Transcription Empty",
                    subtitle="",
                    message="No speech detected in the recording.",
                    sound=True
                )

        except Exception as e:
            rumps.notification(
                title="Transcription Error",
                subtitle="",
                message=str(e)[:100],
                sound=True
            )
        finally:
            # Reset UI
            self.processing = False
            self.title = self.ICON_IDLE
            self.record_button.title = "Start Recording"

    def show_about(self, _):
        """Show about dialog."""
        rumps.alert(
            title="Parakeet Voice-to-Clipboard",
            message=(
                "Quick voice transcription for macOS.\n\n"
                "Click the microphone icon to start recording.\n"
                "Click again to stop and copy transcription to clipboard.\n\n"
                "Powered by Parakeet-MLX"
            )
        )


def main():
    """Run the menu bar app."""
    ParakeetMenuBarApp().run()


if __name__ == "__main__":
    main()
