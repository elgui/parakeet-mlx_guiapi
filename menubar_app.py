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
import json
from datetime import datetime
from pathlib import Path

import rumps
import pyperclip

# Add current dir for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from parakeet_mlx_guiapi.utils.config import get_config, save_config


# Available models (from mlx-community on HuggingFace)
AVAILABLE_MODELS = [
    {
        "id": "mlx-community/parakeet-tdt-0.6b-v3",
        "name": "Parakeet TDT 0.6B v3",
        "description": "Fast, good accuracy (recommended)",
        "size": "~600MB"
    },
    {
        "id": "mlx-community/parakeet-tdt-1.1b-v2",
        "name": "Parakeet TDT 1.1B v2",
        "description": "Slower, better accuracy",
        "size": "~1.1GB"
    },
    {
        "id": "mlx-community/parakeet-ctc-0.6b-v2",
        "name": "Parakeet CTC 0.6B v2",
        "description": "CTC model, different architecture",
        "size": "~600MB"
    },
    {
        "id": "mlx-community/parakeet-ctc-1.1b",
        "name": "Parakeet CTC 1.1B",
        "description": "Larger CTC model",
        "size": "~1.1GB"
    },
]


class ParakeetMenuBarApp(rumps.App):
    """Menu bar app for voice-to-clipboard transcription."""

    # Icons
    ICON_IDLE = "🎤"
    ICON_RECORDING = "🔴"
    ICON_PROCESSING = "⏳"
    ICON_READY = "✅"
    ICON_ERROR = "❌"

    def __init__(self):
        super().__init__(
            name="Parakeet",
            title=self.ICON_IDLE,
            quit_button=None  # We'll add our own quit button
        )

        self.recording = False
        self.processing = False
        self.transcriber = None
        self._stream = None
        self._audio_data = []
        self._recording_start_time = None
        self._timer = None

        # Load config
        self.config = get_config()

        # History of transcriptions (last 10)
        self.history = []
        self._load_history()

        # Build menu - rumps requires menu items to be created here
        self._setup_menu()

        # Lazy-load transcriber in background
        threading.Thread(target=self._init_transcriber, daemon=True).start()

    def _setup_menu(self):
        """Set up the initial menu structure."""
        # Record button (main action)
        self.record_button = rumps.MenuItem(
            "Start Recording",
            callback=self.toggle_recording
        )

        # Status display
        self.status_item = rumps.MenuItem("Status: Loading model...")

        # Model selection submenu - build items directly
        self.model_menu = rumps.MenuItem("Model")
        self._populate_model_menu()

        # Settings submenu
        self.settings_menu = rumps.MenuItem("Settings")
        self._populate_settings_menu()

        # History submenu
        self.history_menu = rumps.MenuItem("History")
        self._populate_history_menu()

        # About and Quit
        about_item = rumps.MenuItem("About Parakeet", callback=self.show_about)
        quit_item = rumps.MenuItem("Quit Parakeet", callback=self.quit_app)

        self.menu = [
            self.record_button,
            None,  # Separator
            self.status_item,
            None,
            self.model_menu,
            self.settings_menu,
            self.history_menu,
            None,
            about_item,
            quit_item,
        ]

    def _populate_model_menu(self):
        """Populate the model selection menu."""
        current_model = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])

        for model in AVAILABLE_MODELS:
            # Create menu item with checkmark for current model
            title = model["name"]
            if model["id"] == current_model:
                title = f"✓ {title}"

            item = rumps.MenuItem(
                title,
                callback=lambda sender, m=model: self.select_model(m)
            )
            self.model_menu.add(item)

        # Add separator and info
        self.model_menu.add(None)
        info_item = rumps.MenuItem(f"Size: {self._get_model_size(current_model)}")
        self.model_menu.add(info_item)

    def _get_model_short_name(self, model_id):
        """Get short display name for a model ID."""
        for model in AVAILABLE_MODELS:
            if model["id"] == model_id:
                return model["name"]
        return model_id.split("/")[-1]

    def _get_model_size(self, model_id):
        """Get model size for display."""
        for model in AVAILABLE_MODELS:
            if model["id"] == model_id:
                return model["size"]
        return "Unknown"

    def _populate_settings_menu(self):
        """Populate the settings submenu."""
        # Chunk duration options
        chunk_menu = rumps.MenuItem("Chunk Duration")
        chunk_options = [30, 60, 120, 180, 300]
        current_chunk = self.config.get("default_chunk_duration", 120)

        for duration in chunk_options:
            title = f"{duration}s"
            if duration == current_chunk:
                title = f"✓ {title}"
            item = rumps.MenuItem(
                title,
                callback=lambda sender, d=duration: self.set_chunk_duration(d)
            )
            chunk_menu.add(item)

        self.settings_menu.add(chunk_menu)

        # Auto-copy to clipboard toggle
        auto_copy = self.config.get("auto_copy_clipboard", True)
        copy_title = "✓ Auto-copy to Clipboard" if auto_copy else "Auto-copy to Clipboard"
        copy_item = rumps.MenuItem(copy_title, callback=self.toggle_auto_copy)
        self.settings_menu.add(copy_item)

        # Show notifications toggle
        show_notif = self.config.get("show_notifications", True)
        notif_title = "✓ Show Notifications" if show_notif else "Show Notifications"
        notif_item = rumps.MenuItem(notif_title, callback=self.toggle_notifications)
        self.settings_menu.add(notif_item)

        # Separator and config file location
        self.settings_menu.add(None)
        config_item = rumps.MenuItem("Config: ~/.parakeet_mlx_guiapi.json")
        self.settings_menu.add(config_item)

    def _populate_history_menu(self):
        """Populate the history submenu."""
        if not self.history:
            empty_item = rumps.MenuItem("No transcriptions yet")
            self.history_menu.add(empty_item)
        else:
            for i, entry in enumerate(self.history[:10]):
                # Truncate text for menu display
                text = entry.get("text", "")[:50]
                if len(entry.get("text", "")) > 50:
                    text += "..."
                timestamp = entry.get("timestamp", "")

                item = rumps.MenuItem(
                    f"{timestamp}: {text}",
                    callback=lambda sender, e=entry: self.copy_history_item(e)
                )
                self.history_menu.add(item)

            # Clear history option
            self.history_menu.add(None)
            clear_item = rumps.MenuItem("Clear History", callback=self.clear_history)
            self.history_menu.add(clear_item)

    def _refresh_model_menu(self):
        """Refresh the model menu after a change."""
        # Remove all items
        keys = list(self.model_menu.keys())
        for key in keys:
            del self.model_menu[key]
        # Re-populate
        self._populate_model_menu()

    def _refresh_settings_menu(self):
        """Refresh settings menu after a change."""
        keys = list(self.settings_menu.keys())
        for key in keys:
            del self.settings_menu[key]
        self._populate_settings_menu()

    def _refresh_history_menu(self):
        """Refresh history menu."""
        keys = list(self.history_menu.keys())
        for key in keys:
            del self.history_menu[key]
        self._populate_history_menu()

    def _load_history(self):
        """Load transcription history from file."""
        history_path = Path.home() / ".parakeet_history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def _save_history(self):
        """Save transcription history to file."""
        history_path = Path.home() / ".parakeet_history.json"
        try:
            with open(history_path, "w") as f:
                json.dump(self.history[:20], f)  # Keep last 20
        except Exception:
            pass

    def _add_to_history(self, text, duration):
        """Add a transcription to history."""
        entry = {
            "text": text,
            "duration": f"{duration:.1f}s",
            "timestamp": datetime.now().strftime("%H:%M"),
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
        self.history.insert(0, entry)
        self.history = self.history[:20]  # Keep last 20
        self._save_history()
        self._refresh_history_menu()

    def _init_transcriber(self):
        """Initialize transcriber in background."""
        try:
            from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber

            model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
            self.status_item.title = f"Loading {self._get_model_short_name(model_name)}..."

            self.transcriber = AudioTranscriber(model_name=model_name)

            self.status_item.title = f"Ready: {self._get_model_short_name(model_name)}"

            if self.config.get("show_notifications", True):
                rumps.notification(
                    title="Parakeet Ready",
                    subtitle=self._get_model_short_name(model_name),
                    message="Click the mic icon to record",
                    sound=False
                )
        except Exception as e:
            self.status_item.title = "Error: Model failed to load"
            rumps.notification(
                title="Parakeet Error",
                subtitle="Failed to load model",
                message=str(e)[:100],
                sound=True
            )

    def select_model(self, model):
        """Change the transcription model."""
        if self.recording or self.processing:
            rumps.notification(
                title="Cannot Change Model",
                subtitle="",
                message="Please wait until current operation completes",
                sound=False
            )
            return

        # Update config
        self.config["model_name"] = model["id"]
        save_config(self.config)

        # Update menu
        self._refresh_model_menu()

        # Reload transcriber
        self.transcriber = None
        self.status_item.title = f"Switching to {model['name']}..."
        threading.Thread(target=self._init_transcriber, daemon=True).start()

        rumps.notification(
            title="Model Changed",
            subtitle=model["name"],
            message=f"Loading {model['description']}...",
            sound=False
        )

    def set_chunk_duration(self, duration):
        """Set chunk duration for long audio processing."""
        self.config["default_chunk_duration"] = duration
        save_config(self.config)
        self._refresh_settings_menu()

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Setting Updated",
                subtitle="Chunk Duration",
                message=f"Set to {duration} seconds",
                sound=False
            )

    def toggle_auto_copy(self, _):
        """Toggle auto-copy to clipboard."""
        current = self.config.get("auto_copy_clipboard", True)
        self.config["auto_copy_clipboard"] = not current
        save_config(self.config)
        self._refresh_settings_menu()

    def toggle_notifications(self, _):
        """Toggle notification display."""
        current = self.config.get("show_notifications", True)
        self.config["show_notifications"] = not current
        save_config(self.config)
        self._refresh_settings_menu()

    def copy_history_item(self, entry):
        """Copy a history item to clipboard."""
        pyperclip.copy(entry.get("text", ""))
        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Copied to Clipboard",
                subtitle="",
                message=entry.get("text", "")[:80],
                sound=False
            )

    def clear_history(self, _):
        """Clear transcription history."""
        self.history = []
        self._save_history()
        self._refresh_history_menu()

    def toggle_recording(self, _):
        """Toggle recording state."""
        if self.processing:
            if self.config.get("show_notifications", True):
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
            self._recording_start_time = time.time()
            self.title = self.ICON_RECORDING
            self.record_button.title = "⏹ Stop Recording"
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

            # Start timer to update title
            self._start_recording_timer()

            if self.config.get("show_notifications", True):
                rumps.notification(
                    title="Recording Started",
                    subtitle="",
                    message="Click the icon again to stop",
                    sound=False
                )

        except Exception as e:
            self.recording = False
            self.title = self.ICON_ERROR
            self.record_button.title = "Start Recording"
            rumps.notification(
                title="Recording Error",
                subtitle="",
                message=str(e)[:100],
                sound=True
            )
            # Reset icon after a moment
            threading.Timer(2.0, lambda: setattr(self, 'title', self.ICON_IDLE)).start()

    def _start_recording_timer(self):
        """Start a timer to update recording duration in title."""
        def update_title():
            while self.recording:
                elapsed = time.time() - self._recording_start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                self.title = f"🔴 {mins}:{secs:02d}"
                time.sleep(1)

        self._timer = threading.Thread(target=update_title, daemon=True)
        self._timer.start()

    def stop_recording(self):
        """Stop recording and start transcription."""
        import numpy as np
        from scipy.io import wavfile

        self.recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

        if not self._audio_data:
            self.title = self.ICON_IDLE
            self.record_button.title = "Start Recording"
            rumps.notification(
                title="No Audio",
                subtitle="",
                message="No audio was recorded",
                sound=True
            )
            return

        # Calculate duration
        recording_duration = time.time() - self._recording_start_time

        # Update UI for processing
        self.processing = True
        self.title = self.ICON_PROCESSING
        self.record_button.title = "Processing..."
        self.status_item.title = "Transcribing..."

        # Process in background thread
        threading.Thread(
            target=self._process_audio,
            args=(recording_duration,),
            daemon=True
        ).start()

    def _process_audio(self, recording_duration):
        """Process recorded audio and transcribe."""
        import numpy as np
        from scipy.io import wavfile

        try:
            # Concatenate audio data
            audio_data = np.concatenate(self._audio_data, axis=0)

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
            wait_count = 0
            while self.transcriber is None and wait_count < 60:
                time.sleep(0.5)
                wait_count += 1

            if self.transcriber is None:
                raise Exception("Model not loaded. Please wait and try again.")

            # Transcribe
            chunk_duration = self.config.get("default_chunk_duration", 120)
            df, full_text = self.transcriber.transcribe(
                temp_path,
                chunk_duration=chunk_duration
            )

            # Clean up temp file
            os.remove(temp_path)

            if full_text:
                # Copy to clipboard if enabled
                if self.config.get("auto_copy_clipboard", True):
                    pyperclip.copy(full_text)

                # Add to history
                self._add_to_history(full_text, recording_duration)

                # Show notification with preview
                if self.config.get("show_notifications", True):
                    preview = full_text[:80] + "..." if len(full_text) > 80 else full_text
                    copied_msg = " - Copied!" if self.config.get("auto_copy_clipboard", True) else ""
                    rumps.notification(
                        title=f"Transcription Complete{copied_msg}",
                        subtitle=f"{recording_duration:.1f}s of audio",
                        message=preview,
                        sound=True
                    )

                # Flash success icon
                self.title = self.ICON_READY
                threading.Timer(2.0, lambda: setattr(self, 'title', self.ICON_IDLE)).start()
            else:
                if self.config.get("show_notifications", True):
                    rumps.notification(
                        title="Transcription Empty",
                        subtitle="",
                        message="No speech detected in the recording",
                        sound=True
                    )
                self.title = self.ICON_IDLE

        except Exception as e:
            rumps.notification(
                title="Transcription Error",
                subtitle="",
                message=str(e)[:100],
                sound=True
            )
            self.title = self.ICON_ERROR
            threading.Timer(2.0, lambda: setattr(self, 'title', self.ICON_IDLE)).start()
        finally:
            # Reset UI
            self.processing = False
            self.record_button.title = "Start Recording"
            model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
            self.status_item.title = f"Ready: {self._get_model_short_name(model_name)}"

    def show_about(self, _):
        """Show about dialog."""
        model_name = self._get_model_short_name(
            self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
        )
        rumps.alert(
            title="Parakeet Voice-to-Clipboard",
            message=(
                "Quick voice transcription for macOS.\n\n"
                "Usage:\n"
                "• Click the mic icon to start recording\n"
                "• Click again to stop and transcribe\n"
                "• Transcription is copied to clipboard\n\n"
                f"Current model: {model_name}\n\n"
                "Powered by parakeet-mlx\n"
                "https://github.com/senstella/parakeet-mlx"
            )
        )

    def quit_app(self, _):
        """Quit the application."""
        if self.recording:
            self.recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
        rumps.quit_application()


def main():
    """Run the menu bar app."""
    ParakeetMenuBarApp().run()


if __name__ == "__main__":
    main()
