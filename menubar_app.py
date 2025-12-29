#!/usr/bin/env python3
"""
Parakeet Menu Bar App - Voice to Clipboard

A macOS menu bar app for quick voice transcription.
Click to start recording, click again to stop and copy to clipboard.
"""

import os
import sys
import builtins

# Fix UTF-8 encoding issues for macOS GUI apps
# parakeet_mlx opens files without encoding specified, and macOS GUI apps
# don't inherit UTF-8 locale from terminal, defaulting to ASCII
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

# Fix PATH for macOS GUI apps - they don't inherit shell PATH
# Add common Homebrew paths where ffmpeg is typically installed
homebrew_paths = [
    '/opt/homebrew/bin',  # Apple Silicon
    '/usr/local/bin',     # Intel Mac
]
current_path = os.environ.get('PATH', '')
for path in homebrew_paths:
    if path not in current_path and os.path.isdir(path):
        os.environ['PATH'] = f"{path}:{current_path}"
        current_path = os.environ['PATH']

# Monkey-patch open() to default to UTF-8 for text mode
_original_open = builtins.open

def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None,
               newline=None, closefd=True, opener=None):
    """Wrapper around open() that defaults to UTF-8 encoding for text mode."""
    if encoding is None and 'b' not in mode:
        encoding = 'utf-8'
    return _original_open(file, mode, buffering, encoding, errors,
                         newline, closefd, opener)

builtins.open = _utf8_open

import threading
import tempfile
import time
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import rumps
import pyperclip
import subprocess
import webbrowser
import signal

# Setup logging to file
LOG_PATH = Path.home() / ".parakeet_mlx.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger("parakeet")

# Add current dir for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from parakeet_mlx_guiapi.utils.config import get_config, save_config


# Available models (from mlx-community on HuggingFace)
# Organized by category with detailed metadata
AVAILABLE_MODELS = [
    # === TDT Models (Best accuracy, good speed) ===
    {
        "id": "mlx-community/parakeet-tdt-0.6b-v3",
        "name": "TDT 0.6B v3 Multilingual",
        "category": "Multilingual",
        "description": "25 languages incl. French, Spanish",
        "size": "~1.2GB",
        "languages": "EN, FR, ES, DE, IT, PT + 19 more",
        "lang_list": ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk", "cs", "sk", "bg", "hr", "da", "et", "fi", "el", "hu", "lv", "lt", "mt", "ro", "sl", "sv"],
        "wer": "6.34%",
        "speed": "Fast",
        "features": ["Auto punctuation", "Auto language detection", "Best for multilingual"],
        "recommended": True,
    },
    {
        "id": "mlx-community/parakeet-tdt-0.6b-v2",
        "name": "TDT 0.6B v2 English",
        "category": "English",
        "description": "English-only, very accurate",
        "size": "~1.2GB",
        "languages": "English only",
        "lang_list": ["en"],
        "wer": "6.5%",
        "speed": "Fast",
        "features": ["Auto punctuation", "Timestamps"],
        "recommended": False,
    },
    {
        "id": "mlx-community/parakeet-tdt-1.1b",
        "name": "TDT 1.1B English",
        "category": "English",
        "description": "Best English accuracy",
        "size": "~2.2GB",
        "languages": "English only",
        "lang_list": ["en"],
        "wer": "~5.5%",
        "speed": "Slower",
        "features": ["Auto punctuation", "Best for meetings/interviews"],
        "recommended": False,
    },

    # === CTC Models (Fastest inference) ===
    {
        "id": "mlx-community/parakeet-ctc-0.6b",
        "name": "CTC 0.6B English",
        "category": "Fast",
        "description": "Fastest inference",
        "size": "~1.2GB",
        "languages": "English only",
        "lang_list": ["en"],
        "wer": "~7%",
        "speed": "Fastest",
        "features": ["Non-autoregressive", "Real-time capable"],
        "recommended": False,
    },
    {
        "id": "mlx-community/parakeet-ctc-1.1b",
        "name": "CTC 1.1B English",
        "category": "Fast",
        "description": "Fast + better accuracy",
        "size": "~2.2GB",
        "languages": "English only",
        "lang_list": ["en"],
        "wer": "~6%",
        "speed": "Very Fast",
        "features": ["Non-autoregressive", "Long audio support"],
        "recommended": False,
    },

    # === Hybrid & Special Models ===
    {
        "id": "mlx-community/parakeet-tdt_ctc-1.1b",
        "name": "TDT+CTC 1.1B English",
        "category": "Long Audio",
        "description": "11hr audio in one pass",
        "size": "~2.2GB",
        "languages": "English only",
        "lang_list": ["en"],
        "wer": "~5.8%",
        "speed": "Medium",
        "features": ["Dual decoder", "Best for long recordings", "Podcasts/lectures"],
        "recommended": False,
    },
    {
        "id": "mlx-community/parakeet-tdt_ctc-110m",
        "name": "TDT+CTC 110M Tiny",
        "category": "Lightweight",
        "description": "Smallest, instant loading",
        "size": "~220MB",
        "languages": "English only",
        "lang_list": ["en"],
        "wer": "~12%",
        "speed": "Instant",
        "features": ["Ultra lightweight", "Quick notes"],
        "recommended": False,
    },
]

# Group models by category for menu display
def get_models_by_category():
    """Group models by their category."""
    categories = {}
    for model in AVAILABLE_MODELS:
        cat = model.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(model)
    return categories


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

        # Server control
        self._server_process = None
        self._server_port = 8080  # Default port (5000 is used by macOS AirPlay)
        self._gradio_port = 8081

        # Load config
        self.config = get_config()

        # History of transcriptions (last 10)
        self.history = []
        self._load_history()

        # Error tracking for debugging
        self._last_error = None

        # Build menu - rumps requires menu items to be created here
        self._setup_menu()

        # Lazy-load transcriber in background (with download progress if needed)
        threading.Thread(target=self._init_transcriber_with_download, daemon=True).start()

    def _init_transcriber_with_download(self):
        """Initialize transcriber, downloading in Terminal if needed."""
        model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
        model_info = self._get_model_by_id(model_name) or {}

        logger.info(f"Starting transcriber initialization for: {model_name}")

        is_cached = self._is_model_cached(model_name)
        logger.info(f"Model cache check: {'cached' if is_cached else 'not cached'}")

        if not is_cached:
            # Model needs download - use Terminal for progress
            logger.info("Model needs download, opening Terminal...")
            self._download_and_load_model(model_info if model_info else {"id": model_name, "name": model_name.split("/")[-1]})
        else:
            # Model is cached, load directly
            logger.info("Model is cached, loading directly...")
            self._init_transcriber()

    def _setup_menu(self):
        """Set up the initial menu structure."""
        # === Primary Actions ===
        # Record button (main action)
        self.record_button = rumps.MenuItem(
            "🎤 Start Recording",
            callback=self.toggle_recording
        )

        # Cancel recording button (hidden by default, shown during recording)
        self.cancel_button = rumps.MenuItem(
            "✖ Cancel Recording",
            callback=self.cancel_recording
        )

        # Transcribe file option
        self.transcribe_file_button = rumps.MenuItem(
            "📁 Transcribe File...",
            callback=self.transcribe_file
        )

        # === Status ===
        self.status_item = rumps.MenuItem("Status: Loading model...", callback=self.status_clicked)

        # === Server Controls ===
        self.server_menu = rumps.MenuItem("🌐 Server")
        self._populate_server_menu()

        # === Model Selection ===
        self.model_menu = rumps.MenuItem("🤖 Model")
        self._populate_model_menu()

        # === Settings ===
        self.settings_menu = rumps.MenuItem("⚙️ Settings")
        self._populate_settings_menu()

        # === History ===
        self.history_menu = rumps.MenuItem("📜 History")
        self._populate_history_menu()

        # === About and Quit ===
        about_item = rumps.MenuItem("ℹ️ About Parakeet", callback=self.show_about)
        help_item = rumps.MenuItem("❓ Help", callback=self.show_help)
        quit_item = rumps.MenuItem("⏻ Quit Parakeet", callback=self.quit_app)

        self.menu = [
            self.record_button,
            self.cancel_button,
            self.transcribe_file_button,
            None,  # Separator
            self.status_item,
            None,
            self.server_menu,
            self.model_menu,
            self.settings_menu,
            self.history_menu,
            None,
            help_item,
            about_item,
            quit_item,
        ]

        # Hide cancel button initially
        self.cancel_button.set_callback(None)  # Disable it initially

    def _populate_server_menu(self):
        """Populate the server control menu."""
        # Server status
        if self._server_process and self._server_process.poll() is None:
            status = "● Server Running"
            self.server_menu.add(rumps.MenuItem(f"✅ {status}"))
        else:
            status = "○ Server Stopped"
            self.server_menu.add(rumps.MenuItem(f"⚪ {status}"))

        self.server_menu.add(None)

        # Start/Stop buttons
        if self._server_process and self._server_process.poll() is None:
            self.server_menu.add(rumps.MenuItem("⏹ Stop Server", callback=self.stop_server))
            self.server_menu.add(rumps.MenuItem("🔄 Restart Server", callback=self.restart_server))
        else:
            self.server_menu.add(rumps.MenuItem("▶️ Start Server", callback=self.start_server))

        self.server_menu.add(None)

        # Open Web UI
        self.server_menu.add(rumps.MenuItem("🌐 Open Web UI", callback=self.open_web_ui))
        self.server_menu.add(rumps.MenuItem("📊 Open API Docs", callback=self.open_api_docs))

        self.server_menu.add(None)

        # Server config submenu
        config_menu = rumps.MenuItem("⚙️ Server Config")

        # Port configuration
        port_menu = rumps.MenuItem("API Port")
        current_port = self.config.get("server_port", 8080)
        for port in [8080, 8000, 3000, 5000]:
            title = f"{'✓ ' if port == current_port else ''}{port}"
            port_menu.add(rumps.MenuItem(title, callback=lambda _, p=port: self.set_server_port(p)))
        config_menu.add(port_menu)

        # Gradio port
        gradio_port_menu = rumps.MenuItem("Gradio Port")
        current_gradio = self.config.get("gradio_port", 8081)
        for port in [8081, 7860, 5001]:
            title = f"{'✓ ' if port == current_gradio else ''}{port}"
            gradio_port_menu.add(rumps.MenuItem(title, callback=lambda _, p=port: self.set_gradio_port(p)))
        config_menu.add(gradio_port_menu)

        # Debug mode toggle
        debug_mode = self.config.get("server_debug", False)
        debug_title = f"{'✓ ' if debug_mode else ''}Debug Mode"
        config_menu.add(rumps.MenuItem(debug_title, callback=self.toggle_debug_mode))

        self.server_menu.add(config_menu)

    def _refresh_server_menu(self):
        """Refresh the server menu."""
        keys = list(self.server_menu.keys())
        for key in keys:
            del self.server_menu[key]
        self._populate_server_menu()

    def _populate_model_menu(self):
        """Populate the model selection menu organized by category."""
        current_model = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
        categories = get_models_by_category()

        # Define category order
        category_order = [
            "Multilingual",
            "English",
            "Fast",
            "Long Audio",
            "Lightweight",
        ]

        for category in category_order:
            if category not in categories:
                continue

            # Add category header
            cat_submenu = rumps.MenuItem(category)

            for model in categories[category]:
                # Build display title with checkmark and details
                title = model["name"]
                if model["id"] == current_model:
                    title = f"✓ {title}"
                if model.get("recommended"):
                    title = f"⭐ {title}"

                item = rumps.MenuItem(
                    title,
                    callback=lambda sender, m=model: self.select_model(m)
                )
                cat_submenu.add(item)

            self.model_menu.add(cat_submenu)

        # Add separator and current model info
        self.model_menu.add(None)

        # Show current model details
        current = self._get_model_by_id(current_model)
        if current:
            info_items = [
                f"Current: {current['name']}",
                f"Languages: {current.get('languages', 'Unknown')}",
                f"WER: {current.get('wer', 'N/A')}",
                f"Speed: {current.get('speed', 'N/A')}",
                f"Size: {current.get('size', 'Unknown')}",
            ]
            for info in info_items:
                self.model_menu.add(rumps.MenuItem(info))

            # Show features if available
            features = current.get("features", [])
            if features:
                self.model_menu.add(None)
                feat_menu = rumps.MenuItem("Features")
                for feat in features:
                    feat_menu.add(rumps.MenuItem(f"• {feat}"))
                self.model_menu.add(feat_menu)

    def _get_model_by_id(self, model_id):
        """Get model dict by its ID."""
        for model in AVAILABLE_MODELS:
            if model["id"] == model_id:
                return model
        return None

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
        # === Diarization (Speaker ID) ===
        diarize_enabled = self.config.get("diarization_enabled", False)
        diarize_available, diarize_msg = self._check_diarization_available()

        # Create diarization submenu
        diarize_menu = rumps.MenuItem("Speaker Diarization")

        if diarize_available:
            # Check for missing model access (do this in background to avoid blocking menu)
            missing_models = []
            try:
                # Quick check - only block briefly
                missing_models = self._check_all_models_accessible()
            except Exception:
                pass  # Network error, etc. - proceed optimistically

            if missing_models:
                # Some models still need access
                diarize_menu.add(rumps.MenuItem("⚠️ Model access incomplete"))
                diarize_menu.add(None)
                for model_id, desc in missing_models:
                    short_name = model_id.split("/")[-1]
                    diarize_menu.add(rumps.MenuItem(f"❌ {short_name}"))
                diarize_menu.add(None)
                diarize_menu.add(rumps.MenuItem("🚀 Complete Setup...", callback=self.start_diarization_setup))
            else:
                # All good - show full options
                # Toggle option
                toggle_title = "✓ Enabled" if diarize_enabled else "Enabled"
                diarize_menu.add(rumps.MenuItem(toggle_title, callback=self.toggle_diarization))
                diarize_menu.add(None)

                # Number of speakers submenu
                speakers_menu = rumps.MenuItem("Number of Speakers")
                current_speakers = self.config.get("diarization_num_speakers", 0)  # 0 = auto

                # Auto-detect option
                auto_title = "✓ Auto-detect" if current_speakers == 0 else "Auto-detect"
                speakers_menu.add(rumps.MenuItem(
                    auto_title,
                    callback=lambda _: self.set_num_speakers(0)
                ))
                speakers_menu.add(None)

                # Preset options: 2-6 speakers
                for num in range(2, 7):
                    title = f"{num} speakers"
                    if current_speakers == num:
                        title = f"✓ {title}"
                    speakers_menu.add(rumps.MenuItem(
                        title,
                        callback=lambda _, n=num: self.set_num_speakers(n)
                    ))

                diarize_menu.add(speakers_menu)
                diarize_menu.add(None)
                diarize_menu.add(rumps.MenuItem("✅ Setup complete"))
        else:
            # Show what's missing and setup options
            diarize_menu.add(rumps.MenuItem("⚠️ Setup required"))
            diarize_menu.add(None)

            # Check specific issues
            pyannote_ok, token_ok = self._check_diarization_components()

            if pyannote_ok:
                diarize_menu.add(rumps.MenuItem("✅ pyannote.audio installed"))
            else:
                diarize_menu.add(rumps.MenuItem("❌ pyannote.audio not installed"))
                diarize_menu.add(rumps.MenuItem("   Install: pip install pyannote.audio"))

            if token_ok:
                diarize_menu.add(rumps.MenuItem("✅ HuggingFace token set"))
            else:
                diarize_menu.add(rumps.MenuItem("❌ HuggingFace token missing"))

            diarize_menu.add(None)
            diarize_menu.add(rumps.MenuItem("🚀 Quick Setup...", callback=self.start_diarization_setup))

        self.settings_menu.add(diarize_menu)
        self.settings_menu.add(None)

        # === Chunk duration options ===
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

        # === Advanced section ===
        self.settings_menu.add(None)
        advanced_menu = rumps.MenuItem("Advanced")

        # Show Python environment
        python_path = sys.executable
        python_short = python_path if len(python_path) < 40 else "..." + python_path[-37:]
        advanced_menu.add(rumps.MenuItem(f"Python: {python_short}"))

        # Show cache location
        cache_path = self._get_cache_path()
        cache_short = cache_path if len(cache_path) < 40 else "..." + cache_path[-37:]
        advanced_menu.add(rumps.MenuItem(f"Cache: {cache_short}"))

        advanced_menu.add(None)

        # Pre-download models option
        advanced_menu.add(rumps.MenuItem("Pre-download Model...", callback=self.predownload_model))

        # Open cache folder
        advanced_menu.add(rumps.MenuItem("Open Cache Folder", callback=self.open_cache_folder))

        # Config file location
        advanced_menu.add(rumps.MenuItem("Open Config File", callback=self.open_config_file))

        advanced_menu.add(None)

        # Logging and debugging
        advanced_menu.add(rumps.MenuItem("View Logs", callback=self.view_logs))
        advanced_menu.add(rumps.MenuItem("View Last Error", callback=self.view_last_error))

        advanced_menu.add(None)

        # Reload/Restart
        advanced_menu.add(rumps.MenuItem("Reload Model", callback=self.reload_model))

        self.settings_menu.add(advanced_menu)

    def _check_diarization_available(self):
        """Check if diarization is fully available."""
        try:
            from parakeet_mlx_guiapi.diarization import SpeakerDiarizer
            # Also check config for token
            token = (
                self.config.get("huggingface_token")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HF_TOKEN")
            )
            if not token:
                return False, "HuggingFace token not set"
            return SpeakerDiarizer.is_available()
        except ImportError:
            return False, "pyannote.audio not installed"

    def _check_diarization_components(self):
        """Check individual diarization components."""
        # Check pyannote
        try:
            import pyannote.audio
            pyannote_ok = True
        except ImportError:
            pyannote_ok = False

        # Check token (config or env)
        token = (
            self.config.get("huggingface_token")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        token_ok = bool(token)

        return pyannote_ok, token_ok

    def _check_model_access(self, model_id: str) -> bool:
        """Check if user has access to a HuggingFace model."""
        try:
            from huggingface_hub import model_info
            token = (
                self.config.get("huggingface_token")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HF_TOKEN")
            )
            # Try to get model info - will raise if no access
            model_info(model_id, token=token)
            return True
        except Exception as e:
            if "403" in str(e) or "restricted" in str(e) or "gated" in str(e).lower():
                return False
            # Other errors (network, etc.) - assume accessible
            return True

    def _get_required_diarization_models(self):
        """Get list of required models for diarization."""
        return [
            ("pyannote/speaker-diarization-3.1", "Main pipeline"),
            ("pyannote/segmentation-3.0", "Voice detection"),
            ("pyannote/wespeaker-voxceleb-resnet34-LM", "Speaker embeddings"),
            ("pyannote/speaker-diarization", "Base diarization"),
            ("pyannote/speaker-diarization-community-1", "Community model"),
        ]

    def _check_all_models_accessible(self):
        """Check if all required diarization models are accessible."""
        models = self._get_required_diarization_models()
        missing = []
        for model_id, desc in models:
            if not self._check_model_access(model_id):
                missing.append((model_id, desc))
        return missing

    def toggle_diarization(self, _):
        """Toggle speaker diarization."""
        available, msg = self._check_diarization_available()
        if not available:
            self.start_diarization_setup(None)
            return

        current = self.config.get("diarization_enabled", False)
        self.config["diarization_enabled"] = not current
        save_config(self.config)
        self._refresh_settings_menu()

        status = "enabled" if not current else "disabled"
        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Speaker Diarization",
                subtitle=status.capitalize(),
                message="Transcripts will include speaker labels" if not current else "Speaker labels disabled",
                sound=False
            )

    def set_num_speakers(self, num_speakers):
        """Set the number of speakers for diarization."""
        self.config["diarization_num_speakers"] = num_speakers
        save_config(self.config)
        self._refresh_settings_menu()

        if num_speakers == 0:
            msg = "Will auto-detect number of speakers"
        else:
            msg = f"Will identify {num_speakers} speakers"

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Speaker Diarization",
                subtitle=f"{'Auto-detect' if num_speakers == 0 else f'{num_speakers} speakers'}",
                message=msg,
                sound=False
            )

    def start_diarization_setup(self, _):
        """Interactive diarization setup wizard."""
        pyannote_ok, token_ok = self._check_diarization_components()

        # Step 1: Check pyannote
        if not pyannote_ok:
            response = rumps.alert(
                title="Speaker Diarization Setup (1/3)",
                message=(
                    "pyannote.audio is not installed.\n\n"
                    "This is required for speaker identification.\n"
                    "Install size: ~500MB\n\n"
                    "Install now?"
                ),
                ok="Install",
                cancel="Cancel"
            )
            if response == 1:  # OK clicked
                self._install_pyannote()
            return

        # Step 2: Check token
        if not token_ok:
            response = rumps.alert(
                title="Speaker Diarization Setup (2/3)",
                message=(
                    "HuggingFace token is required.\n\n"
                    "You need:\n"
                    "1. A free HuggingFace account\n"
                    "2. Accept the pyannote model license\n"
                    "3. A 'Read' access token (not Write)\n\n"
                    "Already have a token starting with 'hf_...'?"
                ),
                ok="I have a token",
                cancel="Guide me through setup"
            )

            if response == 0:  # Cancel = Open HuggingFace
                self._open_huggingface_setup()
            else:  # OK = Enter token
                self._prompt_for_token()
            return

        # Step 3: All set - enable and test
        self._finalize_diarization_setup()

    def _install_pyannote(self):
        """Install pyannote.audio package with visible progress in Terminal."""
        # Get Python executable path
        python_path = sys.executable

        # Build the install command with progress display
        # Note: On macOS, pip automatically installs CPU/MPS version of PyTorch (no CUDA)
        install_cmd = f'''
clear
echo "══════════════════════════════════════════════════════════════"
echo "  Installing pyannote.audio for Speaker Diarization"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Python: {python_path}"
echo "Platform: Apple Silicon (MPS/CPU - no CUDA needed)"
echo ""
echo "This may take a few minutes..."
echo "Installing PyTorch + pyannote.audio..."
echo ""
"{python_path}" -m pip install --progress-bar on "pyannote.audio>=3.1.0"
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Installation complete!"
    echo ""
    echo "pyannote will use CPU for maximum compatibility."
    echo "(Apple Silicon GPU/MPS is experimental for pyannote)"
    echo ""
    echo "You can close this window and continue setup in Parakeet."
else
    echo "❌ Installation failed (exit code: $EXIT_CODE)"
    echo ""
    echo "Try running manually:"
    echo "  {python_path} -m pip install pyannote.audio"
fi
echo ""
echo "Press any key to close..."
read -n 1
'''

        # Open Terminal with the install command
        script = f'''
        tell application "Terminal"
            activate
            do script "{install_cmd.replace('"', '\\"').replace('\n', '\\n')}"
        end tell
        '''

        try:
            subprocess.run(["osascript", "-e", script], check=True)
            self.status_item.title = "Installing... (see Terminal)"
        except Exception as e:
            rumps.alert(
                title="Could not open Terminal",
                message=f"Error: {e}\n\nTry installing manually:\npip install pyannote.audio"
            )

    def _open_huggingface_setup(self):
        """Open HuggingFace pages for setup."""
        # Show detailed instructions with ALL required models
        models = self._get_required_diarization_models()
        rumps.alert(
            title=f"HuggingFace Setup (Step 1 of {len(models) + 1})",
            message=(
                f"Speaker diarization requires access to {len(models)} models.\n\n"
                "For each model, you need to:\n"
                "1. Sign in (or create a free account)\n"
                "2. Scroll to 'Agree and access repository'\n"
                "3. Click to accept the license\n\n"
                "I'll open each model page in sequence."
            ),
            ok="Start Setup"
        )

        for i, (model, desc) in enumerate(models, 1):
            rumps.alert(
                title=f"Accept Model License ({i}/{len(models)})",
                message=(
                    f"Model: {model}\n"
                    f"Purpose: {desc}\n\n"
                    "Click OK to open the model page.\n"
                    "Accept the license, then come back."
                ),
                ok="Open Model Page"
            )
            webbrowser.open(f"https://huggingface.co/{model}")

        # Show token instructions
        rumps.alert(
            title=f"HuggingFace Setup (Final Step)",
            message=(
                "Now create an access token.\n\n"
                "Create a token with these settings:\n"
                "• Name: 'Parakeet' (or anything)\n"
                "• Type: 'Read' (NOT Write)\n\n"
                "Copy the token (starts with 'hf_...')\n"
                "Then click 'Quick Setup' → 'I have a token'"
            ),
            ok="Open Token Page"
        )

        webbrowser.open("https://huggingface.co/settings/tokens/new?tokenType=read")

    def _prompt_for_token(self):
        """Prompt user to enter their HuggingFace token."""
        # Use AppleScript for text input (rumps doesn't have input dialogs)
        script = '''
        tell application "System Events"
            display dialog "Paste your HuggingFace token:" default answer "" with title "Enter Token" buttons {"Cancel", "Save"} default button "Save"
            if button returned of result is "Save" then
                return text returned of result
            end if
        end tell
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True
            )
            token = result.stdout.strip()

            if token and len(token) > 10:  # Basic validation
                # Save to config
                self.config["huggingface_token"] = token
                save_config(self.config)
                self._refresh_settings_menu()

                rumps.notification(
                    title="Token Saved",
                    subtitle="",
                    message="HuggingFace token saved to config",
                    sound=False
                )

                # Continue to finalize
                self._finalize_diarization_setup()
            elif token:
                rumps.alert(
                    title="Invalid Token",
                    message="The token seems too short. Please try again."
                )
        except Exception as e:
            rumps.alert(
                title="Error",
                message=f"Could not prompt for token: {e}"
            )

    def _finalize_diarization_setup(self):
        """Final step: enable diarization and test."""
        response = rumps.alert(
            title="Speaker Diarization Setup (3/3)",
            message=(
                "Setup complete! ✅\n\n"
                "First use will download the diarization model (~1GB).\n"
                "Diarization adds ~10-30 seconds processing time.\n\n"
                "Enable speaker diarization now?"
            ),
            ok="Enable",
            cancel="Later"
        )

        if response == 1:  # Enable
            self.config["diarization_enabled"] = True
            save_config(self.config)
            self._refresh_settings_menu()

            rumps.notification(
                title="Speaker Diarization Enabled",
                subtitle="",
                message="Your next transcription will identify speakers",
                sound=False
            )

    def _get_cache_path(self):
        """Get the HuggingFace cache path."""
        try:
            from huggingface_hub import constants
            return constants.HF_HUB_CACHE
        except Exception:
            return os.path.expanduser("~/.cache/huggingface/hub")

    def open_cache_folder(self, _):
        """Open the model cache folder in Finder."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            subprocess.run(["open", cache_path])
        else:
            rumps.alert(
                title="Cache Not Found",
                message=f"Cache folder does not exist yet:\n{cache_path}\n\nIt will be created when you download your first model."
            )

    def open_config_file(self, _):
        """Open the config file in default editor."""
        config_path = os.path.expanduser("~/.parakeet_mlx_guiapi.json")
        if os.path.exists(config_path):
            subprocess.run(["open", config_path])
        else:
            rumps.alert(
                title="Config File",
                message=f"Config file will be created at:\n{config_path}\n\nIt's created when you first change a setting."
            )

    def view_logs(self, _):
        """Open the log file in Console.app or default text editor."""
        if LOG_PATH.exists():
            # Use Console.app for better log viewing on macOS
            subprocess.run(["open", "-a", "Console", str(LOG_PATH)])
        else:
            rumps.alert(
                title="No Logs Yet",
                message=f"Log file will be created at:\n{LOG_PATH}\n\nLogs are written when the app starts or encounters errors."
            )

    def view_last_error(self, _):
        """Show details of the last error that occurred."""
        if self._last_error is None:
            rumps.alert(
                title="No Errors",
                message="No errors have occurred since the app started.\n\nIf you're experiencing issues, try:\n• View Logs for full history\n• Reload Model to retry loading"
            )
            return

        error_info = self._last_error
        rumps.alert(
            title="Last Error Details",
            message=(
                f"Time: {error_info.get('time', 'Unknown')}\n"
                f"Model: {error_info.get('model', 'Unknown')}\n\n"
                f"Error: {error_info.get('error', 'Unknown')}\n\n"
                f"Full traceback saved to:\n{LOG_PATH}"
            )
        )

    def status_clicked(self, _):
        """Handle click on status item - show error details or info."""
        if self._last_error:
            # Show error details with options
            error_info = self._last_error
            response = rumps.alert(
                title="Model Load Error",
                message=(
                    f"Model: {error_info.get('model', 'Unknown')}\n\n"
                    f"Error: {error_info.get('error', 'Unknown')[:200]}\n\n"
                    f"Time: {error_info.get('time', 'Unknown')}\n\n"
                    "Would you like to try reloading the model?"
                ),
                ok="Reload Model",
                cancel="View Full Logs"
            )
            if response == 1:  # Reload
                self.reload_model(None)
            else:  # View logs
                self.view_logs(None)
        elif self.transcriber is None:
            rumps.alert(
                title="Loading...",
                message="Model is still loading. Please wait."
            )
        else:
            model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
            model_info = self._get_model_by_id(model_name)
            if model_info:
                rumps.alert(
                    title="Parakeet Ready",
                    message=(
                        f"Model: {model_info['name']}\n"
                        f"Languages: {model_info.get('languages', 'Unknown')}\n"
                        f"Accuracy: {model_info.get('wer', 'N/A')}\n\n"
                        "Click the mic icon to start recording!"
                    )
                )

    def reload_model(self, _):
        """Reload the current model (useful after errors)."""
        if self.recording or self.processing:
            rumps.notification(
                title="Cannot Reload",
                subtitle="",
                message="Please wait until current operation completes",
                sound=False
            )
            return

        model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
        model_short = self._get_model_short_name(model_name)

        # Clear existing transcriber
        self.transcriber = None
        self._last_error = None

        # Reset status
        self.status_item.title = f"Reloading {model_short}..."

        logger.info(f"User requested model reload: {model_name}")

        # Reload in background
        threading.Thread(target=self._init_transcriber_with_download, daemon=True).start()

        rumps.notification(
            title="Reloading Model",
            subtitle=model_short,
            message="Model is being reloaded...",
            sound=False
        )

    def predownload_model(self, _):
        """Pre-download a model with progress display in Terminal."""
        # Build list of models not yet cached
        uncached = []
        for model in AVAILABLE_MODELS:
            if not self._is_model_cached(model["id"]):
                uncached.append(model)

        if not uncached:
            rumps.alert(
                title="All Models Cached",
                message="All available models are already downloaded!"
            )
            return

        # Show selection dialog
        model_list = "\n".join([f"  {i+1}. {m['name']} ({m['size']})" for i, m in enumerate(uncached)])

        script = f'''
        tell application "System Events"
            display dialog "Select model to download:\\n\\n{model_list}\\n\\nEnter number (1-{len(uncached)}):" default answer "1" with title "Download Model" buttons {{"Cancel", "Download"}} default button "Download"
            if button returned of result is "Download" then
                return text returned of result
            end if
        end tell
        '''

        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            choice = result.stdout.strip()

            if choice and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(uncached):
                    model = uncached[idx]
                    self._download_model_in_terminal(model)
        except Exception as e:
            rumps.alert(title="Error", message=str(e))

    def _download_model_in_terminal(self, model):
        """Download a model with visible progress in Terminal."""
        model_id = model["id"]
        model_name = model["name"]
        model_size = model.get("size", "unknown size")
        python_path = sys.executable

        download_cmd = f'''
clear
echo "══════════════════════════════════════════════════════════════"
echo "  Downloading: {model_name}"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Model: {model_id}"
echo "Size: {model_size}"
echo ""
echo "Downloading from HuggingFace..."
echo ""
"{python_path}" -c "
from huggingface_hub import snapshot_download
import sys

def progress_callback(progress):
    pass  # HF handles its own progress bar

print('Starting download...')
try:
    path = snapshot_download('{model_id}', local_files_only=False)
    print(f'\\n✅ Download complete!')
    print(f'Saved to: {{path}}')
except Exception as e:
    print(f'\\n❌ Download failed: {{e}}')
    sys.exit(1)
"
echo ""
echo "Press any key to close..."
read -n 1
'''

        script = f'''
        tell application "Terminal"
            activate
            do script "{download_cmd.replace('"', '\\"').replace('\n', '\\n')}"
        end tell
        '''

        try:
            subprocess.run(["osascript", "-e", script], check=True)
            self.status_item.title = f"Downloading... (see Terminal)"
        except Exception as e:
            rumps.alert(
                title="Could not open Terminal",
                message=f"Error: {e}"
            )

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
        """Initialize transcriber in background with progress feedback."""
        model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
        model_short = self._get_model_short_name(model_name)

        try:
            logger.info(f"Initializing transcriber with model: {model_name}")
            model_info = self._get_model_by_id(model_name) or {}

            # Check if model is cached
            is_cached = self._is_model_cached(model_name)
            logger.info(f"Model cached: {is_cached}")

            if is_cached:
                self.status_item.title = f"Loading {model_short}..."
            else:
                # Model needs to be downloaded
                size = model_info.get("size", "~1GB")
                self.status_item.title = f"Downloading {model_short}..."

                if self.config.get("show_notifications", True):
                    rumps.notification(
                        title="Downloading Model",
                        subtitle=model_short,
                        message=f"First-time download: {size}\nThis may take a few minutes...",
                        sound=False
                    )

            # Import and load model
            logger.info("Importing AudioTranscriber...")
            from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber

            logger.info(f"Creating AudioTranscriber with model: {model_name}")
            logger.info("This may take a moment for first load...")

            try:
                self.transcriber = AudioTranscriber(model_name=model_name)
                logger.info("AudioTranscriber created successfully")
            except Exception as load_error:
                logger.error(f"AudioTranscriber creation failed: {load_error}")
                logger.error(f"Model ID: {model_name}")
                raise

            logger.info("Model loaded successfully")
            self.status_item.title = f"Ready: {model_short}"
            self._last_error = None

            if self.config.get("show_notifications", True):
                msg = "Model loaded from cache" if is_cached else "Download complete!"
                rumps.notification(
                    title="Parakeet Ready",
                    subtitle=model_short,
                    message=f"{msg}\nClick the mic icon to record",
                    sound=False
                )

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            logger.error(f"Failed to load model: {error_msg}")
            logger.error(f"Traceback:\n{error_trace}")

            self._last_error = {
                "model": model_name,
                "error": error_msg,
                "traceback": error_trace,
                "time": datetime.now().isoformat()
            }

            self.status_item.title = "⚠️ Error - Click for options"
            rumps.notification(
                title="Parakeet Error",
                subtitle=f"Failed to load {model_short}",
                message=f"{error_msg[:80]}...\nCheck Settings > View Logs",
                sound=True
            )

    def _is_model_cached(self, model_name):
        """Check if a model is already downloaded/cached."""
        try:
            from huggingface_hub import try_to_load_from_cache
            # Try to find the config file in cache
            cached = try_to_load_from_cache(model_name, "config.json")
            return cached is not None
        except Exception:
            # If we can't check, assume not cached
            return False

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

        # Check if model needs to be downloaded
        is_cached = self._is_model_cached(model["id"])

        if not is_cached:
            # Model needs download - ask user and show progress in Terminal
            response = rumps.alert(
                title="Download Required",
                message=(
                    f"Model: {model['name']}\n"
                    f"Size: {model.get('size', 'unknown')}\n\n"
                    "This model needs to be downloaded first.\n"
                    "Download progress will be shown in Terminal."
                ),
                ok="Download",
                cancel="Cancel"
            )

            if response != 1:  # User cancelled
                return

            # Download in Terminal with progress, then load
            self._download_and_load_model(model)
        else:
            # Model is cached, load directly
            self._switch_to_model(model)

    def _switch_to_model(self, model):
        """Switch to an already-cached model."""
        # Update config
        self.config["model_name"] = model["id"]
        save_config(self.config)

        # Update menu
        self._refresh_model_menu()

        # Reload transcriber
        self.transcriber = None
        self.status_item.title = f"Loading {model['name']}..."
        threading.Thread(target=self._init_transcriber, daemon=True).start()

        rumps.notification(
            title="Loading Model",
            subtitle=model["name"],
            message="Loading from cache...",
            sound=False
        )

    def _download_and_load_model(self, model):
        """Download a model in Terminal with progress, then load it."""
        model_id = model["id"]
        model_name = model["name"]
        model_size = model.get("size", "unknown size")
        python_path = sys.executable

        # Update config now so it loads this model after download
        self.config["model_name"] = model_id
        save_config(self.config)
        self._refresh_model_menu()

        download_cmd = f'''
clear
echo "══════════════════════════════════════════════════════════════"
echo "  Downloading: {model_name}"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Model: {model_id}"
echo "Size: {model_size}"
echo ""
"{python_path}" -c "
from huggingface_hub import snapshot_download
import sys

print('Downloading from HuggingFace...')
print('(Progress bar will appear below)')
print('')

try:
    path = snapshot_download('{model_id}', local_files_only=False)
    print('')
    print('✅ Download complete!')
    print(f'Saved to: {{path}}')
    print('')
    print('The model will now load in Parakeet.')
except Exception as e:
    print(f'')
    print(f'❌ Download failed: {{e}}')
    sys.exit(1)
"
echo ""
echo "You can close this window."
echo "Press any key to close..."
read -n 1
'''

        script = f'''
        tell application "Terminal"
            activate
            do script "{download_cmd.replace('"', '\\"').replace('\n', '\\n')}"
        end tell
        '''

        try:
            subprocess.run(["osascript", "-e", script], check=True)
            self.status_item.title = f"Downloading... (see Terminal)"

            # Start a background thread to wait for download and then load
            def wait_and_load():
                # Poll until model is cached
                for _ in range(600):  # Max 10 minutes
                    time.sleep(1)
                    if self._is_model_cached(model_id):
                        # Model downloaded, now load it
                        self.status_item.title = f"Loading {model_name}..."
                        self._init_transcriber()
                        return
                # Timeout
                self.status_item.title = "Download timeout"
                rumps.notification(
                    title="Download Timeout",
                    subtitle="",
                    message="Model download took too long. Try again.",
                    sound=True
                )

            threading.Thread(target=wait_and_load, daemon=True).start()

        except Exception as e:
            rumps.alert(
                title="Could not open Terminal",
                message=f"Error: {e}"
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
        logger.info(f"toggle_recording called - recording={self.recording}, processing={self.processing}")
        if self.processing:
            logger.info("Still processing, ignoring toggle")
            if self.config.get("show_notifications", True):
                rumps.notification(
                    title="Parakeet",
                    subtitle="",
                    message="Still processing previous recording...",
                    sound=False
                )
            return

        if not self.recording:
            logger.info("Starting recording...")
            self.start_recording()
        else:
            logger.info("Stopping recording...")
            self.stop_recording()

    def start_recording(self):
        """Start recording from microphone."""
        try:
            import sounddevice as sd
            import numpy as np

            logger.info("start_recording: Initializing...")
            self.recording = True
            self._recording_start_time = time.time()
            self.title = self.ICON_RECORDING
            self.record_button.title = "⏹ Stop Recording"
            self.cancel_button.set_callback(self.cancel_recording)  # Enable cancel button
            self._audio_data = []

            # Recording parameters
            self.sample_rate = 16000
            self.channels = 1

            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                if self.recording:
                    self._audio_data.append(indata.copy())

            # Start recording stream
            logger.info(f"start_recording: Creating InputStream (rate={self.sample_rate}, channels={self.channels})")
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=audio_callback
            )
            self._stream.start()
            logger.info("start_recording: Stream started successfully")

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
            logger.error(f"start_recording: Error - {e}", exc_info=True)
            self.recording = False
            self.title = self.ICON_ERROR
            self.record_button.title = "🎤 Start Recording"
            self.cancel_button.set_callback(None)  # Disable cancel button
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

        logger.info("stop_recording: Stopping stream...")
        self.recording = False
        self.cancel_button.set_callback(None)  # Disable cancel button
        if self._stream:
            self._stream.stop()
            self._stream.close()
            logger.info("stop_recording: Stream closed")

        if not self._audio_data:
            logger.warning("stop_recording: No audio data captured")
            self.title = self.ICON_IDLE
            self.record_button.title = "🎤 Start Recording"
            rumps.notification(
                title="No Audio",
                subtitle="",
                message="No audio was recorded",
                sound=True
            )
            return

        # Calculate duration
        recording_duration = time.time() - self._recording_start_time
        logger.info(f"stop_recording: Recorded {recording_duration:.1f}s, {len(self._audio_data)} chunks")

        # Update UI for processing
        self.processing = True
        self.title = self.ICON_PROCESSING
        self.record_button.title = "Processing..."
        self.status_item.title = "Transcribing..."

        # Process in background thread
        logger.info("stop_recording: Starting processing thread...")
        threading.Thread(
            target=self._process_audio,
            args=(recording_duration,),
            daemon=True
        ).start()

    def _process_audio(self, recording_duration):
        """Process recorded audio and transcribe (with optional diarization)."""
        import numpy as np
        from scipy.io import wavfile

        process_start = time.time()
        logger.info(f"_process_audio: Starting processing for {recording_duration:.1f}s recording")

        try:
            # Concatenate audio data
            audio_data = np.concatenate(self._audio_data, axis=0)
            logger.info(f"_process_audio: Audio data shape: {audio_data.shape}")

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
            logger.info("_process_audio: Starting transcription...")
            transcribe_start = time.time()
            chunk_duration = self.config.get("default_chunk_duration", 120)
            df, full_text = self.transcriber.transcribe(
                temp_path,
                chunk_duration=chunk_duration
            )
            transcribe_time = time.time() - transcribe_start
            logger.info(f"_process_audio: Transcription complete in {transcribe_time:.2f}s")

            # Handle None result
            if full_text is None:
                logger.warning("_process_audio: Transcription returned None")
                full_text = ""
            logger.info(f"_process_audio: Result: {len(full_text)} chars")

            # === Speaker Diarization (optional) ===
            output_text = full_text
            num_speakers = 0

            diarization_enabled = self.config.get("diarization_enabled", False)
            logger.info(f"_process_audio: Diarization enabled={diarization_enabled}, df is None={df is None}")

            if diarization_enabled and df is not None:
                try:
                    logger.info("_process_audio: Starting speaker diarization...")
                    self.status_item.title = "Identifying speakers..."
                    from parakeet_mlx_guiapi.diarization import SpeakerDiarizer

                    # Initialize diarizer if needed
                    if not hasattr(self, '_diarizer') or self._diarizer is None:
                        self._diarizer = SpeakerDiarizer()

                    # Get speaker count setting (0 = auto-detect)
                    configured_speakers = self.config.get("diarization_num_speakers", 0)

                    # Run diarization with speaker hint if configured
                    if configured_speakers > 0:
                        diarization = self._diarizer.diarize(
                            temp_path,
                            num_speakers=configured_speakers
                        )
                    else:
                        diarization = self._diarizer.diarize(temp_path)
                    num_speakers = diarization.num_speakers

                    logger.info(f"_process_audio: Diarization complete, found {num_speakers} speakers")

                    # Convert DataFrame to list of dicts for merging
                    segments = df.to_dict('records')

                    # Format with speaker labels (markdown format)
                    output_text = diarization.format_transcript_markdown(segments)
                    logger.info(f"_process_audio: Formatted transcript with speaker labels")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"_process_audio: Diarization failed - {e}", exc_info=True)

                    # Provide helpful error messages for common issues
                    if "403" in error_msg or "restricted" in error_msg or "authorized" in error_msg:
                        rumps.notification(
                            title="Diarization Access Denied",
                            subtitle="Model license not accepted",
                            message="Visit huggingface.co/pyannote to accept the model license",
                            sound=True
                        )
                    elif "401" in error_msg or "token" in error_msg.lower():
                        rumps.notification(
                            title="Diarization Auth Error",
                            subtitle="Invalid HuggingFace token",
                            message="Check your token in Settings > Speaker Diarization",
                            sound=True
                        )
                    else:
                        rumps.notification(
                            title="Diarization Failed",
                            subtitle="",
                            message=error_msg[:80],
                            sound=True
                        )

                    # Fall back to plain transcription
                    output_text = full_text

            # Clean up temp file
            os.remove(temp_path)

            if output_text:
                # Copy to clipboard if enabled
                if self.config.get("auto_copy_clipboard", True):
                    pyperclip.copy(output_text)

                # Add to history
                self._add_to_history(output_text, recording_duration)

                # Show notification with preview
                if self.config.get("show_notifications", True):
                    preview = output_text[:80] + "..." if len(output_text) > 80 else output_text
                    copied_msg = " - Copied!" if self.config.get("auto_copy_clipboard", True) else ""
                    speaker_info = f" ({num_speakers} speakers)" if num_speakers > 0 else ""
                    rumps.notification(
                        title=f"Transcription Complete{copied_msg}",
                        subtitle=f"{recording_duration:.1f}s of audio{speaker_info}",
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
            logger.error(f"_process_audio: Error - {e}", exc_info=True)
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
            total_time = time.time() - process_start
            logger.info(f"_process_audio: Complete. Total processing time: {total_time:.2f}s")
            self.processing = False
            self.record_button.title = "🎤 Start Recording"
            model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
            self.status_item.title = f"Ready: {self._get_model_short_name(model_name)}"

    # === Server Control Methods ===

    def start_server(self, _):
        """Start the Flask + Gradio server."""
        if self._server_process and self._server_process.poll() is None:
            rumps.notification(
                title="Server Already Running",
                subtitle="",
                message=f"Server is already running on port {self._server_port}",
                sound=False
            )
            return

        try:
            # Get config
            port = self.config.get("server_port", 8080)
            gradio_port = self.config.get("gradio_port", 5001)
            debug = self.config.get("server_debug", False)
            model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])

            # Build command
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
            cmd = [
                sys.executable, script_path,
                "--host", "127.0.0.1",
                "--port", str(port),
                "--model", model_name
            ]
            if debug:
                cmd.append("--debug")

            # Set environment variables for Gradio port
            env = os.environ.copy()
            env["GRADIO_SERVER_PORT"] = str(gradio_port)

            logger.info(f"Starting server: {' '.join(cmd)}")

            # Start server process
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True  # Allow it to run independently
            )
            self._server_port = port
            self._gradio_port = gradio_port

            # Refresh menu
            self._refresh_server_menu()

            if self.config.get("show_notifications", True):
                rumps.notification(
                    title="Server Started",
                    subtitle=f"Port {port}",
                    message=f"API: http://127.0.0.1:{port}\nWeb UI: http://127.0.0.1:{gradio_port}",
                    sound=False
                )

            logger.info(f"Server started on port {port}")

        except Exception as e:
            logger.error(f"Failed to start server: {e}", exc_info=True)
            rumps.notification(
                title="Server Error",
                subtitle="Failed to start",
                message=str(e)[:100],
                sound=True
            )

    def stop_server(self, _):
        """Stop the running server."""
        if not self._server_process or self._server_process.poll() is not None:
            rumps.notification(
                title="Server Not Running",
                subtitle="",
                message="No server is currently running",
                sound=False
            )
            return

        try:
            # Send SIGTERM for graceful shutdown
            os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
            self._server_process.wait(timeout=5)
            logger.info("Server stopped gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
            logger.warning("Server force-killed after timeout")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")

        self._server_process = None
        self._refresh_server_menu()

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Server Stopped",
                subtitle="",
                message="The server has been stopped",
                sound=False
            )

    def restart_server(self, _):
        """Restart the server."""
        self.stop_server(None)
        time.sleep(1)
        self.start_server(None)

    def open_web_ui(self, _):
        """Open the Gradio web UI in browser."""
        port = self.config.get("gradio_port", 8081)
        webbrowser.open(f"http://127.0.0.1:{port}")

    def open_api_docs(self, _):
        """Open the API documentation."""
        port = self.config.get("server_port", 5000)
        # Show API endpoints info
        rumps.alert(
            title="API Documentation",
            message=(
                f"Base URL: http://127.0.0.1:{port}\n\n"
                "Endpoints:\n"
                "• POST /api/transcribe - Transcribe audio file\n"
                "• POST /api/segment - Extract audio segment\n"
                "• GET /api/models - List available models\n\n"
                "Example:\n"
                f"curl -X POST -F 'file=@audio.mp3' http://127.0.0.1:{port}/api/transcribe"
            )
        )

    def set_server_port(self, port):
        """Set the server API port."""
        self.config["server_port"] = port
        save_config(self.config)
        self._refresh_server_menu()

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Server Port Updated",
                subtitle="",
                message=f"API port set to {port}. Restart server to apply.",
                sound=False
            )

    def set_gradio_port(self, port):
        """Set the Gradio web UI port."""
        self.config["gradio_port"] = port
        save_config(self.config)
        self._refresh_server_menu()

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Gradio Port Updated",
                subtitle="",
                message=f"Web UI port set to {port}. Restart server to apply.",
                sound=False
            )

    def toggle_debug_mode(self, _):
        """Toggle server debug mode."""
        current = self.config.get("server_debug", False)
        self.config["server_debug"] = not current
        save_config(self.config)
        self._refresh_server_menu()

    # === Cancel Recording ===

    def cancel_recording(self, _):
        """Cancel the current recording without processing."""
        if not self.recording:
            return

        logger.info("Recording cancelled by user")
        self.recording = False

        # Stop the audio stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
            self._stream = None

        # Clear audio data
        self._audio_data = []

        # Reset UI
        self.title = self.ICON_IDLE
        self.record_button.title = "🎤 Start Recording"
        self.cancel_button.set_callback(None)  # Disable cancel button

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Recording Cancelled",
                subtitle="",
                message="Recording was cancelled",
                sound=False
            )

    # === Transcribe File ===

    def transcribe_file(self, _):
        """Open file picker and transcribe selected audio file."""
        if self.recording or self.processing:
            rumps.notification(
                title="Busy",
                subtitle="",
                message="Please wait for current operation to complete",
                sound=False
            )
            return

        # Use AppleScript to open file picker
        script = '''
        set theFile to choose file with prompt "Select an audio file to transcribe:" of type {"public.audio", "com.apple.m4a-audio", "public.mp3", "com.microsoft.waveform-audio"}
        return POSIX path of theFile
        '''

        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True
            )

            file_path = result.stdout.strip()

            if file_path and os.path.exists(file_path):
                logger.info(f"Transcribing file: {file_path}")
                self._transcribe_file_path(file_path)
            elif result.returncode != 0:
                # User cancelled the dialog
                pass

        except Exception as e:
            logger.error(f"File picker error: {e}", exc_info=True)
            rumps.alert(
                title="Error",
                message=f"Could not open file picker: {e}"
            )

    def _transcribe_file_path(self, file_path):
        """Transcribe an audio file at the given path."""
        self.processing = True
        self.title = self.ICON_PROCESSING
        self.status_item.title = "Transcribing file..."

        def do_transcribe():
            try:
                # Wait for transcriber if needed
                wait_count = 0
                while self.transcriber is None and wait_count < 60:
                    time.sleep(0.5)
                    wait_count += 1

                if self.transcriber is None:
                    raise Exception("Model not loaded. Please wait and try again.")

                # Get file info
                file_name = os.path.basename(file_path)
                from pydub import AudioSegment
                audio = AudioSegment.from_file(file_path)
                duration = audio.duration_seconds

                logger.info(f"Transcribing: {file_name} ({duration:.1f}s)")

                # Transcribe
                chunk_duration = self.config.get("default_chunk_duration", 120)
                df, full_text = self.transcriber.transcribe(
                    file_path,
                    chunk_duration=chunk_duration
                )

                if full_text is None:
                    full_text = ""

                # Handle diarization if enabled
                output_text = full_text
                num_speakers = 0

                if self.config.get("diarization_enabled", False) and df is not None:
                    try:
                        self.status_item.title = "Identifying speakers..."
                        from parakeet_mlx_guiapi.diarization import SpeakerDiarizer

                        if not hasattr(self, '_diarizer') or self._diarizer is None:
                            self._diarizer = SpeakerDiarizer()

                        configured_speakers = self.config.get("diarization_num_speakers", 0)

                        if configured_speakers > 0:
                            diarization = self._diarizer.diarize(
                                file_path,
                                num_speakers=configured_speakers
                            )
                        else:
                            diarization = self._diarizer.diarize(file_path)

                        num_speakers = diarization.num_speakers
                        segments = df.to_dict('records')
                        output_text = diarization.format_transcript_markdown(segments)

                    except Exception as e:
                        logger.error(f"Diarization failed: {e}")
                        output_text = full_text

                if output_text:
                    # Copy to clipboard if enabled
                    if self.config.get("auto_copy_clipboard", True):
                        pyperclip.copy(output_text)

                    # Add to history
                    self._add_to_history(output_text, duration)

                    # Show notification
                    if self.config.get("show_notifications", True):
                        preview = output_text[:80] + "..." if len(output_text) > 80 else output_text
                        copied_msg = " - Copied!" if self.config.get("auto_copy_clipboard", True) else ""
                        speaker_info = f" ({num_speakers} speakers)" if num_speakers > 0 else ""
                        rumps.notification(
                            title=f"Transcription Complete{copied_msg}",
                            subtitle=f"{file_name}{speaker_info}",
                            message=preview,
                            sound=True
                        )

                    self.title = self.ICON_READY
                    threading.Timer(2.0, lambda: setattr(self, 'title', self.ICON_IDLE)).start()
                else:
                    rumps.notification(
                        title="Transcription Empty",
                        subtitle="",
                        message="No speech detected in the audio file",
                        sound=True
                    )
                    self.title = self.ICON_IDLE

            except Exception as e:
                logger.error(f"File transcription error: {e}", exc_info=True)
                rumps.notification(
                    title="Transcription Error",
                    subtitle="",
                    message=str(e)[:100],
                    sound=True
                )
                self.title = self.ICON_ERROR
                threading.Timer(2.0, lambda: setattr(self, 'title', self.ICON_IDLE)).start()
            finally:
                self.processing = False
                model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
                self.status_item.title = f"Ready: {self._get_model_short_name(model_name)}"

        threading.Thread(target=do_transcribe, daemon=True).start()

    # === Help ===

    def show_help(self, _):
        """Show help information."""
        rumps.alert(
            title="Parakeet Help",
            message=(
                "QUICK START\n"
                "• Click 🎤 to start recording\n"
                "• Click again to stop & transcribe\n"
                "• Text is copied to clipboard automatically\n\n"
                "MENU OPTIONS\n"
                "• Transcribe File: Pick an audio file\n"
                "• Server: Start/stop the web API\n"
                "• Model: Change transcription model\n"
                "• Settings: Configure diarization, etc.\n"
                "• History: View recent transcriptions\n\n"
                "KEYBOARD TIPS\n"
                "The menu bar icon is always accessible.\n"
                "Use with Alfred/Raycast for quick access.\n\n"
                "LOGS & DEBUGGING\n"
                f"Log file: {LOG_PATH}\n"
                "Settings > Advanced > View Logs\n\n"
                "NEED MORE HELP?\n"
                "Visit: github.com/senstella/parakeet-mlx"
            )
        )

    def show_about(self, _):
        """Show about dialog."""
        current_model_id = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
        current = self._get_model_by_id(current_model_id)

        if current:
            model_info = (
                f"Current model: {current['name']}\n"
                f"  Languages: {current.get('languages', 'Unknown')}\n"
                f"  Accuracy (WER): {current.get('wer', 'N/A')}\n"
                f"  Speed: {current.get('speed', 'N/A')}\n"
                f"  Size: {current.get('size', 'Unknown')}"
            )
        else:
            model_info = f"Current model: {current_model_id}"

        # Server status
        if self._server_process and self._server_process.poll() is None:
            server_status = f"Server: Running (port {self._server_port})"
        else:
            server_status = "Server: Stopped"

        rumps.alert(
            title="Parakeet Voice-to-Clipboard",
            message=(
                "Quick voice transcription for macOS.\n\n"
                "Features:\n"
                "• Voice recording to clipboard\n"
                "• File transcription\n"
                "• Speaker diarization (who spoke when)\n"
                "• Web API server\n\n"
                f"{model_info}\n\n"
                f"{server_status}\n\n"
                "Model Types:\n"
                "• TDT: Best accuracy\n"
                "• CTC: Fastest inference\n"
                "• Hybrid: Long audio support\n\n"
                "Powered by NVIDIA Parakeet + Apple MLX\n"
                "https://github.com/senstella/parakeet-mlx"
            )
        )

    def quit_app(self, _):
        """Quit the application."""
        # Stop recording if active
        if self.recording:
            self.recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()

        # Stop server if running
        if self._server_process and self._server_process.poll() is None:
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                self._server_process.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                except Exception:
                    pass

        rumps.quit_application()


def main():
    """Run the menu bar app."""
    import platform

    # Log startup info for debugging
    logger.info("=" * 60)
    logger.info("Parakeet Menu Bar App Starting")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Log file: {LOG_PATH}")
    logger.info("-" * 60)

    ParakeetMenuBarApp().run()


if __name__ == "__main__":
    main()
