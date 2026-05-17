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
import requests

# Setup logging to file
LOG_PATH = Path.home() / ".parakeet_mlx.log"
DAEMON_BASE_URL = "http://localhost:8080"
DAEMON_LABEL = "com.gui.parakeet"
DAEMON_PLIST = os.path.expanduser("~/Library/LaunchAgents/com.gui.parakeet.plist")
DAEMON_STDERR_LOG = "/Users/gui/dev/parakeet-mlx_guiapi/stderr.log"
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

# Available STT providers
AVAILABLE_PROVIDERS = [
    {
        "id": "parakeet",
        "name": "Parakeet-MLX (Local)",
        "description": "Local transcription on Apple Silicon",
        "requires_api_key": False,
        "models": AVAILABLE_MODELS,  # Uses the AVAILABLE_MODELS list
    },
    {
        "id": "deepgram",
        "name": "Deepgram (Cloud)",
        "description": "Cloud-based high-accuracy transcription",
        "requires_api_key": True,
        "models": [
            # Nova-3 models (latest, best accuracy)
            {"id": "nova-3", "name": "Nova-3 (General)", "description": "Latest, best accuracy", "category": "Nova-3"},
            {"id": "nova-3-meeting", "name": "Nova-3 Meeting", "description": "Meetings & conferences", "category": "Nova-3"},
            {"id": "nova-3-phonecall", "name": "Nova-3 Phone", "description": "Phone calls", "category": "Nova-3"},
            {"id": "nova-3-voicemail", "name": "Nova-3 Voicemail", "description": "Voicemails", "category": "Nova-3"},
            {"id": "nova-3-finance", "name": "Nova-3 Finance", "description": "Finance terminology", "category": "Nova-3"},
            {"id": "nova-3-medical", "name": "Nova-3 Medical", "description": "Medical terminology", "category": "Nova-3"},
            # Nova-2 models (still excellent)
            {"id": "nova-2", "name": "Nova-2 (General)", "description": "Proven general-purpose", "category": "Nova-2"},
            {"id": "nova-2-meeting", "name": "Nova-2 Meeting", "description": "Meetings", "category": "Nova-2"},
            {"id": "nova-2-phonecall", "name": "Nova-2 Phone", "description": "Phone calls", "category": "Nova-2"},
            {"id": "nova-2-voicemail", "name": "Nova-2 Voicemail", "description": "Voicemails", "category": "Nova-2"},
            {"id": "nova-2-finance", "name": "Nova-2 Finance", "description": "Finance", "category": "Nova-2"},
            {"id": "nova-2-medical", "name": "Nova-2 Medical", "description": "Medical", "category": "Nova-2"},
        ],
        # Deepgram-specific configurable options
        "options": {
            "smart_format": {"name": "Smart Format", "description": "Auto-capitalize, format numbers", "default": True},
            "punctuate": {"name": "Punctuation", "description": "Add punctuation marks", "default": True},
            "paragraphs": {"name": "Paragraphs", "description": "Group text into paragraphs", "default": True},
            "profanity_filter": {"name": "Profanity Filter", "description": "Filter profane words", "default": False},
            "numerals": {"name": "Numerals", "description": "Convert numbers to digits", "default": False},
        },
    },
]


def get_provider_by_id(provider_id):
    """Get provider dict by its ID."""
    for provider in AVAILABLE_PROVIDERS:
        if provider["id"] == provider_id:
            return provider
    return None


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


class TranscriptionClient:
    """Thin HTTP client for the launchd daemon's /api/transcribe endpoint."""

    def __init__(self, base_url=DAEMON_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def transcribe(self, wav_bytes, config, recording_duration):
        """POST WAV bytes to /api/transcribe and return parsed JSON.

        Raises requests exceptions on network/timeout/non-2xx so callers can
        surface the failure to the user.
        """
        provider = config.get("stt_provider", "parakeet")
        if provider == "parakeet":
            model = config.get("model_name", "")
        else:
            model = config.get("deepgram_model", "nova-3")

        files = {
            "file": ("recording.wav", wav_bytes, "audio/wav"),
        }
        data = {
            "provider": provider,
            "model": model,
            "deepgram_options": json.dumps(config.get("deepgram_options", {})),
            "enable_diarization": str(config.get("diarization_enabled", False)).lower(),
            "chunk_duration": str(config.get("default_chunk_duration", 120)),
            "output_format": "json",
        }

        timeout = max(60, int(recording_duration * 2))
        url = f"{self.base_url}/api/transcribe"
        logger.info(
            "TranscriptionClient: POST %s provider=%s model=%s diar=%s timeout=%ds",
            url, provider, model, data["enable_diarization"], timeout,
        )

        response = requests.post(url, files=files, data=data, timeout=timeout)
        response.raise_for_status()
        return response.json()


class DaemonHealth:
    """Periodic health probe for the launchd daemon at localhost:8080."""

    def __init__(self, base_url=DAEMON_BASE_URL, interval=30.0, on_change=None):
        self.base_url = base_url.rstrip("/")
        self.interval = interval
        self.on_change = on_change  # callback(is_online: bool) when state changes
        self._stop = threading.Event()
        self._thread = None
        self._online = None  # None = unknown, True/False once checked

    @property
    def online(self):
        return bool(self._online)

    def check_once(self):
        """Single synchronous health check. Returns True if daemon answers."""
        try:
            r = requests.get(f"{self.base_url}/api/models", timeout=2.0)
            return r.status_code == 200
        except Exception as e:
            logger.debug("DaemonHealth: probe failed: %s", e)
            return False

    def _set(self, is_online):
        prev = self._online
        self._online = is_online
        if prev != is_online and self.on_change is not None:
            try:
                self.on_change(is_online)
            except Exception as e:
                logger.warning("DaemonHealth on_change callback failed: %s", e)

    def start(self):
        """Run an initial check, then loop every `interval` seconds until stop()."""
        def loop():
            self._set(self.check_once())
            while not self._stop.wait(self.interval):
                self._set(self.check_once())

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


def _run_launchctl(args):
    """Run a launchctl command, log it, return (returncode, stdout, stderr)."""
    cmd = ["launchctl"] + list(args)
    logger.info("launchctl: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(
            "launchctl rc=%d stdout=%r stderr=%r",
            result.returncode, result.stdout.strip(), result.stderr.strip(),
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        logger.error("launchctl invocation failed: %s", e)
        return 1, "", str(e)


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
        self._stream = None
        self._audio_data = []
        self._recording_start_time = None
        self._timer = None

        # Load config
        self.config = get_config()

        # History of transcriptions (last 10)
        self.history = []
        self._load_history()

        # Error tracking for debugging
        self._last_error = None

        # HTTP client + daemon health probe
        self.client = TranscriptionClient(DAEMON_BASE_URL)
        self._offline_notified = False
        self.daemon_health = DaemonHealth(
            DAEMON_BASE_URL,
            interval=30.0,
            on_change=self._on_daemon_state_change,
        )

        # Build menu — rumps requires menu items to be created here
        self._setup_menu()

        # Start background health polling (initial check + every 30s)
        self.daemon_health.start()

    def _on_daemon_state_change(self, is_online):
        """Callback from DaemonHealth when the daemon flips state."""
        try:
            if is_online:
                self.status_item.title = "Daemon: ● ready"
                self._offline_notified = False
            else:
                self.status_item.title = "Daemon: ○ offline"
                if not self._offline_notified:
                    self._offline_notified = True
                    rumps.notification(
                        title="Daemon offline",
                        subtitle="",
                        message="Server > Start to launch the daemon.",
                        sound=False,
                    )
            # Server submenu shows status — refresh it
            self._refresh_server_menu()
        except Exception as e:
            logger.warning("on_daemon_state_change failed: %s", e)

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
        self.status_item = rumps.MenuItem("Daemon: ○ checking…", callback=self.status_clicked)

        # === Server Controls ===
        self.server_menu = rumps.MenuItem("🌐 Server")
        self._populate_server_menu()

        # === Provider Selection ===
        self.provider_menu = rumps.MenuItem("🔊 Provider")
        self._populate_provider_menu()

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
            self.provider_menu,
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
        """Populate the server control menu (controls the launchd daemon)."""
        # Daemon liveness from the health helper
        is_online = getattr(self, "daemon_health", None) and self.daemon_health.online
        if is_online:
            self.server_menu.add(rumps.MenuItem("✅ ● Daemon Running"))
        else:
            self.server_menu.add(rumps.MenuItem("⚪ ○ Daemon Stopped"))

        self.server_menu.add(None)

        # launchctl actions — always available; daemon decides validity
        self.server_menu.add(rumps.MenuItem("▶️ Start Daemon", callback=self.start_server))
        self.server_menu.add(rumps.MenuItem("⏹ Stop Daemon", callback=self.stop_server))
        self.server_menu.add(rumps.MenuItem("🔄 Restart Daemon", callback=self.restart_server))

        self.server_menu.add(None)

        # Quick links
        self.server_menu.add(rumps.MenuItem("🎤 Live Transcription", callback=self.open_live_transcription))
        self.server_menu.add(rumps.MenuItem("🌐 Open Web UI", callback=self.open_web_ui))
        self.server_menu.add(rumps.MenuItem("📊 Open API Docs", callback=self.open_api_docs))
        self.server_menu.add(rumps.MenuItem("📝 View Daemon Logs", callback=self.view_daemon_logs))

    def _refresh_server_menu(self):
        """Refresh the server menu."""
        keys = list(self.server_menu.keys())
        for key in keys:
            del self.server_menu[key]
        self._populate_server_menu()

    def _populate_provider_menu(self):
        """Populate the STT provider selection menu - simple flat list."""
        current_provider = self.config.get("stt_provider", "parakeet")

        # Simple flat list of providers (like radio buttons)
        for provider in AVAILABLE_PROVIDERS:
            is_selected = provider["id"] == current_provider
            title = provider["name"]
            if is_selected:
                title = f"✓ {title}"

            self.provider_menu.add(rumps.MenuItem(
                title,
                callback=lambda _, p=provider: self.select_provider(p)
            ))

        # Show current provider info
        self.provider_menu.add(None)
        current = get_provider_by_id(current_provider)
        if current:
            self.provider_menu.add(rumps.MenuItem(f"📝 {current['description']}"))

    def _refresh_provider_menu(self):
        """Refresh the provider menu."""
        keys = list(self.provider_menu.keys())
        for key in keys:
            del self.provider_menu[key]
        self._populate_provider_menu()

    def select_provider(self, provider):
        """Switch to a different STT provider."""
        if self.recording or self.processing:
            rumps.notification(
                title="Cannot Change Provider",
                subtitle="",
                message="Please wait until current operation completes",
                sound=False
            )
            return

        provider_id = provider["id"]

        # For Deepgram, check API key
        if provider_id == "deepgram":
            api_key = self.config.get("deepgram_api_key", "")
            if not api_key:
                response = rumps.alert(
                    title="API Key Required",
                    message="Deepgram requires an API key to function.\n\nWould you like to configure it now?",
                    ok="Configure",
                    cancel="Cancel"
                )
                if response == 1:
                    self.configure_deepgram_api_key(None)
                return

        # Update config
        self.config["stt_provider"] = provider_id
        save_config(self.config)

        # Refresh menus
        self._refresh_provider_menu()
        self._refresh_model_menu()

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Provider Changed",
                subtitle="",
                message=f"Now using {provider['name']}",
                sound=False
            )

        logger.info(f"Switched to provider: {provider_id}")

    def select_deepgram_model(self, model):
        """Select a Deepgram model."""
        self.config["deepgram_model"] = model["id"]
        save_config(self.config)
        self._refresh_model_menu()

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Model Changed",
                subtitle="",
                message=f"Deepgram model: {model['name']}",
                sound=False
            )

        logger.info(f"Selected Deepgram model: {model['id']}")

    def configure_deepgram_api_key(self, _):
        """Configure Deepgram API key."""
        current_key = self.config.get("deepgram_api_key", "")

        # Use rumps.Window for text input
        window = rumps.Window(
            title="Deepgram API Key",
            message="Enter your Deepgram API key.\n\nGet a free key at: console.deepgram.com",
            default_text=current_key,
            ok="Save",
            cancel="Cancel",
            dimensions=(320, 24)
        )

        # Add a button to open the console
        response = window.run()

        if response.clicked:
            new_key = response.text.strip()
            if new_key:
                self.config["deepgram_api_key"] = new_key
                save_config(self.config)
                self._refresh_provider_menu()

                if self.config.get("show_notifications", True):
                    rumps.notification(
                        title="API Key Saved",
                        subtitle="",
                        message="Deepgram API key has been configured",
                        sound=False
                    )
                logger.info("Deepgram API key saved")
            else:
                rumps.alert(
                    title="No Key Entered",
                    message="API key was not saved because no key was entered."
                )

    def open_deepgram_console(self, _):
        """Open Deepgram console in browser."""
        webbrowser.open("https://console.deepgram.com")

    def configure_huggingface_token(self, _):
        """Configure HuggingFace token for diarization."""
        current_token = self.config.get("huggingface_token", "")

        window = rumps.Window(
            title="HuggingFace Token",
            message="Enter your HuggingFace token.\n\nRequired for speaker diarization.\nGet a token at: huggingface.co/settings/tokens",
            default_text=current_token,
            ok="Save",
            cancel="Cancel",
            dimensions=(320, 24)
        )

        response = window.run()

        if response.clicked:
            new_token = response.text.strip()
            if new_token:
                self.config["huggingface_token"] = new_token
                save_config(self.config)
                self._refresh_settings_menu()

                if self.config.get("show_notifications", True):
                    rumps.notification(
                        title="Token Saved",
                        subtitle="",
                        message="HuggingFace token has been configured",
                        sound=False
                    )
                logger.info("HuggingFace token saved")

    def _populate_model_menu(self):
        """Populate the model selection menu based on current provider."""
        current_provider = self.config.get("stt_provider", "parakeet")

        if current_provider == "parakeet":
            self._populate_parakeet_models()
        elif current_provider == "deepgram":
            self._populate_deepgram_models()

    def _populate_parakeet_models(self):
        """Populate Parakeet model menu organized by category."""
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

    def _populate_deepgram_models(self):
        """Populate Deepgram model menu."""
        current_model = self.config.get("deepgram_model", "nova-2")
        deepgram_provider = get_provider_by_id("deepgram")

        if not deepgram_provider:
            return

        # List all Deepgram models
        for model in deepgram_provider["models"]:
            title = model["name"]
            if model["id"] == current_model:
                title = f"✓ {title}"

            self.model_menu.add(rumps.MenuItem(
                title,
                callback=lambda _, m=model: self.select_deepgram_model(m)
            ))

        # Add separator and current model info
        self.model_menu.add(None)

        # Find current model info
        current = None
        for m in deepgram_provider["models"]:
            if m["id"] == current_model:
                current = m
                break

        if current:
            self.model_menu.add(rumps.MenuItem(f"Current: {current['name']}"))
            if current.get("description"):
                self.model_menu.add(rumps.MenuItem(f"📝 {current['description']}"))

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

    def predownload_model(self, _):
        """Models are downloaded by the daemon on first use."""
        rumps.notification(
            title="Daemon Handles Downloads",
            subtitle="",
            message="Models are fetched lazily by the daemon on first use.",
            sound=False,
        )

    def reload_model(self, _):
        """Restart the daemon to reload the model with new config."""
        rc, _stdout, stderr = _run_launchctl(
            ["kickstart", "-k", f"gui/{os.getuid()}/com.gui.parakeet"]
        )
        if rc == 0:
            rumps.notification(
                title="Daemon Restarted",
                subtitle="",
                message="Model will reload on next request.",
                sound=False,
            )
        else:
            rumps.notification(
                title="Restart Failed",
                subtitle="",
                message=(stderr or "Unknown error")[:100],
                sound=True,
            )

    def start_diarization_setup(self, _):
        """Diarization dependencies are owned by the daemon."""
        rumps.notification(
            title="Daemon Handles Diarization",
            subtitle="",
            message="Pyannote install + HF token are configured on the daemon side.",
            sound=False,
        )

    # Daemon now owns diarization availability/dependency checks.
    # These stubs preserve _populate_settings_menu's call sites without
    # re-introducing local pyannote/HF probes in the thin client.
    def _check_diarization_available(self):
        return True, "Handled by daemon"

    def _check_all_models_accessible(self):
        return []

    def _check_diarization_components(self):
        return True, True

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

        # === Provider-specific options ===
        current_provider = self.config.get("stt_provider", "parakeet")

        # Deepgram Options (only show when Deepgram is selected)
        if current_provider == "deepgram":
            deepgram_options_menu = rumps.MenuItem("🔧 Deepgram Options")
            self._populate_deepgram_options_menu(deepgram_options_menu)
            self.settings_menu.add(deepgram_options_menu)
            self.settings_menu.add(None)

        # === Chunk duration options (for Parakeet) ===
        if current_provider == "parakeet":
            parakeet_options_menu = rumps.MenuItem("🔧 Parakeet Options")
            self._populate_parakeet_options_menu(parakeet_options_menu)
            self.settings_menu.add(parakeet_options_menu)
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

        # === Microphone Selection ===
        self.settings_menu.add(None)
        mic_menu = rumps.MenuItem("🎙️ Microphone")
        self._populate_microphone_menu(mic_menu)
        self.settings_menu.add(mic_menu)

        # === API Keys section ===
        self.settings_menu.add(None)
        api_keys_menu = rumps.MenuItem("🔑 API Keys")

        # Deepgram API key
        deepgram_key = self.config.get("deepgram_api_key", "")
        if deepgram_key:
            key_preview = deepgram_key[:8] + "..." if len(deepgram_key) > 8 else deepgram_key
            api_keys_menu.add(rumps.MenuItem(f"Deepgram: {key_preview}"))
        else:
            api_keys_menu.add(rumps.MenuItem("Deepgram: Not configured"))
        api_keys_menu.add(rumps.MenuItem("Configure Deepgram Key...", callback=self.configure_deepgram_api_key))
        api_keys_menu.add(rumps.MenuItem("Get Deepgram Key", callback=self.open_deepgram_console))

        api_keys_menu.add(None)

        # HuggingFace token (for diarization)
        hf_token = self.config.get("huggingface_token", "")
        if hf_token:
            token_preview = hf_token[:8] + "..." if len(hf_token) > 8 else hf_token
            api_keys_menu.add(rumps.MenuItem(f"HuggingFace: {token_preview}"))
        else:
            api_keys_menu.add(rumps.MenuItem("HuggingFace: Not configured"))
        api_keys_menu.add(rumps.MenuItem("Configure HuggingFace Token...", callback=self.configure_huggingface_token))

        self.settings_menu.add(api_keys_menu)

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

    def _populate_deepgram_options_menu(self, menu):
        """Populate the Deepgram options submenu."""
        # Get Deepgram provider info
        deepgram_provider = get_provider_by_id("deepgram")
        if not deepgram_provider:
            return

        options = deepgram_provider.get("options", {})
        current_options = self.config.get("deepgram_options", {})

        # Add header
        menu.add(rumps.MenuItem("Formatting Options:"))
        menu.add(None)

        # Add toggleable options
        for opt_key, opt_info in options.items():
            # Get current value (default from provider definition)
            is_enabled = current_options.get(opt_key, opt_info.get("default", False))
            title = f"{'✓ ' if is_enabled else ''}{opt_info['name']}"
            menu.add(rumps.MenuItem(
                title,
                callback=lambda _, k=opt_key: self.toggle_deepgram_option(k)
            ))

        # Add info about what these options do
        menu.add(None)
        menu.add(rumps.MenuItem("ℹ️ Changes apply to next transcription"))

    def toggle_deepgram_option(self, option_key):
        """Toggle a Deepgram option."""
        # Get current options
        current_options = self.config.get("deepgram_options", {})

        # Get default from provider definition
        deepgram_provider = get_provider_by_id("deepgram")
        options = deepgram_provider.get("options", {})
        default_value = options.get(option_key, {}).get("default", False)

        # Toggle the option
        current_value = current_options.get(option_key, default_value)
        current_options[option_key] = not current_value

        # Save to config
        self.config["deepgram_options"] = current_options
        save_config(self.config)

        # Refresh the settings menu
        self._refresh_settings_menu()

        # Show notification
        option_name = options.get(option_key, {}).get("name", option_key)
        status = "enabled" if current_options[option_key] else "disabled"
        logger.info(f"Deepgram option '{option_name}' {status}")

    def _populate_parakeet_options_menu(self, menu):
        """Populate the Parakeet options submenu."""
        # Chunk duration submenu
        chunk_menu = rumps.MenuItem("Chunk Duration")
        chunk_options = [30, 60, 120, 180, 300]
        current_chunk = self.config.get("default_chunk_duration", 120)

        for duration in chunk_options:
            title = f"{duration}s"
            if duration == current_chunk:
                title = f"✓ {title}"
            chunk_menu.add(rumps.MenuItem(
                title,
                callback=lambda _, d=duration: self.set_chunk_duration(d)
            ))
        menu.add(chunk_menu)

        # Language selection submenu (for multilingual models)
        current_model = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
        model_info = self._get_model_by_id(current_model)

        if model_info and "lang_list" in model_info and len(model_info["lang_list"]) > 1:
            lang_menu = rumps.MenuItem("Language")
            current_lang = self.config.get("parakeet_language", "auto")

            # Language names
            lang_names = {
                "auto": "Auto-detect",
                "en": "English",
                "fr": "French",
                "de": "German",
                "es": "Spanish",
                "it": "Italian",
                "pt": "Portuguese",
                "nl": "Dutch",
                "pl": "Polish",
                "ru": "Russian",
                "uk": "Ukrainian",
                "cs": "Czech",
                "sk": "Slovak",
                "bg": "Bulgarian",
                "hr": "Croatian",
                "da": "Danish",
                "et": "Estonian",
                "fi": "Finnish",
                "el": "Greek",
                "hu": "Hungarian",
                "lv": "Latvian",
                "lt": "Lithuanian",
                "mt": "Maltese",
                "ro": "Romanian",
                "sl": "Slovenian",
                "sv": "Swedish",
            }

            # Add auto-detect option
            auto_title = "✓ Auto-detect" if current_lang == "auto" else "Auto-detect"
            lang_menu.add(rumps.MenuItem(
                auto_title,
                callback=lambda _: self.set_parakeet_language("auto")
            ))
            lang_menu.add(None)

            # Add supported languages
            for lang_code in model_info["lang_list"]:
                lang_name = lang_names.get(lang_code, lang_code.upper())
                title = f"{'✓ ' if current_lang == lang_code else ''}{lang_name}"
                lang_menu.add(rumps.MenuItem(
                    title,
                    callback=lambda _, l=lang_code: self.set_parakeet_language(l)
                ))

            menu.add(lang_menu)

        # Info about current model
        menu.add(None)
        if model_info:
            menu.add(rumps.MenuItem(f"📝 Model: {model_info.get('name', current_model)}"))
            if "wer" in model_info:
                menu.add(rumps.MenuItem(f"📊 WER: {model_info['wer']}"))

    def set_parakeet_language(self, lang_code):
        """Set the Parakeet transcription language."""
        self.config["parakeet_language"] = lang_code
        save_config(self.config)
        self._refresh_settings_menu()

        lang_name = "Auto-detect" if lang_code == "auto" else lang_code.upper()
        logger.info(f"Parakeet language set to: {lang_name}")

    def _populate_microphone_menu(self, menu):
        """Populate the microphone selection submenu."""
        input_devices = self._get_input_devices()
        selected_device = self.config.get("selected_microphone", None)
        default_device = self._get_default_input_device()

        if not input_devices:
            menu.add(rumps.MenuItem("No input devices found"))
            return

        # System Default option
        is_default_selected = selected_device is None
        default_title = "✓ System Default" if is_default_selected else "System Default"
        if default_device is not None:
            # Find the name of the default device
            default_name = next((d['name'] for d in input_devices if d['index'] == default_device), "Unknown")
            default_title += f" ({default_name})"
        menu.add(rumps.MenuItem(
            default_title,
            callback=lambda _: self.select_microphone(None)
        ))
        menu.add(None)

        # List all input devices
        for device in input_devices:
            is_selected = selected_device == device['index']
            title = f"{'✓ ' if is_selected else ''}{device['name']}"
            menu.add(rumps.MenuItem(
                title,
                callback=lambda _, idx=device['index']: self.select_microphone(idx)
            ))

        # Show currently active device
        menu.add(None)
        if selected_device is not None:
            active_name = next((d['name'] for d in input_devices if d['index'] == selected_device), "Unknown")
        else:
            active_name = next((d['name'] for d in input_devices if d['index'] == default_device), "System Default")
        menu.add(rumps.MenuItem(f"📍 Active: {active_name}"))

    def select_microphone(self, device_index):
        """Select a microphone device."""
        self.config["selected_microphone"] = device_index
        save_config(self.config)
        self._refresh_settings_menu()

        if device_index is None:
            logger.info("Microphone set to: System Default")
        else:
            devices = self._get_input_devices()
            device_name = next((d['name'] for d in devices if d['index'] == device_index), f"Device {device_index}")
            logger.info(f"Microphone set to: {device_name}")

    def _get_input_devices(self):
        """Get list of available input devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = []
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    input_devices.append({
                        'index': i,
                        'name': d['name'],
                        'channels': d['max_input_channels']
                    })
            return input_devices
        except Exception as e:
            logger.error(f"Error getting input devices: {e}")
            return []

    def _get_default_input_device(self):
        """Get the default input device index."""
        try:
            import sounddevice as sd
            return sd.default.device[0]  # Returns (input, output) tuple
        except Exception:
            return None

    def toggle_diarization(self, _):
        """Toggle speaker diarization (daemon owns model availability)."""
        current = self.config.get("diarization_enabled", False)
        self.config["diarization_enabled"] = not current
        save_config(self.config)
        self._refresh_settings_menu()

        status = "enabled" if not current else "disabled"
        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Speaker Diarization",
                subtitle=status.capitalize(),
                message=(
                    "Transcripts will include speaker labels"
                    if not current
                    else "Speaker labels disabled"
                ),
                sound=False,
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
        """Handle click on status item — show daemon state."""
        is_online = getattr(self, "daemon_health", None) and self.daemon_health.online
        if is_online:
            model_name = self.config.get("model_name", AVAILABLE_MODELS[0]["id"])
            provider = self.config.get("stt_provider", "parakeet")
            rumps.alert(
                title="Daemon Ready",
                message=(
                    f"Provider: {provider}\n"
                    f"Model (parakeet): {model_name}\n"
                    f"Daemon: {DAEMON_BASE_URL}\n\n"
                    "Click the mic icon to start recording!"
                ),
            )
        else:
            response = rumps.alert(
                title="Daemon Offline",
                message=(
                    f"No response from {DAEMON_BASE_URL}.\n\n"
                    "Start it from the Server submenu, or view logs."
                ),
                ok="Start Daemon",
                cancel="View Logs",
            )
            if response == 1:
                self.start_server(None)
            else:
                self.view_daemon_logs(None)

    def restart_daemon(self, _):
        """Kickstart the launchd daemon (replaces old reload_model)."""
        self.restart_server(None)

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

    def select_model(self, model):
        """Change the transcription model (config-only; daemon picks it up per request)."""
        if self.recording or self.processing:
            rumps.notification(
                title="Cannot Change Model",
                subtitle="",
                message="Please wait until current operation completes",
                sound=False,
            )
            return

        self.config["model_name"] = model["id"]
        save_config(self.config)
        self._refresh_model_menu()

        if self.config.get("show_notifications", True):
            rumps.notification(
                title="Model Selected",
                subtitle=model["name"],
                message="Daemon will use this model on the next request.",
                sound=False,
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
        # Preflight: don't start recording if the daemon is known-offline.
        # check_once() is cheap (2s timeout) and avoids the case where the
        # 30s background poller hasn't observed a recent failure yet.
        if getattr(self, "daemon_health", None) is not None:
            if self.daemon_health._online is False or not self.daemon_health.check_once():
                logger.warning("start_recording: daemon offline — aborting")
                rumps.notification(
                    title="Daemon offline",
                    subtitle="",
                    message="Start it from the Server submenu, then try again.",
                    sound=True,
                )
                self.daemon_health._set(False)
                return

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

            # Start recording stream with selected microphone
            selected_device = self.config.get("selected_microphone", None)
            device_name = "System Default"
            if selected_device is not None:
                devices = self._get_input_devices()
                device_name = next((d['name'] for d in devices if d['index'] == selected_device), f"Device {selected_device}")

            logger.info(f"start_recording: Creating InputStream (device={device_name}, rate={self.sample_rate}, channels={self.channels})")
            self._stream = sd.InputStream(
                device=selected_device,  # None = system default
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=audio_callback
            )
            self._stream.start()
            logger.info(f"start_recording: Stream started successfully on {device_name}")

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
        """Encode recorded audio as WAV and POST it to the daemon."""
        import numpy as np
        from scipy.io import wavfile
        import io

        process_start = time.time()
        logger.info(
            "_process_audio: starting for %.1fs recording", recording_duration
        )

        try:
            # Concatenate audio chunks → WAV bytes (no temp file needed)
            audio_data = np.concatenate(self._audio_data, axis=0)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            buf = io.BytesIO()
            wavfile.write(buf, self.sample_rate, audio_int16)
            wav_bytes = buf.getvalue()
            logger.info("_process_audio: WAV encoded, %d bytes", len(wav_bytes))

            # POST to daemon
            transcribe_start = time.time()
            payload = self.client.transcribe(wav_bytes, self.config, recording_duration)
            transcribe_time = time.time() - transcribe_start
            logger.info(
                "_process_audio: daemon responded in %.2fs", transcribe_time
            )

            output_text = (payload.get("text") or "").strip()
            # Best-effort speaker count from segments (daemon-side diarization)
            num_speakers = 0
            segments = payload.get("segments") or []
            if isinstance(segments, list) and segments:
                speakers = {
                    s.get("Speaker") or s.get("speaker")
                    for s in segments
                    if isinstance(s, dict)
                }
                speakers.discard(None)
                num_speakers = len(speakers)

            if output_text:
                if self.config.get("auto_copy_clipboard", True):
                    pyperclip.copy(output_text)

                self._add_to_history(output_text, recording_duration)

                if self.config.get("show_notifications", True):
                    preview = (
                        output_text[:80] + "..." if len(output_text) > 80 else output_text
                    )
                    copied_msg = (
                        " - Copied!" if self.config.get("auto_copy_clipboard", True) else ""
                    )
                    speaker_info = (
                        f" ({num_speakers} speakers)" if num_speakers > 1 else ""
                    )
                    rumps.notification(
                        title=f"Transcription Complete{copied_msg}",
                        subtitle=f"{recording_duration:.1f}s of audio{speaker_info}",
                        message=preview,
                        sound=True,
                    )

                self.title = self.ICON_READY
                threading.Timer(
                    2.0, lambda: setattr(self, "title", self.ICON_IDLE)
                ).start()
            else:
                if self.config.get("show_notifications", True):
                    rumps.notification(
                        title="Transcription Empty",
                        subtitle="",
                        message="No speech detected in the recording",
                        sound=True,
                    )
                self.title = self.ICON_IDLE

        except (requests.RequestException, requests.Timeout) as e:
            logger.error("_process_audio: HTTP error - %s", e, exc_info=True)
            rumps.notification(
                title="Transcription Failed",
                subtitle="",
                message=str(e)[:100],
                sound=True,
            )
            self.title = self.ICON_ERROR
            threading.Timer(
                2.0, lambda: setattr(self, "title", self.ICON_IDLE)
            ).start()
        except Exception as e:
            logger.error("_process_audio: error - %s", e, exc_info=True)
            rumps.notification(
                title="Transcription Failed",
                subtitle="",
                message=str(e)[:100],
                sound=True,
            )
            self.title = self.ICON_ERROR
            threading.Timer(
                2.0, lambda: setattr(self, "title", self.ICON_IDLE)
            ).start()
        finally:
            total_time = time.time() - process_start
            logger.info(
                "_process_audio: complete; total processing %.2fs", total_time
            )
            self.processing = False
            self.record_button.title = "🎤 Start Recording"
            is_online = (
                getattr(self, "daemon_health", None) and self.daemon_health.online
            )
            self.status_item.title = (
                "Daemon: ● ready" if is_online else "Daemon: ○ offline"
            )

    # === Server Control Methods ===

    def start_server(self, _):
        """Bootstrap the launchd daemon."""
        rc, _stdout, stderr = _run_launchctl(
            ["bootstrap", f"gui/{os.getuid()}", DAEMON_PLIST]
        )
        # rc 37 / "already loaded" → treat as success
        already_loaded = (
            rc == 37
            or "already loaded" in (stderr or "").lower()
            or "service is already loaded" in (stderr or "").lower()
        )
        success = rc == 0 or already_loaded

        if success:
            self._offline_notified = False
            self.daemon_health._set(self.daemon_health.check_once())
            self._refresh_server_menu()
            msg = (
                "Daemon already loaded."
                if already_loaded and rc != 0
                else "Daemon bootstrapped."
            )
            if self.config.get("show_notifications", True):
                rumps.notification(
                    title="Server", subtitle="Start", message=msg, sound=False,
                )
        else:
            rumps.notification(
                title="Server",
                subtitle="Start failed",
                message=(stderr or f"rc={rc}")[:100],
                sound=True,
            )

    def stop_server(self, _):
        """Bootout the launchd daemon."""
        rc, _stdout, stderr = _run_launchctl(
            ["bootout", f"gui/{os.getuid()}", DAEMON_PLIST]
        )
        not_loaded = (
            "could not find service" in (stderr or "").lower()
            or "service not loaded" in (stderr or "").lower()
            or "no such process" in (stderr or "").lower()
        )
        success = rc == 0 or not_loaded

        if success:
            self.daemon_health._set(False)
            self._refresh_server_menu()
            if self.config.get("show_notifications", True):
                rumps.notification(
                    title="Server",
                    subtitle="Stop",
                    message=(
                        "Daemon already stopped."
                        if not_loaded and rc != 0
                        else "Daemon stopped."
                    ),
                    sound=False,
                )
        else:
            rumps.notification(
                title="Server",
                subtitle="Stop failed",
                message=(stderr or f"rc={rc}")[:100],
                sound=True,
            )

    def restart_server(self, _):
        """Kickstart the launchd daemon (restarts in place)."""
        rc, _stdout, stderr = _run_launchctl(
            ["kickstart", "-k", f"gui/{os.getuid()}/{DAEMON_LABEL}"]
        )
        if rc == 0:
            # Give the daemon a moment, then probe
            threading.Timer(
                1.5,
                lambda: self.daemon_health._set(self.daemon_health.check_once()),
            ).start()
            self._refresh_server_menu()
            if self.config.get("show_notifications", True):
                rumps.notification(
                    title="Server",
                    subtitle="Restart",
                    message="Daemon kickstarted.",
                    sound=False,
                )
        else:
            rumps.notification(
                title="Server",
                subtitle="Restart failed",
                message=(stderr or f"rc={rc}")[:100],
                sound=True,
            )

    def open_web_ui(self, _):
        """Open the daemon's web UI in the default browser."""
        webbrowser.open(f"{DAEMON_BASE_URL}/")

    def open_live_transcription(self, _):
        """Open the live transcription page in browser."""
        webbrowser.open(f"{DAEMON_BASE_URL}/live")

    def open_api_docs(self, _):
        """Show API documentation."""
        rumps.alert(
            title="API Documentation",
            message=(
                f"Base URL: {DAEMON_BASE_URL}\n\n"
                "Endpoints:\n"
                "• POST /api/transcribe - Transcribe audio file\n"
                "• POST /api/segment - Extract audio segment\n"
                "• GET /api/models - List available models\n\n"
                "Example:\n"
                f"curl -X POST -F 'file=@audio.mp3' {DAEMON_BASE_URL}/api/transcribe"
            ),
        )

    def view_daemon_logs(self, _):
        """Open the daemon's stderr log file."""
        try:
            if os.path.exists(DAEMON_STDERR_LOG):
                subprocess.run(["open", DAEMON_STDERR_LOG])
            else:
                rumps.alert(
                    title="Daemon log not found",
                    message=(
                        f"Expected log at:\n{DAEMON_STDERR_LOG}\n\n"
                        "Update the plist's StandardErrorPath, or start the daemon "
                        "at least once so the launchd-owned log appears."
                    ),
                )
        except Exception as e:
            rumps.alert(title="Could not open log", message=str(e))

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
                # Abort if daemon is offline
                if not (getattr(self, "daemon_health", None) and self.daemon_health.online):
                    rumps.notification(
                        title="Daemon Offline",
                        subtitle="",
                        message="Start the daemon from the Server submenu first.",
                        sound=True,
                    )
                    self.title = self.ICON_ERROR
                    return

                # Read file + derive duration
                file_name = os.path.basename(file_path)
                from pydub import AudioSegment
                audio = AudioSegment.from_file(file_path)
                duration = audio.duration_seconds
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                logger.info(f"Transcribing file: {file_name} ({duration:.1f}s)")

                # Build POST payload — preserve real filename/extension so the
                # daemon's audio processor decodes correctly (mp3, m4a, wav, …)
                ext = os.path.splitext(file_name)[1].lower() or ".wav"
                content_type = {
                    ".wav": "audio/wav",
                    ".mp3": "audio/mpeg",
                    ".m4a": "audio/mp4",
                    ".flac": "audio/flac",
                    ".ogg": "audio/ogg",
                    ".webm": "audio/webm",
                }.get(ext, "application/octet-stream")

                provider = self.config.get("stt_provider", "parakeet")
                model = (
                    self.config.get("model_name", "")
                    if provider == "parakeet"
                    else self.config.get("deepgram_model", "nova-3")
                )
                files = {"file": (file_name, file_bytes, content_type)}
                data = {
                    "provider": provider,
                    "model": model,
                    "deepgram_options": json.dumps(self.config.get("deepgram_options", {})),
                    "enable_diarization": str(self.config.get("diarization_enabled", False)).lower(),
                    "chunk_duration": str(self.config.get("default_chunk_duration", 120)),
                    "output_format": "json",
                }
                timeout = max(120, int(duration * 2))
                url = f"{self.client.base_url}/api/transcribe"
                logger.info(f"POST {url} file={file_name} provider={provider} timeout={timeout}s")

                response = requests.post(url, files=files, data=data, timeout=timeout)
                response.raise_for_status()
                payload = response.json()

                output_text = (payload.get("text") or "").strip()
                # Best-effort speaker count from daemon-side diarization
                num_speakers = 0
                segments = payload.get("segments") or []
                if isinstance(segments, list) and segments:
                    speakers = {
                        s.get("Speaker") or s.get("speaker")
                        for s in segments
                        if isinstance(s, dict)
                    }
                    speakers.discard(None)
                    num_speakers = len(speakers)

                if output_text:
                    if self.config.get("auto_copy_clipboard", True):
                        pyperclip.copy(output_text)
                    self._add_to_history(output_text, duration)
                    if self.config.get("show_notifications", True):
                        preview = (
                            output_text[:80] + "..." if len(output_text) > 80 else output_text
                        )
                        copied_msg = (
                            " - Copied!" if self.config.get("auto_copy_clipboard", True) else ""
                        )
                        speaker_info = (
                            f" ({num_speakers} speakers)" if num_speakers > 1 else ""
                        )
                        rumps.notification(
                            title=f"Transcription Complete{copied_msg}",
                            subtitle=f"{file_name}{speaker_info}",
                            message=preview,
                            sound=True,
                        )
                    self.title = self.ICON_READY
                    threading.Timer(2.0, lambda: setattr(self, "title", self.ICON_IDLE)).start()
                else:
                    rumps.notification(
                        title="Transcription Empty",
                        subtitle="",
                        message="No speech detected in the audio file",
                        sound=True,
                    )
                    self.title = self.ICON_IDLE

            except (requests.RequestException, requests.Timeout) as e:
                logger.error(f"File transcription HTTP error: {e}", exc_info=True)
                rumps.notification(
                    title="Transcription Failed",
                    subtitle="",
                    message=str(e)[:100],
                    sound=True,
                )
                self.title = self.ICON_ERROR
                threading.Timer(2.0, lambda: setattr(self, "title", self.ICON_IDLE)).start()
            except Exception as e:
                logger.error(f"File transcription error: {e}", exc_info=True)
                rumps.notification(
                    title="Transcription Error",
                    subtitle="",
                    message=str(e)[:100],
                    sound=True,
                )
                self.title = self.ICON_ERROR
                threading.Timer(2.0, lambda: setattr(self, "title", self.ICON_IDLE)).start()
            finally:
                self.processing = False
                is_online = (
                    getattr(self, "daemon_health", None) and self.daemon_health.online
                )
                self.status_item.title = (
                    "Daemon: ● ready" if is_online else "Daemon: ○ offline"
                )

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

        # Daemon status (launchctl-managed; menu bar just observes)
        server_status = (
            "Daemon: ● Running (port 8080)"
            if getattr(self, "daemon_health", None) and self.daemon_health.online
            else "Daemon: ○ Stopped"
        )

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

        # Stop daemon health probe (the launchd daemon itself stays running)
        try:
            if getattr(self, "daemon_health", None):
                self.daemon_health.stop()
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
