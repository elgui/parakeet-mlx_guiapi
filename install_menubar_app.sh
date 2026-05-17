#!/bin/bash
#
# Parakeet Menu Bar App Installer
#
# This script builds and installs the Parakeet voice-to-clipboard app
# to your Applications folder and optionally adds it to Login Items.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="Parakeet"
INSTALL_DIR="$HOME/.parakeet"
APP_PATH="$SCRIPT_DIR/dist/$APP_NAME.app"
INSTALL_PATH="/Applications/$APP_NAME.app"

echo "🦜 Parakeet Menu Bar App Installer"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "$SCRIPT_DIR/menubar_app.py" ]; then
    echo "❌ Error: menubar_app.py not found. Run this script from the project directory."
    exit 1
fi

# Detect or create dev .venv
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "📦 No .venv found in $SCRIPT_DIR — creating one..."
    python3 -m venv "$VENV_DIR" || { echo "❌ Failed to create .venv (is python3 installed?)"; exit 1; }
    echo "📦 Installing requirements (this takes a few minutes)..."
    "$VENV_PYTHON" -m pip install --upgrade pip --quiet
    "$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt" || { echo "❌ Failed to install requirements"; exit 1; }
fi

echo "📍 Using venv Python: $VENV_PYTHON"

# Check for required dependencies
echo "📦 Checking menu-bar deps..."
"$VENV_PYTHON" -c "import rumps, pyperclip, sounddevice, requests, scipy, numpy" 2>/dev/null || {
    echo "📦 Installing thin-client deps into venv..."
    "$VENV_PYTHON" -m pip install rumps pyperclip sounddevice requests scipy numpy
}

# Install py2app if needed
"$VENV_PYTHON" -m pip install -q py2app 2>/dev/null || true

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/dist"

# Build the app using alias mode (fast, lightweight)
echo "🔨 Building Parakeet.app..."
cd "$SCRIPT_DIR"
"$VENV_PYTHON" setup_app.py py2app --alias 2>&1 | grep -E "(error|Error|Done)" || true

if [ ! -d "$APP_PATH" ]; then
    echo "❌ Error: Build failed."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Create installation directory for source files
echo "📁 Setting up installation..."
mkdir -p "$INSTALL_DIR"

# Copy source files to install directory
cp -R "$SCRIPT_DIR/menubar_app.py" "$INSTALL_DIR/"

VENV_SITE_PACKAGES="$(dirname "$VENV_PYTHON")/../lib/python3.*/site-packages"
VENV_SITE_PACKAGES=$(echo $VENV_SITE_PACKAGES)  # Expand glob

# Create a wrapper script that the app will use
cat > "$INSTALL_DIR/run_parakeet.py" << PYTHON_SCRIPT
#!${VENV_PYTHON}
import os
import sys

# Add install dir and venv site-packages to path
install_dir = os.path.dirname(os.path.abspath(__file__))
venv_site = "${VENV_SITE_PACKAGES}"

# Insert venv site-packages FIRST to override system packages
if os.path.exists(venv_site):
    sys.path.insert(0, venv_site)
sys.path.insert(0, install_dir)

# Import and run
from menubar_app import main
main()
PYTHON_SCRIPT

chmod +x "$INSTALL_DIR/run_parakeet.py"

# Build the app pointing to installed location
echo "🔨 Building final app..."
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/dist"

# Create a temporary setup for the installed location
cat > "$INSTALL_DIR/setup_app.py" << 'SETUP_SCRIPT'
import sys
from setuptools import setup
sys.setrecursionlimit(5000)

OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'Parakeet',
        'CFBundleDisplayName': 'Parakeet Voice-to-Clipboard',
        'CFBundleIdentifier': 'com.parakeet.menubar',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'LSUIElement': True,
        'NSMicrophoneUsageDescription': 'Parakeet needs microphone access to record and transcribe your voice.',
    },
}

setup(
    app=['run_parakeet.py'],
    name='Parakeet',
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
SETUP_SCRIPT

cd "$INSTALL_DIR"
"$VENV_PYTHON" setup_app.py py2app --alias 2>&1 | grep -E "(error|Error|Done)" || true

if [ ! -d "$INSTALL_DIR/dist/$APP_NAME.app" ]; then
    echo "❌ Error: Final build failed."
    exit 1
fi

# Install to Applications
echo "📁 Installing to /Applications..."
if [ -d "$INSTALL_PATH" ]; then
    echo "   Removing existing installation..."
    rm -rf "$INSTALL_PATH"
fi
cp -R "$INSTALL_DIR/dist/$APP_NAME.app" "$INSTALL_PATH"
echo "✅ Installed to $INSTALL_PATH"
echo ""

# Clean up build artifacts in install dir
rm -rf "$INSTALL_DIR/build" "$INSTALL_DIR/dist" "$INSTALL_DIR/setup_app.py"

# Ask about Login Items
echo "🚀 Would you like Parakeet to start automatically at login?"
echo "   (You can change this later in System Settings → General → Login Items)"
read -p "   Add to Login Items? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$INSTALL_PATH\", hidden:false}" 2>/dev/null && \
    echo "✅ Added to Login Items" || \
    echo "⚠️  Could not add to Login Items automatically. Please add manually in System Settings."
fi

echo ""
echo "=================================="
echo "🎉 Installation complete!"
echo ""
echo "Source files installed to: $INSTALL_DIR"
echo "App installed to: $INSTALL_PATH"
echo ""
echo "To start Parakeet now:"
echo "   open /Applications/Parakeet.app"
echo ""
echo "Or find it in Spotlight (Cmd+Space) and type 'Parakeet'"
echo ""
echo "Usage:"
echo "   🎤 Click the mic icon in the menu bar to start recording"
echo "   🔴 Click again to stop and copy transcription to clipboard"
echo "=================================="

# Offer to launch now
read -p "Launch Parakeet now? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    # Kill any existing instance first
    pkill -f "Parakeet.app" 2>/dev/null || true
    sleep 1
    open "$INSTALL_PATH"
    echo "🦜 Parakeet is now running in your menu bar!"
fi
