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

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -q rumps py2app pyobjc-framework-Cocoa

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/dist"

# Build the app
echo "🔨 Building Parakeet.app..."
cd "$SCRIPT_DIR"
python3 setup_app.py py2app --quiet 2>/dev/null || python3 setup_app.py py2app

if [ ! -d "$APP_PATH" ]; then
    echo "❌ Error: Build failed. Check the output above for errors."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Install to Applications
echo "📁 Installing to /Applications..."
if [ -d "$INSTALL_PATH" ]; then
    echo "   Removing existing installation..."
    rm -rf "$INSTALL_PATH"
fi
cp -R "$APP_PATH" "$INSTALL_PATH"
echo "✅ Installed to $INSTALL_PATH"
echo ""

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
    open "$INSTALL_PATH"
    echo "🦜 Parakeet is now running in your menu bar!"
fi
