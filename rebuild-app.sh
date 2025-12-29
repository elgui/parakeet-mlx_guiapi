#!/bin/bash
# Rebuild and install the Parakeet menu bar app
# Usage: ./rebuild-app.sh [--watch]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="Parakeet.app"
INSTALL_DIR="/Applications"

cd "$SCRIPT_DIR"

rebuild() {
    echo "🔨 Building $APP_NAME..."

    # Clean previous build
    rm -rf build dist

    # Build with py2app in alias mode (references source files)
    python3 setup_app.py py2app --alias

    # Remove old app if exists
    if [ -d "$INSTALL_DIR/$APP_NAME" ]; then
        echo "🗑️  Removing old $APP_NAME from $INSTALL_DIR..."
        rm -rf "$INSTALL_DIR/$APP_NAME"
    fi

    # Install new app (use ditto for proper macOS bundle handling)
    echo "📦 Installing to $INSTALL_DIR..."
    ditto "dist/$APP_NAME" "$INSTALL_DIR/$APP_NAME"

    echo "✅ $APP_NAME rebuilt and installed!"
    echo ""
    echo "Note: With --alias mode, changes to Python files (except menubar_app.py)"
    echo "take effect immediately without rebuilding."
}

watch_mode() {
    echo "👀 Watching for changes to menubar_app.py..."
    echo "   Press Ctrl+C to stop"
    echo ""

    # Use fswatch if available, otherwise fallback to polling
    if command -v fswatch &> /dev/null; then
        fswatch -o "$SCRIPT_DIR/menubar_app.py" | while read; do
            echo ""
            echo "🔄 Change detected in menubar_app.py"
            rebuild
        done
    else
        echo "⚠️  fswatch not found. Install with: brew install fswatch"
        echo "   Falling back to polling mode (5 second intervals)..."

        LAST_HASH=""
        while true; do
            CURRENT_HASH=$(md5 -q "$SCRIPT_DIR/menubar_app.py" 2>/dev/null || echo "")
            if [ -n "$CURRENT_HASH" ] && [ "$CURRENT_HASH" != "$LAST_HASH" ] && [ -n "$LAST_HASH" ]; then
                echo ""
                echo "🔄 Change detected in menubar_app.py"
                rebuild
            fi
            LAST_HASH="$CURRENT_HASH"
            sleep 5
        done
    fi
}

# Main
case "${1:-}" in
    --watch|-w)
        rebuild
        watch_mode
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --watch, -w    Rebuild on changes to menubar_app.py"
        echo "  --help, -h     Show this help message"
        echo ""
        echo "Without options, performs a one-time rebuild and install."
        ;;
    *)
        rebuild
        ;;
esac
