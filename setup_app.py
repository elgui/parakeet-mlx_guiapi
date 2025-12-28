"""
py2app setup script for Parakeet Menu Bar App.

Usage:
    python setup_app.py py2app --alias

This creates a macOS app that references the source files directly.
The app requires Python and dependencies to be installed on the system.
"""

import sys
from setuptools import setup

# Increase recursion limit for py2app
sys.setrecursionlimit(5000)

APP = ['menubar_app.py']
DATA_FILES = []

OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,  # Add path to .icns file if you have one
    'plist': {
        'CFBundleName': 'Parakeet',
        'CFBundleDisplayName': 'Parakeet Voice-to-Clipboard',
        'CFBundleIdentifier': 'com.parakeet.menubar',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'LSUIElement': True,  # Hide from Dock (menu bar app)
        'NSMicrophoneUsageDescription': 'Parakeet needs microphone access to record and transcribe your voice.',
    },
}

setup(
    app=APP,
    name='Parakeet',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
