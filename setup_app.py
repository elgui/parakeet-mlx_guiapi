"""
py2app setup script for Parakeet Menu Bar App.

Usage:
    python setup_app.py py2app

This creates a standalone macOS app in the dist/ folder.
"""

from setuptools import setup

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
    'packages': [
        'parakeet_mlx_guiapi',
        'rumps',
        'pyperclip',
        'sounddevice',
        'numpy',
        'scipy',
        'pandas',
        'mlx',
    ],
    'includes': [
        'parakeet_mlx',
    ],
}

setup(
    app=APP,
    name='Parakeet',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
