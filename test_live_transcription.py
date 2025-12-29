#!/usr/bin/env python3
"""
Test script for live transcription WebSocket endpoint.

Simulates a mobile client by:
1. Connecting to WebSocket
2. Sending audio chunks (simulating VAD-detected speech segments)
3. Displaying transcription results with speaker colors
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

# Use websockets library for async WebSocket client
try:
    import websockets
except ImportError:
    print("Installing websockets library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

from pydub import AudioSegment


# ANSI color codes for terminal output
COLORS = {
    "#E3F2FD": "\033[94m",  # Blue
    "#FFF3E0": "\033[93m",  # Yellow/Orange
    "#E8F5E9": "\033[92m",  # Green
    "#FCE4EC": "\033[95m",  # Pink/Magenta
    "#F3E5F5": "\033[35m",  # Purple
    "#FFFDE7": "\033[33m",  # Yellow
    "#E0F7FA": "\033[96m",  # Cyan
    "#FBE9E7": "\033[91m",  # Red/Orange
}
RESET = "\033[0m"
BOLD = "\033[1m"


def get_terminal_color(hex_color):
    """Convert hex color to terminal ANSI code."""
    return COLORS.get(hex_color, "")


def split_audio_into_chunks(audio_path, chunk_duration_ms=10000):
    """
    Split audio file into chunks (simulating VAD speech segments).

    Args:
        audio_path: Path to the audio file
        chunk_duration_ms: Duration of each chunk in milliseconds

    Yields:
        (chunk_bytes, chunk_start_time) tuples
    """
    audio = AudioSegment.from_file(audio_path)

    # Ensure 16kHz mono
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    if audio.channels > 1:
        audio = audio.set_channels(1)

    total_duration = len(audio)
    print(f"Audio duration: {total_duration/1000:.1f} seconds")
    print(f"Splitting into {chunk_duration_ms/1000:.0f}-second chunks...")
    print()

    for start_ms in range(0, total_duration, chunk_duration_ms):
        end_ms = min(start_ms + chunk_duration_ms, total_duration)
        chunk = audio[start_ms:end_ms]

        # Export chunk to WAV bytes
        import io
        buffer = io.BytesIO()
        chunk.export(buffer, format="wav")
        chunk_bytes = buffer.getvalue()

        yield chunk_bytes, start_ms / 1000.0


async def test_live_transcription(audio_path, server_url="ws://localhost:8080/ws/live-transcribe"):
    """
    Test the live transcription WebSocket endpoint.

    Args:
        audio_path: Path to test audio file
        server_url: WebSocket server URL
    """
    print(f"Connecting to {server_url}...")

    async with websockets.connect(server_url) as ws:
        # Wait for connection confirmation
        response = await ws.recv()
        data = json.loads(response)

        if data["type"] == "connected":
            session_id = data["session_id"]
            diarization = data.get("diarization_available", False)
            print(f"Connected! Session ID: {session_id}")
            print(f"Speaker diarization: {'enabled' if diarization else 'disabled'}")
            print()

        # Send config
        await ws.send(json.dumps({
            "type": "config",
            "enable_diarization": True
        }))

        # Wait for config ack
        response = await ws.recv()

        # Process audio chunks
        all_messages = []

        for chunk_bytes, chunk_start in split_audio_into_chunks(audio_path):
            # Encode chunk as base64
            chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")

            print(f"Sending chunk at {chunk_start:.1f}s ({len(chunk_bytes)/1024:.1f} KB)...")

            # Send audio chunk
            await ws.send(json.dumps({
                "type": "audio_chunk",
                "data": chunk_b64,
                "chunk_start": chunk_start
            }))

            # Wait for responses (status + transcription)
            while True:
                response = await ws.recv()
                data = json.loads(response)

                if data["type"] == "status":
                    print(f"  Status: {data['message']}")

                elif data["type"] == "transcription":
                    messages = data.get("messages", [])
                    all_messages.extend(messages)

                    for msg in messages:
                        color = get_terminal_color(msg["color"])
                        speaker = msg["speaker"]
                        text = msg["text"]
                        start = msg["start_time"]
                        end = msg["end_time"]

                        print(f"  {color}{BOLD}[{speaker}]{RESET} {color}({start:.1f}s-{end:.1f}s){RESET}")
                        print(f"  {color}{text}{RESET}")
                        print()

                    break  # Got transcription, move to next chunk

                elif data["type"] == "error":
                    print(f"  ERROR: {data['message']}")
                    break

        # Request export
        print("\n" + "="*60)
        print("Requesting TXT export...")

        await ws.send(json.dumps({
            "type": "export",
            "format": "txt"
        }))

        response = await ws.recv()
        data = json.loads(response)

        if data["type"] == "export_result":
            print(f"\nExported to: {data['filename']}")
            print("-"*40)
            print(data["content"])
            print("-"*40)

        print(f"\nTotal messages transcribed: {len(all_messages)}")

        # Count unique speakers
        speakers = set(msg["speaker"] for msg in all_messages)
        print(f"Unique speakers detected: {len(speakers)} ({', '.join(sorted(speakers))})")


def main():
    if len(sys.argv) < 2:
        # Default to test_segment.wav
        audio_path = Path(__file__).parent / "test_segment.wav"
        if not audio_path.exists():
            print("Usage: python test_live_transcription.py <audio_file>")
            print("Or place a test_segment.wav file in the current directory.")
            sys.exit(1)
    else:
        audio_path = Path(sys.argv[1])

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print("="*60)
    print("LIVE TRANSCRIPTION TEST")
    print("="*60)
    print(f"Audio file: {audio_path}")
    print()

    asyncio.run(test_live_transcription(str(audio_path)))


if __name__ == "__main__":
    main()
