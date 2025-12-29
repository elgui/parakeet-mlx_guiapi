#!/usr/bin/env python3
"""
Automated test for live transcription WebSocket.
Uses macOS 'say' command to generate test audio.
"""

import json
import base64
import time
import threading
import subprocess
import tempfile
import os
from websocket import create_connection, WebSocketTimeoutException


def generate_test_audio(text: str, output_path: str) -> bool:
    """Generate WAV audio from text using macOS say command."""
    try:
        # Generate AIFF first (say's native format)
        aiff_path = output_path.replace('.wav', '.aiff')
        subprocess.run(
            ['say', '-v', 'Samantha', text, '-o', aiff_path],
            check=True, capture_output=True
        )

        # Convert to 16kHz mono WAV using ffmpeg
        subprocess.run(
            ['ffmpeg', '-y', '-i', aiff_path, '-ar', '16000', '-ac', '1', output_path],
            check=True, capture_output=True
        )

        # Cleanup
        os.unlink(aiff_path)
        return True
    except Exception as e:
        print(f"Error generating audio: {e}")
        return False


def test_live_transcription():
    """Test the live transcription WebSocket endpoint."""
    print("=" * 70)
    print("AUTOMATED LIVE TRANSCRIPTION TEST")
    print("=" * 70)

    ws_url = "ws://127.0.0.1:8080/ws/live-transcribe"
    print(f"\n1. Connecting to {ws_url}...")

    try:
        ws = create_connection(ws_url, timeout=10)
        print("   ✓ Connected!")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        print("\n   Make sure the server is running: python run.py")
        return False

    # Set up message receiver
    received_messages = []
    transcriptions = []
    running = True

    def receive_messages():
        while running:
            try:
                ws.settimeout(0.5)
                msg = ws.recv()
                if msg:
                    data = json.loads(msg)
                    received_messages.append(data)

                    msg_type = data.get('type', 'unknown')
                    if msg_type == 'connected':
                        print(f"   Session: {data.get('session_id')}")
                        print(f"   Model: {data.get('model_name')}")
                        print(f"   Diarization: {data.get('diarization_available')}")
                    elif msg_type == 'transcription':
                        msgs = data.get('messages', [])
                        for m in msgs:
                            text = m.get('text', '').strip()
                            speaker = m.get('speaker', 'Unknown')
                            print(f"   📝 [{speaker}] {text}")
                            transcriptions.append(text)
                    elif msg_type == 'status':
                        print(f"   📊 {data.get('message')}")
                    elif msg_type == 'debug':
                        print(f"   🔧 {data.get('message')}")
                    elif msg_type == 'error':
                        print(f"   ❌ ERROR: {data.get('message')}")
            except WebSocketTimeoutException:
                continue
            except Exception as e:
                if running:
                    print(f"   Receive error: {e}")
                break

    receiver = threading.Thread(target=receive_messages, daemon=True)
    receiver.start()

    # Wait for connection message
    time.sleep(1)

    print("\n2. Checking connection...")
    connected = any(m.get('type') == 'connected' for m in received_messages)
    if not connected:
        print("   ✗ No connection response received!")
        running = False
        ws.close()
        return False
    print("   ✓ Connection confirmed")

    # Generate test audio
    test_phrases = [
        "Hello, this is a test of the live transcription system.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("\n3. Generating test audio...")
    audio_chunks = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, phrase in enumerate(test_phrases):
            wav_path = os.path.join(tmpdir, f"test_{i}.wav")
            print(f"   Generating: \"{phrase[:50]}...\"")

            if not generate_test_audio(phrase, wav_path):
                print(f"   ✗ Failed to generate audio for phrase {i}")
                continue

            with open(wav_path, 'rb') as f:
                audio_data = f.read()

            audio_chunks.append({
                'phrase': phrase,
                'data': audio_data,
                'size_kb': len(audio_data) / 1024
            })
            print(f"   ✓ Generated {len(audio_data)/1024:.1f} KB")

    if not audio_chunks:
        print("   ✗ No audio chunks generated!")
        running = False
        ws.close()
        return False

    print(f"\n4. Sending {len(audio_chunks)} audio chunks...")
    chunk_start = 0.0

    for i, chunk in enumerate(audio_chunks):
        audio_base64 = base64.b64encode(chunk['data']).decode('utf-8')

        print(f"   >> Sending chunk #{i+1}: {chunk['size_kb']:.1f} KB")
        ws.send(json.dumps({
            'type': 'audio_chunk',
            'data': audio_base64,
            'chunk_start': chunk_start
        }))

        # Estimate duration (16kHz, 16-bit mono = 32000 bytes/sec)
        duration = len(chunk['data']) / 32000
        chunk_start += duration

        # Small delay between chunks
        time.sleep(0.5)

    print(f"\n5. Waiting for transcriptions (max 60s)...")
    start_time = time.time()

    while time.time() - start_time < 60:
        if len(transcriptions) >= len(audio_chunks):
            print(f"   ✓ Received {len(transcriptions)} transcriptions!")
            break
        time.sleep(1)
    else:
        print(f"   ⚠ Timeout - only received {len(transcriptions)}/{len(audio_chunks)} transcriptions")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTotal messages received: {len(received_messages)}")
    print(f"Transcriptions received: {len(transcriptions)}")

    if transcriptions:
        print("\nTranscriptions:")
        for i, text in enumerate(transcriptions, 1):
            print(f"  {i}. {text}")

    errors = [m for m in received_messages if m.get('type') == 'error']
    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  - {e.get('message')}")

    # Verify transcriptions match input
    print("\n" + "-" * 70)
    success = len(transcriptions) >= len(audio_chunks)

    if success:
        print("✅ TEST PASSED: All audio chunks were transcribed!")
    else:
        print("❌ TEST FAILED: Some audio chunks were not transcribed.")

    running = False
    ws.close()

    return success


if __name__ == '__main__':
    import sys
    success = test_live_transcription()
    sys.exit(0 if success else 1)
