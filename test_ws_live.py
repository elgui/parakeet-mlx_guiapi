#!/usr/bin/env python3
"""Test script for WebSocket live transcription."""

import json
import base64
import time
import threading
import sys
from websocket import create_connection, WebSocketTimeoutException


def test_websocket(audio_file=None):
    """Test the WebSocket connection and transcription."""
    print("=" * 60)
    print("WebSocket Live Transcription Test")
    print("=" * 60)

    ws_url = "ws://127.0.0.1:8080/ws/live-transcribe"
    print(f"\n1. Connecting to {ws_url}...")

    try:
        ws = create_connection(ws_url, timeout=10)
        print("   Connected!")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    # Set up receiver thread
    received_messages = []
    running = True

    def receive_messages():
        while running:
            try:
                ws.settimeout(0.5)
                msg = ws.recv()
                if msg:
                    data = json.loads(msg)
                    received_messages.append(data)
                    msg_preview = data.get('message', data.get('session_id', ''))
                    if isinstance(msg_preview, str):
                        msg_preview = msg_preview[:60]
                    print(f"   << {data.get('type')}: {msg_preview}")

                    # Print transcription text
                    if data.get('type') == 'transcription':
                        for m in data.get('messages', []):
                            print(f"      [{m.get('speaker')}] {m.get('text')}")
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

    print("\n2. Checking connection response...")
    connected = any(m.get('type') == 'connected' for m in received_messages)
    if connected:
        conn_msg = next(m for m in received_messages if m.get('type') == 'connected')
        print(f"   Session ID: {conn_msg.get('session_id')}")
        print(f"   Model: {conn_msg.get('model_name')}")
        print(f"   Diarization: {conn_msg.get('diarization_available')}")
    else:
        print("   FAILED: No connection response")
        return

    # Load audio file
    if audio_file:
        print(f"\n3. Loading audio from {audio_file}...")
        with open(audio_file, 'rb') as f:
            wav_data = f.read()
    else:
        print("\n3. Generating test audio (beep)...")
        wav_data = generate_test_wav(duration_sec=2.0)

    wav_base64 = base64.b64encode(wav_data).decode('utf-8')
    print(f"   Audio size: {len(wav_data)} bytes ({len(wav_data)/1024:.1f} KB)")

    print("\n4. Sending audio chunk #1...")
    ws.send(json.dumps({
        'type': 'audio_chunk',
        'data': wav_base64,
        'chunk_start': 0.0
    }))
    print("   Sent! Waiting for transcription (up to 30s)...")

    # Wait for transcription
    start = time.time()
    while time.time() - start < 30:
        transcriptions = [m for m in received_messages if m.get('type') == 'transcription' and m.get('messages')]
        if transcriptions:
            break
        time.sleep(0.5)

    print("\n5. Results...")
    transcriptions = [m for m in received_messages if m.get('type') == 'transcription']
    print(f"   Transcriptions received: {len(transcriptions)}")

    for i, t in enumerate(transcriptions):
        msgs = t.get('messages', [])
        if msgs:
            print(f"   Transcription #{i+1}:")
            for msg in msgs:
                print(f"      [{msg.get('speaker')}] {msg.get('text')}")
        else:
            print(f"   Transcription #{i+1}: (empty)")

    # Check for errors
    errors = [m for m in received_messages if m.get('type') == 'error']
    if errors:
        print(f"\n   ERRORS: {[e.get('message') for e in errors]}")

    print(f"\n   All message types: {[m.get('type') for m in received_messages]}")

    running = False
    ws.close()
    print("\n" + "=" * 60)
    print("Test complete!")


def generate_test_wav(duration_sec=2.0, sample_rate=16000):
    """Generate a test WAV file with a simple tone."""
    import wave
    import struct
    import io

    num_samples = int(sample_rate * duration_sec)
    frequency = 440
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.5 * (1 if (int(t * frequency * 2) % 2 == 0) else -1))
        samples.append(value)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    return wav_buffer.getvalue()


if __name__ == '__main__':
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_websocket(audio_file)
