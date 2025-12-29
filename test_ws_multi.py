#!/usr/bin/env python3
"""Test script for WebSocket with multiple rapid chunks."""

import json
import base64
import time
import threading
from websocket import create_connection, WebSocketTimeoutException


def test_multiple_chunks():
    """Test sending multiple chunks rapidly."""
    print("=" * 60)
    print("Multi-Chunk WebSocket Test")
    print("=" * 60)

    ws_url = "ws://127.0.0.1:8080/ws/live-transcribe"
    print(f"\n1. Connecting to {ws_url}...")

    ws = create_connection(ws_url, timeout=10)
    print("   Connected!")

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
                    ts = time.strftime('%H:%M:%S')
                    if data.get('type') == 'transcription':
                        for m in data.get('messages', []):
                            print(f"   [{ts}] TRANSCRIPTION: [{m.get('speaker')}] {m.get('text')}")
                    elif data.get('type') == 'status':
                        print(f"   [{ts}] STATUS: {data.get('message')}")
                    elif data.get('type') == 'debug':
                        print(f"   [{ts}] DEBUG: {data.get('message')}")
                    elif data.get('type') == 'error':
                        print(f"   [{ts}] ERROR: {data.get('message')}")
            except WebSocketTimeoutException:
                continue
            except Exception as e:
                if running:
                    print(f"   Receive error: {e}")
                break

    receiver = threading.Thread(target=receive_messages, daemon=True)
    receiver.start()

    time.sleep(1)  # Wait for connection

    # Generate test phrases
    phrases = [
        "Hello, this is the first test message.",
        "This is the second message being sent.",
        "And here comes the third message.",
    ]

    print(f"\n2. Generating {len(phrases)} audio chunks...")
    audio_chunks = []
    for phrase in phrases:
        import subprocess
        subprocess.run(['say', phrase, '-o', '/tmp/test_chunk.aiff'], check=True, capture_output=True)
        subprocess.run(['ffmpeg', '-y', '-i', '/tmp/test_chunk.aiff', '-ar', '16000', '-ac', '1', '/tmp/test_chunk.wav'],
                      check=True, capture_output=True)
        with open('/tmp/test_chunk.wav', 'rb') as f:
            audio_chunks.append(f.read())
        print(f"   Generated: \"{phrase}\" ({len(audio_chunks[-1])/1024:.1f} KB)")

    print(f"\n3. Sending {len(audio_chunks)} chunks RAPIDLY (simulating speech bursts)...")
    chunk_start = 0.0
    for i, audio_data in enumerate(audio_chunks):
        wav_base64 = base64.b64encode(audio_data).decode('utf-8')
        print(f"   >> Sending chunk #{i+1} at {chunk_start:.1f}s...")
        ws.send(json.dumps({
            'type': 'audio_chunk',
            'data': wav_base64,
            'chunk_start': chunk_start
        }))
        chunk_start += 3.0  # Simulate 3 second chunks
        time.sleep(0.1)  # Very short delay - simulating rapid speech

    print(f"\n4. Waiting for all transcriptions (60s max)...")
    start = time.time()
    while time.time() - start < 60:
        transcriptions = [m for m in received_messages if m.get('type') == 'transcription' and m.get('messages')]
        if len(transcriptions) >= len(audio_chunks):
            print(f"   All {len(audio_chunks)} transcriptions received!")
            break
        time.sleep(1)

    print(f"\n5. RESULTS:")
    print(f"   Total messages received: {len(received_messages)}")
    transcriptions = [m for m in received_messages if m.get('type') == 'transcription']
    print(f"   Transcriptions: {len(transcriptions)}")

    for i, t in enumerate(transcriptions):
        msgs = t.get('messages', [])
        if msgs:
            for msg in msgs:
                print(f"      #{i+1}: [{msg.get('speaker')}] {msg.get('text')}")

    errors = [m for m in received_messages if m.get('type') == 'error']
    if errors:
        print(f"   ERRORS: {[e.get('message') for e in errors]}")

    running = False
    ws.close()
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_multiple_chunks()
