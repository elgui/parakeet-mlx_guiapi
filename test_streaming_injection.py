#!/usr/bin/env python3
"""
Test script for streaming audio injection via WebSocket.

Simulates the browser's streaming injection feature to test
the live transcription pipeline end-to-end.
"""

import asyncio
import json
import base64
import wave
import io
import sys
from pathlib import Path

# Use websockets library for async WebSocket
try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets

try:
    from pydub import AudioSegment
except ImportError:
    print("Installing pydub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "-q"])
    from pydub import AudioSegment


# Configuration
WS_URL = "ws://127.0.0.1:8080/ws/live-transcribe"
AUDIO_FILE = Path(__file__).parent / "static" / "test" / "2ppl-FR.mp3"
CHUNK_DURATION_SEC = 8  # Match browser's STREAM_CONFIG
DELAY_BETWEEN_CHUNKS = 0.5  # 500ms delay
TARGET_SAMPLE_RATE = 16000


def load_and_prepare_audio(audio_path: Path) -> tuple[bytes, float]:
    """Load audio file and convert to 16kHz mono WAV."""
    print(f"📁 Loading: {audio_path}")

    # Load with pydub
    audio = AudioSegment.from_file(str(audio_path))
    original_duration = len(audio) / 1000.0
    print(f"   Original: {original_duration:.2f}s, {audio.frame_rate}Hz, {audio.channels}ch")

    # Convert to mono 16kHz
    audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
    print(f"   Converted: {len(audio)/1000:.2f}s, {audio.frame_rate}Hz, {audio.channels}ch")

    # Get raw samples
    samples = audio.raw_data
    return samples, original_duration


def create_wav_chunk(samples: bytes, sample_rate: int = 16000) -> bytes:
    """Create a WAV file from raw PCM samples."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(samples)
    return buf.getvalue()


def split_into_chunks(samples: bytes, chunk_duration_sec: float, sample_rate: int = 16000) -> list[bytes]:
    """Split audio samples into chunks of specified duration."""
    bytes_per_sample = 2  # 16-bit audio
    samples_per_chunk = int(chunk_duration_sec * sample_rate)
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

    chunks = []
    for i in range(0, len(samples), bytes_per_chunk):
        chunk = samples[i:i + bytes_per_chunk]
        if len(chunk) > sample_rate * bytes_per_sample * 0.5:  # Skip chunks < 0.5s
            chunks.append(chunk)

    return chunks


async def test_streaming_injection():
    """Main test function."""
    print("=" * 60)
    print("🧪 LIVE TRANSCRIPTION STREAMING TEST")
    print("=" * 60)

    # Load audio
    if not AUDIO_FILE.exists():
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        return

    samples, duration = load_and_prepare_audio(AUDIO_FILE)
    chunks = split_into_chunks(samples, CHUNK_DURATION_SEC)

    print(f"\n📡 Streaming config:")
    print(f"   Total duration: {duration:.2f}s")
    print(f"   Chunk size: {CHUNK_DURATION_SEC}s")
    print(f"   Number of chunks: {len(chunks)}")
    print(f"   Delay between chunks: {DELAY_BETWEEN_CHUNKS}s")

    # Connect to WebSocket
    print(f"\n🔌 Connecting to {WS_URL}...")

    try:
        async with websockets.connect(WS_URL) as ws:
            print("✅ Connected!")

            # Receive connection message
            msg = await ws.recv()
            data = json.loads(msg)
            if data.get('type') == 'connected':
                print(f"   Session ID: {data.get('session_id')}")
                print(f"   Provider: {data.get('provider_name')}")
                print(f"   Model: {data.get('model_name')}")
                print(f"   Diarization: {'✅' if data.get('diarization_available') else '❌'}")

            # Send config
            config = {
                'type': 'config',
                'enable_diarization': True,
                'similarity_threshold': 0.45
            }
            await ws.send(json.dumps(config))
            print("\n📤 Config sent: diarization=True, threshold=0.45")

            # Start streaming chunks
            print("\n" + "=" * 60)
            print("📡 STREAMING AUDIO CHUNKS")
            print("=" * 60)

            chunk_start = 0.0
            results = []

            for i, chunk_samples in enumerate(chunks):
                chunk_wav = create_wav_chunk(chunk_samples)
                chunk_b64 = base64.b64encode(chunk_wav).decode('ascii')
                chunk_duration = len(chunk_samples) / (TARGET_SAMPLE_RATE * 2)

                print(f"\n📤 Chunk {i+1}/{len(chunks)}:")
                print(f"   Duration: {chunk_duration:.2f}s")
                print(f"   Size: {len(chunk_wav)/1024:.1f} KB")
                print(f"   Start time: {chunk_start:.2f}s")

                # Send chunk
                await ws.send(json.dumps({
                    'type': 'audio_chunk',
                    'data': chunk_b64,
                    'chunk_start': chunk_start
                }))

                chunk_start += chunk_duration

                # Wait for responses (with timeout)
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        data = json.loads(msg)

                        if data.get('type') == 'status':
                            print(f"   ⏳ {data.get('message')}")
                        elif data.get('type') == 'debug':
                            print(f"   🔧 {data.get('message')}")
                        elif data.get('type') == 'transcription':
                            messages = data.get('messages', [])
                            print(f"   ✅ Transcription received: {len(messages)} segment(s)")
                            for msg in messages:
                                speaker = msg.get('speaker', '?')
                                text = msg.get('text', '')
                                start = msg.get('start_time', 0)
                                end = msg.get('end_time', 0)
                                print(f"      [{speaker}] ({start:.1f}-{end:.1f}s): {text[:80]}{'...' if len(text) > 80 else ''}")
                                results.append(msg)
                            break  # Got transcription, move to next chunk
                        elif data.get('type') == 'error':
                            print(f"   ❌ Error: {data.get('message')}")
                            break

                except asyncio.TimeoutError:
                    print(f"   ⚠️ Timeout waiting for response")

                # Delay before next chunk (except for last)
                if i < len(chunks) - 1:
                    print(f"   ⏱️ Waiting {DELAY_BETWEEN_CHUNKS}s before next chunk...")
                    await asyncio.sleep(DELAY_BETWEEN_CHUNKS)

            # Summary
            print("\n" + "=" * 60)
            print("📊 TEST RESULTS SUMMARY")
            print("=" * 60)
            print(f"   Total chunks sent: {len(chunks)}")
            print(f"   Total segments received: {len(results)}")

            # Count unique speakers
            speakers = set(msg.get('speaker_id') for msg in results)
            print(f"   Unique speakers detected: {len(speakers)}")

            # Print full transcript
            print("\n📝 FULL TRANSCRIPT:")
            print("-" * 40)
            for msg in results:
                speaker = msg.get('speaker', '?')
                text = msg.get('text', '')
                print(f"[{speaker}]: {text}")
            print("-" * 40)

            # Test TXT export
            print("\n📥 Testing TXT export...")
            await ws.send(json.dumps({'type': 'export', 'format': 'txt'}))

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                if data.get('type') == 'export_result':
                    print(f"   ✅ TXT Export successful: {data.get('filename')}")
                    print(f"   Content:\n{data.get('content', '')}")
            except asyncio.TimeoutError:
                print("   ⚠️ TXT Export timeout")

            # Test SRT export
            print("\n📥 Testing SRT export...")
            await ws.send(json.dumps({'type': 'export', 'format': 'srt'}))

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                if data.get('type') == 'export_result':
                    print(f"   ✅ SRT Export successful: {data.get('filename')}")
                    print(f"   Content:\n{data.get('content', '')}")
            except asyncio.TimeoutError:
                print("   ⚠️ SRT Export timeout")

            print("\n✅ TEST COMPLETE")

    except ConnectionRefusedError:
        print("❌ Connection refused - is the server running?")
        print("   Start with: python run.py --port 8080")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_streaming_injection())
