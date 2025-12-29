"""
WebSocket handler for live transcription.

This module provides WebSocket endpoints for real-time audio streaming
and transcription with queued processing to prevent dropped chunks.
"""

import json
import traceback
import logging
import sys
import threading
import queue
from typing import Dict, Optional
from flask import Flask, render_template
from flask_sock import Sock

from .session import LiveTranscriptionSession

# Set up logging to stdout for immediate visibility
logging.basicConfig(
    level=logging.DEBUG,
    format='[WS %(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger('live_ws')
logger.setLevel(logging.DEBUG)

# Store active sessions
_sessions: Dict[str, LiveTranscriptionSession] = {}


def get_current_model_name():
    """Get the name of the currently loaded model."""
    try:
        from parakeet_mlx_guiapi.api.routes import get_transcriber
        transcriber = get_transcriber()
        if transcriber and hasattr(transcriber, 'model_name'):
            return transcriber.model_name
        # Fallback to config
        from parakeet_mlx_guiapi.utils.config import get_config
        return get_config().get("model_name", "unknown")
    except Exception:
        try:
            from parakeet_mlx_guiapi.utils.config import get_config
            return get_config().get("model_name", "unknown")
        except Exception:
            return "unknown"


class TranscriptionWorker:
    """
    Background worker for processing audio chunks.

    Uses a queue to ensure chunks are processed in order without blocking
    the WebSocket receive loop.
    """

    def __init__(self, session: LiveTranscriptionSession, ws, send_lock: threading.Lock):
        self.session = session
        self.ws = ws
        self.send_lock = send_lock
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _safe_send(self, data: dict):
        """Thread-safe WebSocket send."""
        try:
            with self.send_lock:
                self.ws.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

    def _process_loop(self):
        """Main processing loop running in background thread."""
        while self.running:
            try:
                # Wait for work with timeout to allow checking running flag
                try:
                    work_item = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if work_item is None:  # Shutdown signal
                    break

                audio_data, chunk_start, chunk_num = work_item

                try:
                    import base64
                    import time
                    audio_bytes = base64.b64decode(audio_data)
                    audio_size_kb = len(audio_bytes) / 1024
                    audio_duration_sec = (len(audio_bytes) - 44) / (16000 * 2)  # WAV: 16kHz, 16-bit

                    logger.info(f"[Worker] Processing chunk #{chunk_num}: {audio_size_kb:.1f} KB ({audio_duration_sec:.1f}s audio) at {chunk_start:.1f}s")

                    self._safe_send({
                        'type': 'debug',
                        'message': f'Processing chunk #{chunk_num}: {audio_size_kb:.1f} KB ({audio_duration_sec:.1f}s audio) at {chunk_start:.1f}s'
                    })

                    # Process the audio with timing
                    start_time = time.time()
                    messages = self.session.process_audio_chunk(audio_data, chunk_start)
                    elapsed = time.time() - start_time

                    logger.info(f"[Worker] Chunk #{chunk_num} complete: {len(messages)} segment(s) in {elapsed:.1f}s (RTF: {elapsed/audio_duration_sec:.2f}x)")

                    # Include detailed diarization info in debug message
                    diar_info = getattr(self.session, '_last_diarization_info', None)
                    if diar_info and diar_info.get('ran'):
                        diar_msg = f"diarization: {diar_info['raw_speakers']} speakers, {diar_info['raw_segments']} segs in {diar_info['time']:.1f}s"
                        speakers_found = diar_info.get('merged_speakers', [])
                    elif diar_info:
                        diar_msg = f"diarization: FAILED ({diar_info.get('error', diar_info.get('reason', 'unknown'))})"
                        speakers_found = []
                    else:
                        diar_msg = "diarization: NO INFO"
                        speakers_found = []

                    self._safe_send({
                        'type': 'debug',
                        'message': f'Chunk #{chunk_num} done: {len(messages)} seg in {elapsed:.1f}s | {diar_msg} | speakers: {speakers_found}'
                    })

                    if messages:
                        for i, msg in enumerate(messages):
                            text_preview = msg.get('text', '')[:50]
                            logger.debug(f"  Segment {i}: [{msg.get('speaker', '?')}] {text_preview}...")

                    # Send transcription result
                    self._safe_send({
                        'type': 'transcription',
                        'messages': messages
                    })

                    # Update queue status
                    remaining = self.queue.qsize()
                    if remaining > 0:
                        self._safe_send({
                            'type': 'status',
                            'message': f'{remaining} chunk(s) queued'
                        })

                except Exception as e:
                    logger.error(f"[Worker] Error processing chunk #{chunk_num}: {e}")
                    logger.error(traceback.format_exc())
                    self._safe_send({
                        'type': 'error',
                        'message': f'Transcription failed: {str(e)}'
                    })

                self.queue.task_done()

            except Exception as e:
                logger.error(f"[Worker] Unexpected error in process loop: {e}")
                logger.error(traceback.format_exc())

    def add_chunk(self, audio_data: str, chunk_start: float, chunk_num: int):
        """Add an audio chunk to the processing queue."""
        queue_size = self.queue.qsize()
        logger.info(f"[Worker] Queuing chunk #{chunk_num} (queue size: {queue_size})")
        self.queue.put((audio_data, chunk_start, chunk_num))

        self._safe_send({
            'type': 'status',
            'message': f'Chunk #{chunk_num} queued ({queue_size + 1} in queue)',
            'debug': {
                'chunk_num': chunk_num,
                'queue_size': queue_size + 1,
                'stage': 'queued'
            }
        })

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.queue.put(None)  # Shutdown signal
        self.thread.join(timeout=2.0)


def setup_live_routes(app: Flask):
    """
    Set up live transcription routes on the Flask app.

    Args:
        app: Flask application instance
    """
    sock = Sock(app)

    @app.route('/live')
    def live_transcription_page():
        """Serve the live transcription UI."""
        return render_template('live_transcription.html')

    @sock.route('/ws/live-transcribe')
    def live_transcribe(ws):
        """
        WebSocket endpoint for live transcription.

        Message Protocol:
        -----------------
        Client -> Server:
            {type: "config", enable_diarization: bool, silence_threshold: float}
            {type: "audio_chunk", data: "<base64 WAV>", chunk_start: float}
            {type: "export", format: "txt" | "srt"}
            {type: "clear"}

        Server -> Client:
            {type: "connected", session_id: str}
            {type: "status", message: str}
            {type: "transcription", messages: [...]}
            {type: "export_result", content: str, filename: str}
            {type: "error", message: str}
        """
        # Read diarization setting from config (menu settings = source of truth)
        from parakeet_mlx_guiapi.utils.config import get_config
        config = get_config()
        enable_diarization = config.get("diarization_enabled", True)
        hf_token = config.get("huggingface_token", None)
        logger.info(f"Creating session: diarization_enabled={enable_diarization}, hf_token={'set' if hf_token else 'NOT SET'}")

        # Create a new session
        session = LiveTranscriptionSession(enable_diarization=enable_diarization)
        _sessions[session.session_id] = session

        # Lock for thread-safe WebSocket sends
        send_lock = threading.Lock()

        # Create worker for background processing
        worker = TranscriptionWorker(session, ws, send_lock)
        chunk_counter = 0

        def safe_send(data: dict):
            """Thread-safe WebSocket send."""
            with send_lock:
                ws.send(json.dumps(data))

        # Check diarization status (this triggers lazy init)
        diarizer_ready = session.diarizer is not None
        logger.info(f"Session {session.session_id}: diarizer_ready={diarizer_ready}, enable_diarization={session.enable_diarization}")

        # Send connection confirmation
        safe_send({
            'type': 'connected',
            'session_id': session.session_id,
            'diarization_available': diarizer_ready,
            'diarization_enabled': session.enable_diarization,
            'model_name': get_current_model_name()
        })

        try:
            while True:
                # Receive message (this won't block processing anymore)
                data = ws.receive()
                if data is None:
                    break

                try:
                    message = json.loads(data)
                    msg_type = message.get('type')

                    if msg_type == 'config':
                        # Update session configuration
                        if 'enable_diarization' in message:
                            session.enable_diarization = message['enable_diarization']
                            # Reset diarizer if needed
                            if not session.enable_diarization:
                                session._diarizer = None

                        if 'similarity_threshold' in message:
                            new_threshold = float(message['similarity_threshold'])
                            session._embedding_similarity_threshold = new_threshold
                            logger.info(f"Updated similarity threshold to {new_threshold}")

                        safe_send({
                            'type': 'status',
                            'message': f'Configuration updated (threshold: {session._embedding_similarity_threshold:.2f})'
                        })

                    elif msg_type == 'audio_chunk':
                        # Queue audio chunk for processing
                        chunk_counter += 1
                        logger.info(f"=== AUDIO CHUNK #{chunk_counter} RECEIVED ===")

                        audio_data = message.get('data')
                        chunk_start = message.get('chunk_start', 0.0)

                        if not audio_data:
                            logger.error("No audio data in message!")
                            safe_send({
                                'type': 'error',
                                'message': 'No audio data provided'
                            })
                            continue

                        # Calculate audio size for debug
                        import base64
                        audio_bytes = base64.b64decode(audio_data)
                        audio_size_kb = len(audio_bytes) / 1024
                        logger.info(f"Audio chunk #{chunk_counter}: {audio_size_kb:.1f} KB at {chunk_start:.1f}s")

                        # Queue for processing (non-blocking)
                        worker.add_chunk(audio_data, chunk_start, chunk_counter)

                    elif msg_type == 'export':
                        # Export conversation
                        export_format = message.get('format', 'txt')

                        if export_format == 'txt':
                            content = session.export_txt()
                            filename = f'transcription_{session.session_id}.txt'
                        elif export_format == 'srt':
                            content = session.export_srt()
                            filename = f'transcription_{session.session_id}.srt'
                        else:
                            safe_send({
                                'type': 'error',
                                'message': f'Unknown export format: {export_format}'
                            })
                            continue

                        safe_send({
                            'type': 'export_result',
                            'content': content,
                            'filename': filename
                        })

                    elif msg_type == 'clear':
                        # Clear session
                        session.clear()
                        safe_send({
                            'type': 'status',
                            'message': 'Session cleared'
                        })

                    else:
                        safe_send({
                            'type': 'error',
                            'message': f'Unknown message type: {msg_type}'
                        })

                except json.JSONDecodeError:
                    safe_send({
                        'type': 'error',
                        'message': 'Invalid JSON message'
                    })
                except Exception as e:
                    print(f"Error handling message: {e}")
                    traceback.print_exc()
                    safe_send({
                        'type': 'error',
                        'message': f'Error: {str(e)}'
                    })

        finally:
            # Stop worker thread
            worker.stop()

            # Cleanup session
            if session.session_id in _sessions:
                del _sessions[session.session_id]
