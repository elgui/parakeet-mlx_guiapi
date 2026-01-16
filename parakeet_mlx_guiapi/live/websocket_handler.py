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
from flask import Flask, render_template, make_response
from flask_sock import Sock

from .session import LiveTranscriptionSession
from parakeet_mlx_guiapi.providers import ProviderType

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

# Default Deepgram API key (can be overridden via config)
DEFAULT_DEEPGRAM_API_KEY = "a783ca9fdf636b7209dfb2cbd8dd8a9636e22a08"


def get_current_model_name():
    """Get the name of the currently loaded model."""
    try:
        from parakeet_mlx_guiapi.api.routes import get_transcriber
        transcriber = get_transcriber()
        if transcriber and hasattr(transcriber, 'model_name'):
            return transcriber.model_name
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
                try:
                    work_item = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if work_item is None:
                    break

                audio_data, chunk_start, chunk_num = work_item

                try:
                    import base64
                    import time
                    audio_bytes = base64.b64decode(audio_data)
                    audio_size_kb = len(audio_bytes) / 1024
                    audio_duration_sec = (len(audio_bytes) - 44) / (16000 * 2)

                    provider_name = self.session.provider_type.value
                    logger.info(f"[Worker] Processing chunk #{chunk_num} with {provider_name}: {audio_size_kb:.1f} KB ({audio_duration_sec:.1f}s audio)")

                    self._safe_send({
                        'type': 'debug',
                        'message': f'[{provider_name}] Processing chunk #{chunk_num}: {audio_size_kb:.1f} KB ({audio_duration_sec:.1f}s audio)'
                    })

                    start_time = time.time()
                    messages = self.session.process_audio_chunk(audio_data, chunk_start)
                    elapsed = time.time() - start_time

                    logger.info(f"[Worker] Chunk #{chunk_num} complete: {len(messages)} segment(s) in {elapsed:.1f}s")

                    diar_info = getattr(self.session, '_last_diarization_info', None)
                    if diar_info and diar_info.get('ran'):
                        provider = diar_info.get('provider', 'unknown')
                        speakers = diar_info.get('speakers', [])
                        diar_msg = f"{provider}: {len(speakers)} speaker(s)"
                    else:
                        diar_msg = "no diarization"

                    self._safe_send({
                        'type': 'debug',
                        'message': f'Chunk #{chunk_num} done: {len(messages)} seg in {elapsed:.1f}s | {diar_msg}'
                    })

                    if messages:
                        for i, msg in enumerate(messages):
                            text_preview = msg.get('text', '')[:50]
                            logger.debug(f"  Segment {i}: [{msg.get('speaker', '?')}] {text_preview}...")

                    self._safe_send({
                        'type': 'transcription',
                        'messages': messages
                    })

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
        self.queue.put(None)
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
        """Serve the live transcription UI with no-cache headers."""
        response = make_response(render_template('live_transcription.html'))
        # Prevent browser caching during development
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    @sock.route('/ws/live-transcribe')
    def live_transcribe(ws):
        """
        WebSocket endpoint for live transcription.

        Message Protocol:
        -----------------
        Client -> Server:
            {type: "config", enable_diarization: bool, provider: "parakeet"|"deepgram"}
            {type: "audio_chunk", data: "<base64 WAV>", chunk_start: float}
            {type: "export", format: "txt" | "srt"}
            {type: "clear"}

        Server -> Client:
            {type: "connected", session_id: str, providers: [...]}
            {type: "status", message: str}
            {type: "transcription", messages: [...]}
            {type: "export_result", content: str, filename: str}
            {type: "error", message: str}
        """
        from parakeet_mlx_guiapi.utils.config import get_config
        config = get_config()
        enable_diarization = config.get("diarization_enabled", True)
        hf_token = config.get("huggingface_token", None)

        # Get provider from config (default to parakeet)
        provider_name = config.get("stt_provider", "parakeet")
        try:
            provider_type = ProviderType(provider_name)
        except ValueError:
            provider_type = ProviderType.PARAKEET

        # Build provider config
        provider_config = {}
        if provider_type == ProviderType.PARAKEET:
            provider_config["hf_token"] = hf_token
        elif provider_type == ProviderType.DEEPGRAM:
            provider_config["api_key"] = config.get("deepgram_api_key", DEFAULT_DEEPGRAM_API_KEY)
            provider_config["model"] = config.get("deepgram_model", "nova-3")
            # Pass Deepgram options from config
            provider_config["options"] = config.get("deepgram_options", {})

        logger.info(f"Creating session: provider={provider_type.value}, diarization={enable_diarization}")

        # Create session with selected provider
        session = LiveTranscriptionSession(
            enable_diarization=enable_diarization,
            provider_type=provider_type,
            provider_config=provider_config
        )
        _sessions[session.session_id] = session

        send_lock = threading.Lock()
        worker = TranscriptionWorker(session, ws, send_lock)
        chunk_counter = 0

        def safe_send(data: dict):
            with send_lock:
                ws.send(json.dumps(data))

        # Check provider availability
        available, msg = session.provider.is_available()
        diarizer_ready = session.provider.supports_diarization and available

        logger.info(f"Session {session.session_id}: provider={session.provider.name}, available={available}")

        # Send connection confirmation with available providers
        safe_send({
            'type': 'connected',
            'session_id': session.session_id,
            'provider': session.provider_type.value,
            'provider_name': session.provider.name,
            'providers': [
                {'id': 'parakeet', 'name': 'Parakeet-MLX (Local)', 'supports_diarization': True},
                {'id': 'deepgram', 'name': 'Deepgram (Cloud)', 'supports_diarization': True}
            ],
            'diarization_available': diarizer_ready,
            'diarization_enabled': session.enable_diarization,
            'model_name': get_current_model_name() if provider_type == ProviderType.PARAKEET else "Deepgram Nova-2"
        })

        try:
            while True:
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

                        if 'similarity_threshold' in message:
                            new_threshold = float(message['similarity_threshold'])
                            session._embedding_similarity_threshold = new_threshold
                            logger.info(f"Updated similarity threshold to {new_threshold}")

                        # Handle provider/model change
                        if 'provider' in message:
                            new_provider = message['provider']
                            new_model = message.get('model')
                            try:
                                new_provider_type = ProviderType(new_provider)
                                # Check if provider or model changed
                                provider_changed = new_provider_type != session.provider_type
                                model_changed = new_model and new_model != session.provider_config.get('model')

                                if provider_changed or model_changed:
                                    # Need to create new session with different provider/model
                                    old_messages = session.messages.copy()

                                    new_config = {}
                                    if new_provider_type == ProviderType.PARAKEET:
                                        new_config["hf_token"] = hf_token
                                        if new_model:
                                            new_config["model_name"] = new_model
                                    elif new_provider_type == ProviderType.DEEPGRAM:
                                        new_config["api_key"] = config.get("deepgram_api_key", DEFAULT_DEEPGRAM_API_KEY)
                                        if new_model:
                                            new_config["model"] = new_model
                                        # Pass Deepgram options from config
                                        new_config["options"] = config.get("deepgram_options", {})

                                    # Create new session
                                    new_session = LiveTranscriptionSession(
                                        session_id=session.session_id,
                                        enable_diarization=session.enable_diarization,
                                        provider_type=new_provider_type,
                                        provider_config=new_config
                                    )
                                    new_session.messages = old_messages

                                    # Copy speaker tracking state from old session
                                    new_session._speaker_color_map = session._speaker_color_map.copy()
                                    new_session._speaker_embeddings = session._speaker_embeddings.copy()
                                    new_session._next_global_speaker_id = session._next_global_speaker_id

                                    # Update worker to use new session
                                    worker.session = new_session
                                    session = new_session
                                    _sessions[session.session_id] = session

                                    logger.info(f"Switched to provider: {new_provider_type.value}, model: {new_model}")

                                    safe_send({
                                        'type': 'provider_changed',
                                        'provider': new_provider_type.value,
                                        'provider_name': session.provider.name,
                                        'model': new_model
                                    })
                            except ValueError:
                                safe_send({
                                    'type': 'error',
                                    'message': f'Unknown provider: {new_provider}'
                                })

                        safe_send({
                            'type': 'status',
                            'message': f'Config updated: {session.provider.name}'
                        })

                    elif msg_type == 'audio_chunk':
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

                        import base64
                        audio_bytes = base64.b64decode(audio_data)
                        audio_size_kb = len(audio_bytes) / 1024
                        logger.info(f"Audio chunk #{chunk_counter}: {audio_size_kb:.1f} KB at {chunk_start:.1f}s")

                        worker.add_chunk(audio_data, chunk_start, chunk_counter)

                    elif msg_type == 'export':
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
            worker.stop()
            if session.session_id in _sessions:
                del _sessions[session.session_id]
