"""
API routes for Parakeet-MLX GUI and API.

This module provides API routes for the Parakeet-MLX GUI and API.
"""

import os
import uuid
import json
import pandas as pd
from flask import request, jsonify, send_file
from werkzeug.utils import secure_filename

from parakeet_mlx_guiapi.utils.config import get_config
from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber
from parakeet_mlx_guiapi.providers.base import ProviderType, get_provider, TranscriptionResult
from parakeet_mlx_guiapi.audio.processor import AudioProcessor
from parakeet_mlx_guiapi.utils.visualization import visualize_transcript, create_transcript_heatmap

# Global transcriber instance
_transcriber = None
_stt_provider_cache = {}

def get_transcriber():
    """
    Get the transcriber instance.
    
    Returns:
    - AudioTranscriber instance
    """
    global _transcriber
    
    if _transcriber is None:
        config = get_config()
        _transcriber = AudioTranscriber(model_name=config["model_name"])
    
    return _transcriber

def get_stt_provider(provider=None, model=None, deepgram_options=None, enable_diarization=None):
    """
    Get a cached STT provider instance for the resolved configuration.

    Parameters:
    - provider: Provider name
    - model: Model name
    - deepgram_options: Deepgram options dictionary
    - enable_diarization: Whether diarization is enabled

    Returns:
    - STTProvider instance
    """
    config = get_config()

    if provider is None:
        provider = config.get("stt_provider", "parakeet")

    if model is None:
        if provider == ProviderType.PARAKEET.value:
            model = config["model_name"]
        elif provider == ProviderType.DEEPGRAM.value:
            model = config.get("deepgram_model", "nova-3")

    if deepgram_options is None:
        deepgram_options = config.get("deepgram_options", {})

    if enable_diarization is None:
        enable_diarization = config.get("diarization_enabled", False)

    cache_key = (
        provider,
        model,
        frozenset(deepgram_options.items()) if deepgram_options else frozenset(),
        bool(enable_diarization)
    )

    if cache_key not in _stt_provider_cache:
        if provider == ProviderType.PARAKEET.value:
            _stt_provider_cache[cache_key] = get_provider(
                ProviderType.PARAKEET,
                model_name=model,
                hf_token=config.get("huggingface_token")
            )
        elif provider == ProviderType.DEEPGRAM.value:
            _stt_provider_cache[cache_key] = get_provider(
                ProviderType.DEEPGRAM,
                api_key=config.get("deepgram_api_key"),
                model=model,
                options=deepgram_options
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    return _stt_provider_cache[cache_key]

def _build_dataframe_from_result(result: TranscriptionResult):
    """
    Convert a TranscriptionResult into the legacy DataFrame/text shape.

    Parameters:
    - result: Provider transcription result

    Returns:
    - Tuple of (DataFrame, full_text)
    """
    has_speakers = any(segment.speaker for segment in result.segments)
    data = {
        "Start (s)": [],
        "End (s)": [],
        "Segment": [],
        "Duration": [],
        "Tokens": []
    }

    if has_speakers:
        data["Speaker"] = []

    for segment in result.segments:
        start = round(segment.start, 2)
        end = round(segment.end, 2)
        data["Start (s)"].append(start)
        data["End (s)"].append(end)
        data["Segment"].append(segment.text)
        data["Duration"].append(round(max(segment.end - segment.start, 0), 2))
        data["Tokens"].append([])

        if has_speakers:
            data["Speaker"].append(segment.speaker)

    return pd.DataFrame(data), result.full_text

def setup_api_routes(app):
    """
    Set up API routes for the Flask app.
    
    Parameters:
    - app: Flask app
    """
    
    @app.route('/api/transcribe', methods=['POST'])
    def api_transcribe():
        """
        Transcribe an audio file.
        """
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        config = get_config()

        # Get parameters from request
        output_format = request.form.get('output_format', 'json')
        highlight_words = request.form.get('highlight_words', 'false').lower() == 'true'
        chunk_duration = float(request.form.get('chunk_duration', config["default_chunk_duration"]))
        overlap_duration = float(request.form.get('overlap_duration', config["default_overlap_duration"]))

        provider = request.form.get('provider')
        if provider == '':
            provider = None
        elif provider is not None:
            provider = provider.lower()
            if provider not in {ProviderType.PARAKEET.value, ProviderType.DEEPGRAM.value}:
                return jsonify({"error": "provider must be 'parakeet' or 'deepgram'"}), 400

        model = request.form.get('model')
        if model == '':
            model = None

        deepgram_options_raw = request.form.get('deepgram_options')
        if deepgram_options_raw is None or deepgram_options_raw.strip() == '':
            deepgram_options = None
        else:
            try:
                deepgram_options = json.loads(deepgram_options_raw)
            except json.JSONDecodeError:
                return jsonify({"error": "deepgram_options must be valid JSON"}), 400

            if not isinstance(deepgram_options, dict):
                return jsonify({"error": "deepgram_options must be valid JSON"}), 400

        enable_diarization_raw = request.form.get('enable_diarization')
        if enable_diarization_raw is None or enable_diarization_raw.strip() == '':
            enable_diarization = None
        else:
            normalized_enable_diarization = enable_diarization_raw.lower()
            if normalized_enable_diarization == 'true':
                enable_diarization = True
            elif normalized_enable_diarization == 'false':
                enable_diarization = False
            else:
                return jsonify({"error": "enable_diarization must be 'true' or 'false'"}), 400
        
        # Save the file
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(config["upload_folder"], f"{file_id}_{filename}")
        file.save(file_path)
        
        try:
            if any(value is not None for value in (provider, model, deepgram_options, enable_diarization)):
                resolved_enable_diarization = enable_diarization
                if resolved_enable_diarization is None:
                    resolved_enable_diarization = config.get("diarization_enabled", False)

                stt = get_stt_provider(
                    provider=provider,
                    model=model,
                    deepgram_options=deepgram_options,
                    enable_diarization=enable_diarization
                )
                result = stt.transcribe(
                    file_path,
                    enable_diarization=resolved_enable_diarization,
                    chunk_duration=chunk_duration if chunk_duration > 0 else None
                )
                df, full_text = _build_dataframe_from_result(result)
            else:
                transcriber = get_transcriber()
                df, full_text = transcriber.transcribe(
                    file_path,
                    chunk_duration=chunk_duration if chunk_duration > 0 else None,
                    overlap_duration=overlap_duration
                )
            
            if df is None:
                return jsonify({"error": "Transcription failed"}), 500
            
            # Prepare response based on output format
            if output_format == 'txt':
                response_data = full_text
                content_type = 'text/plain'
            elif output_format == 'srt':
                # Convert DataFrame to SRT format
                srt_content = ""
                for i, row in df.iterrows():
                    start_time = float(row['Start (s)'])
                    end_time = float(row['End (s)'])
                    
                    # Format times as HH:MM:SS,mmm
                    start_formatted = format_time_srt(start_time)
                    end_formatted = format_time_srt(end_time)
                    
                    srt_content += f"{i+1}\n"
                    srt_content += f"{start_formatted} --> {end_formatted}\n"
                    srt_content += f"{row['Segment']}\n\n"
                
                response_data = srt_content
                content_type = 'text/plain'
            elif output_format == 'vtt':
                # Convert DataFrame to VTT format
                vtt_content = "WEBVTT\n\n"
                for i, row in df.iterrows():
                    start_time = float(row['Start (s)'])
                    end_time = float(row['End (s)'])
                    
                    # Format times as HH:MM:SS.mmm
                    start_formatted = format_time_vtt(start_time)
                    end_formatted = format_time_vtt(end_time)
                    
                    vtt_content += f"{start_formatted} --> {end_formatted}\n"
                    vtt_content += f"{row['Segment']}\n\n"
                
                response_data = vtt_content
                content_type = 'text/plain'
            elif output_format == 'csv':
                # Save to CSV and return the file
                result_path = os.path.join(config["result_folder"], f"{file_id}.csv")
                df.to_csv(result_path, index=False)
                
                return send_file(
                    result_path,
                    as_attachment=True,
                    download_name=f"{os.path.splitext(filename)[0]}.csv",
                    mimetype='text/csv'
                )
            else:  # Default to JSON
                # Create visualization
                viz_img = visualize_transcript(df)
                heatmap_img = create_transcript_heatmap(df)
                
                response_data = {
                    "text": full_text,
                    "segments": df.to_dict(orient='records'),
                    "visualization": viz_img,
                    "heatmap": heatmap_img
                }
                return jsonify(response_data)
            
            # For non-JSON formats, save to file and return the file
            result_path = os.path.join(config["result_folder"], f"{file_id}.{output_format}")
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(response_data)
            
            return send_file(
                result_path,
                as_attachment=True,
                download_name=f"{os.path.splitext(filename)[0]}.{output_format}",
                mimetype=content_type
            )
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @app.route('/api/models', methods=['GET'])
    def api_models():
        """
        Get available models.
        """
        config = get_config()
        return jsonify([config["model_name"]])
    
    @app.route('/api/segment', methods=['POST'])
    def api_segment():
        """
        Get a specific segment of audio.
        """
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Get parameters from request
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        
        if start_time >= end_time:
            return jsonify({"error": "Invalid time range"}), 400
        
        # Save the file
        config = get_config()
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(config["upload_folder"], f"{file_id}_{filename}")
        file.save(file_path)
        
        try:
            # Get the segment
            segment_data = AudioProcessor.get_audio_segment(file_path, start_time, end_time)
            
            if segment_data is None:
                return jsonify({"error": "Failed to extract segment"}), 500
            
            # Create a temporary file for the segment
            segment_path = os.path.join(config["result_folder"], f"{file_id}_segment.wav")
            with open(segment_path, 'wb') as f:
                f.write(segment_data)
            
            return send_file(
                segment_path,
                as_attachment=True,
                download_name=f"{os.path.splitext(filename)[0]}_segment.wav",
                mimetype='audio/wav'
            )
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

def format_time_srt(seconds):
    """
    Format time in seconds to SRT format (HH:MM:SS,mmm).
    
    Parameters:
    - seconds: Time in seconds
    
    Returns:
    - Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def format_time_vtt(seconds):
    """
    Format time in seconds to VTT format (HH:MM:SS.mmm).
    
    Parameters:
    - seconds: Time in seconds
    
    Returns:
    - Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"
