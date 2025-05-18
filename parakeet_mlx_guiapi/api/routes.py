"""
API routes for Parakeet-MLX GUI and API.

This module provides API routes for the Parakeet-MLX GUI and API.
"""

import os
import uuid
import json
from flask import request, jsonify, send_file
from werkzeug.utils import secure_filename

from parakeet_mlx_guiapi.utils.config import get_config
from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber
from parakeet_mlx_guiapi.audio.processor import AudioProcessor
from parakeet_mlx_guiapi.utils.visualization import visualize_transcript, create_transcript_heatmap

# Global transcriber instance
_transcriber = None

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
        
        # Get parameters from request
        output_format = request.form.get('output_format', 'json')
        highlight_words = request.form.get('highlight_words', 'false').lower() == 'true'
        chunk_duration = float(request.form.get('chunk_duration', get_config()["default_chunk_duration"]))
        overlap_duration = float(request.form.get('overlap_duration', get_config()["default_overlap_duration"]))
        
        # Save the file
        config = get_config()
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(config["upload_folder"], f"{file_id}_{filename}")
        file.save(file_path)
        
        try:
            # Get transcriber
            transcriber = get_transcriber()
            
            # Transcribe the file
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
