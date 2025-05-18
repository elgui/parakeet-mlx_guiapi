"""
Gradio interface for Parakeet-MLX GUI and API.

This module provides a Gradio interface for the Parakeet-MLX GUI and API.
"""

import time
import json
import pandas as pd
import gradio as gr
import tempfile
import os
from pathlib import Path

from parakeet_mlx_guiapi.utils.config import get_config
from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber
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

def create_gradio_interface():
    """
    Create a Gradio interface for the Parakeet-MLX GUI and API.
    
    Returns:
    - Gradio interface
    """
    
    def transcribe_audio(audio_file, output_format, highlight_words, chunk_duration, overlap_duration):
        """
        Transcribe an audio file.
        
        Parameters:
        - audio_file: Path to the audio file
        - output_format: Output format (json, txt, srt, vtt, csv)
        - highlight_words: Whether to highlight words in SRT/VTT
        - chunk_duration: Duration of each chunk in seconds
        - overlap_duration: Overlap duration in seconds
        
        Returns:
        - Transcription output
        - Status message
        - Visualization image
        - Segments DataFrame
        - Audio duration
        """
        start_time = time.time()
        
        try:
            # Get transcriber
            transcriber = get_transcriber()
            
            # Transcribe the file
            df, full_text = transcriber.transcribe(
                audio_file,
                chunk_duration=float(chunk_duration) if float(chunk_duration) > 0 else None,
                overlap_duration=float(overlap_duration)
            )
            
            if df is None:
                return None, "Error: Transcription failed", None, None, 0
            
            elapsed_time = time.time() - start_time
            
            # Create visualization
            viz_img = None
            if len(df) > 0:
                viz_base64 = visualize_transcript(df)
                if viz_base64:
                    # Create a temporary file for the visualization
                    temp_dir = tempfile.gettempdir()
                    viz_path = os.path.join(temp_dir, "transcript_viz.png")
                    with open(viz_path, "wb") as f:
                        import base64
                        f.write(base64.b64decode(viz_base64))
                    viz_img = viz_path
            
            # Prepare response based on output format
            if output_format == 'txt':
                response_data = full_text
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
            elif output_format == 'csv':
                # Convert DataFrame to CSV
                temp_dir = tempfile.gettempdir()
                csv_path = os.path.join(temp_dir, "transcript.csv")
                df.to_csv(csv_path, index=False)
                
                with open(csv_path, 'r') as f:
                    response_data = f.read()
            else:  # Default to JSON
                segments_list = df.to_dict(orient='records')
                
                # Convert AlignedToken objects in the "Tokens" column to dictionaries
                for segment in segments_list:
                    if "Tokens" in segment and isinstance(segment["Tokens"], list):
                        segment["Tokens"] = [
                            {"text": token.text, "start": token.start, "end": token.end}
                            for token in segment["Tokens"]
                        ]
                        
                response_data = json.dumps(
                    {
                        "text": full_text,
                        "segments": segments_list
                    }, 
                    indent=2, 
                    ensure_ascii=False
                )
            
            # Get audio duration
            audio_duration = 0
            if len(df) > 0:
                audio_duration = float(df['End (s)'].max())
            
            return response_data, f"Transcription completed in {elapsed_time:.2f} seconds", viz_img, df, audio_duration
            
        except Exception as e:
            return None, f"Error: {str(e)}", None, None, 0
    
    def play_segment(audio_file, segment_idx, segments_df, audio_duration):
        """
        Play a specific segment of audio.
        
        Parameters:
        - audio_file: Path to the audio file
        - segment_idx: Index of the segment to play
        - segments_df: DataFrame with segments
        - audio_duration: Duration of the audio file
        
        Returns:
        - Audio segment
        - Segment text
        - Status message
        """
        try:
            if audio_file is None:
                return None, "", "Error: No audio file"
            
            if segments_df is None or len(segments_df) == 0:
                return None, "", "Error: No segments available"
            
            # Convert segments_df from JSON string to DataFrame if needed
            if isinstance(segments_df, str):
                segments_df = pd.read_json(segments_df)
            
            # Check if segment_idx is valid
            if segment_idx < 0 or segment_idx >= len(segments_df):
                return None, "", f"Error: Invalid segment index {segment_idx}"
            
            # Get segment times
            start_time = float(segments_df.iloc[segment_idx]['Start (s)'])
            end_time = float(segments_df.iloc[segment_idx]['End (s)'])
            segment_text = segments_df.iloc[segment_idx]['Segment']
            
            # Validate times
            if start_time < 0:
                start_time = 0
            if end_time > audio_duration:
                end_time = audio_duration
            if start_time >= end_time:
                return None, "", f"Error: Invalid time range {start_time} - {end_time}"
            
            # Get transcriber
            transcriber = get_transcriber()
            
            # Get segment audio
            segment_data = transcriber.get_segment_audio(audio_file, start_time, end_time)
            
            if segment_data is None:
                return None, "", "Error: Failed to extract segment"
            
            # Create a temporary file for the segment
            temp_dir = tempfile.gettempdir()
            segment_path = os.path.join(temp_dir, "segment.wav")
            with open(segment_path, 'wb') as f:
                f.write(segment_data)
            
            return segment_path, segment_text, f"Playing segment {segment_idx+1}: {start_time:.2f}s - {end_time:.2f}s"
            
        except Exception as e:
            return None, "", f"Error: {str(e)}"
    
    # Create the Gradio interface
    with gr.Blocks(title="Parakeet-MLX Transcription") as demo:
        gr.Markdown("# Parakeet-MLX Transcription Service")
        gr.Markdown("Upload an audio file to transcribe it using Parakeet-MLX")
        
        # Hidden state variables
        segments_df_state = gr.State(None)
        audio_duration_state = gr.State(0)
        
        with gr.Tabs():
            with gr.TabItem("Transcribe"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(type="filepath", label="Audio Input")
                        
                        with gr.Row():
                            output_format = gr.Dropdown(
                                choices=["json", "txt", "srt", "vtt", "csv"], 
                                value="json", 
                                label="Output Format"
                            )
                            highlight_words = gr.Checkbox(
                                label="Highlight Words (for SRT/VTT)", 
                                value=False
                            )
                        
                        with gr.Row():
                            chunk_duration = gr.Number(
                                value=get_config()["default_chunk_duration"], 
                                label="Chunk Duration (seconds, 0 to disable)"
                            )
                            overlap_duration = gr.Number(
                                value=get_config()["default_overlap_duration"], 
                                label="Overlap Duration (seconds)"
                            )
                        
                        transcribe_btn = gr.Button("Transcribe", variant="primary")
                        status_text = gr.Markdown()
                    
                    with gr.Column(scale=3):
                        output_text = gr.Textbox(
                            label="Transcription Output", 
                            lines=20,
                            max_lines=50
                        )
                        
                        visualization_img = gr.Image(
                            label="Transcript Visualization",
                            type="filepath"
                        )
            
            with gr.TabItem("Segment Player"):
                with gr.Row():
                    with gr.Column(scale=1):
                        segment_idx = gr.Number(
                            value=0, 
                            label="Segment Index",
                            precision=0
                        )
                        play_segment_btn = gr.Button("Play Segment", variant="primary")
                        segment_status = gr.Markdown()
                    
                    with gr.Column(scale=2):
                        segment_text = gr.Textbox(
                            label="Segment Text", 
                            lines=3
                        )
                        segment_audio = gr.Audio(
                            label="Segment Audio",
                            type="filepath"
                        )
        
        # Set up event handlers
        transcribe_result = transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, output_format, highlight_words, chunk_duration, overlap_duration],
            outputs=[output_text, status_text, visualization_img, segments_df_state, audio_duration_state]
        )
        
        play_segment_btn.click(
            fn=play_segment,
            inputs=[audio_input, segment_idx, segments_df_state, audio_duration_state],
            outputs=[segment_audio, segment_text, segment_status]
        )
        
    return demo

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
