"""
Audio transcription module for Parakeet-MLX GUI and API.

This module provides the AudioTranscriber class for transcribing audio files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import io

# Import the parakeet_mlx library (installed via pip)
from parakeet_mlx import from_pretrained


class AudioTranscriber:
    def __init__(self, model_name="mlx-community/parakeet-tdt-0.6b-v3"):
        """
        Initialize the transcriber with the specified model.

        Parameters:
        - model_name: HuggingFace model path for the ASR model
        """
        print(f"Loading model: {model_name}...")
        self.model = from_pretrained(model_name)
        print("Model loaded successfully")

    def preprocess_audio(self, audio_path):
        """
        Preprocess audio - convert to mono and resample if needed.

        Parameters:
        - audio_path: Path to the audio file

        Returns:
        - Processed audio path and duration in seconds
        """
        from pydub import AudioSegment
        
        print(f"Loading audio: {Path(audio_path).name}")

        # Load the audio
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds
        print(f"Audio duration: {duration_sec:.2f} seconds")

        # Check if we need to preprocess
        resampled = False
        mono = False
        processed_path = audio_path

        # Resample if needed (Parakeet expects 16kHz)
        target_sr = 16000
        if audio.frame_rate != target_sr:
            print(f"Resampling audio from {audio.frame_rate}Hz to {target_sr}Hz")
            audio = audio.set_frame_rate(target_sr)
            resampled = True

        # Convert to mono if needed
        if audio.channels == 2:
            print("Converting stereo to mono")
            audio = audio.set_channels(1)
            mono = True
        elif audio.channels > 2:
            print(f"Warning: Audio has {audio.channels} channels. Only mono (1) or stereo (2) supported.")
            print("Converting to mono")
            audio = audio.set_channels(1)
            mono = True

        # Export processed audio if needed
        if resampled or mono:
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            processed_path = os.path.join(temp_dir, f"{Path(audio_path).stem}_processed.wav")
            audio.export(processed_path, format="wav")
            print(f"Processed audio saved to: {processed_path}")

        return processed_path, duration_sec

    def transcribe(self, audio_path, chunk_duration=120, overlap_duration=15, output_csv=None):
        """
        Transcribe audio and return timestamps and segments.

        Parameters:
        - audio_path: Path to the audio file
        - chunk_duration: Duration of each chunk in seconds (0 to disable)
        - overlap_duration: Overlap duration in seconds
        - output_csv: Optional path to save CSV output

        Returns:
        - DataFrame with transcription results
        """
        try:
            # Preprocess audio
            processed_path, duration_sec = self.preprocess_audio(audio_path)
            
            # Perform transcription
            print("Transcribing audio...")
            result = self.model.transcribe(
                processed_path,
                chunk_duration=chunk_duration if chunk_duration > 0 else None,
                overlap_duration=overlap_duration
            )
            
            # Extract sentences and tokens
            sentences = result.sentences
            
            # Create DataFrame
            data = {
                "Start (s)": [round(sentence.start, 2) for sentence in sentences],
                "End (s)": [round(sentence.end, 2) for sentence in sentences],
                "Segment": [sentence.text for sentence in sentences],
                "Duration": [round(sentence.duration, 2) for sentence in sentences],
                "Tokens": [sentence.tokens for sentence in sentences]
            }
            
            df = pd.DataFrame(data)
            
            # Save to CSV if requested
            if output_csv:
                df.to_csv(output_csv, index=False)
                print(f"Transcription saved to: {output_csv}")
            
            # Return the DataFrame and full text
            return df, result.text
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None, None
            
        finally:
            # Cleanup
            if 'processed_path' in locals() and processed_path != audio_path and Path(processed_path).exists():
                Path(processed_path).unlink()
                print(f"Removed temporary file: {processed_path}")

    def get_segment_audio(self, audio_path, start_time, end_time):
        """
        Get a specific segment of audio as bytes

        Parameters:
        - audio_path: Path to the audio file
        - start_time: Start time in seconds
        - end_time: End time in seconds

        Returns:
        - Audio segment as bytes
        """
        try:
            from pydub import AudioSegment
            
            # Convert times to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)

            # Load audio
            audio = AudioSegment.from_file(audio_path)

            # Cut segment
            segment = audio[start_ms:end_ms]

            # Export to bytes
            buffer = io.BytesIO()
            segment.export(buffer, format="wav")
            buffer.seek(0)
            
            return buffer.read()

        except Exception as e:
            print(f"Error getting segment audio: {e}")
            return None
