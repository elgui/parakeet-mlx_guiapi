"""
Audio processing module for Parakeet-MLX GUI and API.

This module provides the AudioProcessor class for processing audio files.
"""

import os
import tempfile
from pathlib import Path
import io

class AudioProcessor:
    """
    Class for processing audio files.
    """
    
    @staticmethod
    def preprocess_audio(audio_file, target_sr=16000):
        """
        Preprocess audio file - convert to mono and resample if needed.
        
        Parameters:
        - audio_file: Path to the audio file or file-like object
        - target_sr: Target sample rate
        
        Returns:
        - Processed audio path
        """
        try:
            from pydub import AudioSegment
            
            # Handle file-like objects
            if hasattr(audio_file, 'read'):
                # Create a temporary file to save the uploaded content
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, "temp_upload.wav")
                with open(temp_path, 'wb') as f:
                    f.write(audio_file.read())
                audio_file.seek(0)  # Reset file pointer
                audio_path = temp_path
            else:
                audio_path = audio_file
            
            # Load the audio
            audio = AudioSegment.from_file(audio_path)
            
            # Check if we need to preprocess
            resampled = False
            mono = False
            
            # Resample if needed
            if audio.frame_rate != target_sr:
                print(f"Resampling audio from {audio.frame_rate}Hz to {target_sr}Hz")
                audio = audio.set_frame_rate(target_sr)
                resampled = True
            
            # Convert to mono if needed
            if audio.channels > 1:
                print(f"Converting {audio.channels} channels to mono")
                audio = audio.set_channels(1)
                mono = True
            
            # Export processed audio if needed
            if resampled or mono:
                temp_dir = tempfile.gettempdir()
                processed_path = os.path.join(temp_dir, f"processed_{Path(audio_path).name}")
                audio.export(processed_path, format="wav")
                return processed_path
            
            return audio_path
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return audio_path
    
    @staticmethod
    def get_audio_segment(audio_path, start_time, end_time):
        """
        Extract a segment from an audio file.
        
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
            print(f"Error extracting audio segment: {e}")
            return None
    
    @staticmethod
    def get_audio_duration(audio_path):
        """
        Get the duration of an audio file.
        
        Parameters:
        - audio_path: Path to the audio file
        
        Returns:
        - Duration in seconds
        """
        try:
            from pydub import AudioSegment
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Get duration
            return audio.duration_seconds
            
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0
