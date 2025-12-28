#!/usr/bin/env python3
"""
Client script for Parakeet-MLX API.

This script provides a command-line client for the Parakeet-MLX API.
Supports both file-based transcription via API and live microphone recording.
"""

import argparse
import requests
import json
import os
import sys
from pathlib import Path

# Add the parakeet-mlx directory to sys.path for direct transcription
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parakeet_path = os.path.join(parent_dir, 'parakeet-mlx')
sys.path.insert(0, current_dir)
sys.path.insert(0, parakeet_path)


def transcribe_from_microphone(args):
    """
    Record audio from microphone and transcribe it.

    Parameters:
    - args: Parsed command-line arguments

    Returns:
    - Exit code (0 for success, 1 for failure)
    """
    try:
        import pyperclip
        from parakeet_mlx_guiapi.microphone import MicrophoneRecorder
        from parakeet_mlx_guiapi.transcription.transcriber import AudioTranscriber
    except ImportError as e:
        print(f"Error: Missing dependency for microphone mode: {e}")
        print("Install with: pip install sounddevice pyperclip")
        return 1

    temp_audio_path = None

    try:
        # Record audio from microphone
        print("Starting microphone recording...")
        recorder = MicrophoneRecorder()
        temp_audio_path = recorder.record_until_keypress()

        # Transcribe using local transcriber (bypass API)
        print("Transcribing audio...")
        transcriber = AudioTranscriber()
        df, full_text = transcriber.transcribe(
            temp_audio_path,
            chunk_duration=args.chunk_duration if args.chunk_duration > 0 else None,
            overlap_duration=args.overlap_duration
        )

        if df is None or full_text is None:
            print("Error: Transcription failed")
            return 1

        print("\n" + "=" * 50)
        print("TRANSCRIPTION:")
        print("=" * 50)
        print(full_text)
        print("=" * 50 + "\n")

        # Handle clipboard output
        if args.clipboard:
            pyperclip.copy(full_text)
            print("Transcription copied to clipboard!")

        # Handle file output
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                if args.output_format == 'json':
                    import json
                    json.dump({
                        "text": full_text,
                        "segments": df.to_dict(orient='records')
                    }, f, indent=2, ensure_ascii=False)
                else:
                    f.write(full_text)
            print(f"Transcription saved to: {args.output_file}")

        return 0

    except KeyboardInterrupt:
        print("\nRecording cancelled.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Cleaned up temporary file: {temp_audio_path}")


def main():
    parser = argparse.ArgumentParser(description='Parakeet-MLX API Client')
    parser.add_argument('audio_file', type=str, nargs='?', help='Path to the audio file to transcribe (not needed with --mic)')
    parser.add_argument('--mic', action='store_true', help='Record from microphone instead of using a file')
    parser.add_argument('--clipboard', action='store_true', help='Copy transcription result to clipboard')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000/api', help='Base URL for the API')
    parser.add_argument('--output-format', type=str, choices=['json', 'txt', 'srt', 'vtt', 'csv'], default='json', help='Output format')
    parser.add_argument('--highlight-words', action='store_true', help='Enable word-level timestamps in SRT/VTT')
    parser.add_argument('--chunk-duration', type=float, default=120, help='Chunking duration in seconds (0 to disable)')
    parser.add_argument('--overlap-duration', type=float, default=15, help='Overlap duration in seconds')
    parser.add_argument('--output-file', type=str, help='Output file path (default: based on input filename)')
    parser.add_argument('--segment', type=str, help='Extract a specific segment (format: start_time-end_time in seconds)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization (only for JSON output)')

    args = parser.parse_args()

    # Handle microphone recording mode
    if args.mic:
        return transcribe_from_microphone(args)

    # For file-based mode, audio_file is required
    if not args.audio_file:
        print("Error: audio_file is required (or use --mic for microphone recording)")
        parser.print_help()
        return 1

    # Check if the audio file exists
    if not os.path.isfile(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        return 1

    # Determine output file name if not specified
    if not args.output_file:
        input_path = Path(args.audio_file)
        output_path = input_path.with_suffix(f".{args.output_format}")
        args.output_file = str(output_path)

    # Handle segment extraction
    if args.segment:
        try:
            start_time, end_time = map(float, args.segment.split('-'))
            return extract_segment(args.audio_file, start_time, end_time, args.api_url, args.output_file)
        except ValueError:
            print(f"Error: Invalid segment format. Use 'start_time-end_time' in seconds.")
            return 1

    # Prepare the transcription request
    url = f"{args.api_url}/transcribe"

    with open(args.audio_file, 'rb') as f:
        files = {'file': (os.path.basename(args.audio_file), f)}
        data = {
            'output_format': args.output_format,
            'highlight_words': str(args.highlight_words).lower(),
            'chunk_duration': str(args.chunk_duration),
            'overlap_duration': str(args.overlap_duration)
        }

        print(f"Transcribing {args.audio_file}...")
        print(f"Sending request to {url}")

        try:
            response = requests.post(url, files=files, data=data)

            if response.status_code == 200:
                # Handle JSON response
                if args.output_format == 'json':
                    try:
                        json_data = response.json()

                        # Save the JSON response
                        with open(args.output_file, 'w', encoding='utf-8') as output_file:
                            json.dump(json_data, output_file, indent=2, ensure_ascii=False)

                        print(f"Transcription completed successfully.")
                        print(f"Output saved to: {args.output_file}")

                        # Handle visualization
                        if args.visualize and 'visualization' in json_data:
                            import base64
                            viz_path = Path(args.output_file).with_suffix('.png')
                            with open(viz_path, 'wb') as viz_file:
                                viz_file.write(base64.b64decode(json_data['visualization']))
                            print(f"Visualization saved to: {viz_path}")

                        return 0
                    except json.JSONDecodeError:
                        print("Warning: Could not parse JSON response. Saving raw content.")

                # Save the response content for non-JSON formats
                with open(args.output_file, 'wb') as output_file:
                    output_file.write(response.content)

                print(f"Transcription completed successfully.")
                print(f"Output saved to: {args.output_file}")

                # Print a preview for text formats
                if args.output_format in ['txt', 'srt', 'vtt', 'csv']:
                    content = response.content.decode('utf-8')
                    print("\nPreview:")
                    print("-" * 40)
                    print(content[:500] + "..." if len(content) > 500 else content)
                    print("-" * 40)

                return 0
            else:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return 1

        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            return 1

def extract_segment(audio_file, start_time, end_time, api_url, output_file):
    """
    Extract a specific segment of audio.

    Parameters:
    - audio_file: Path to the audio file
    - start_time: Start time in seconds
    - end_time: End time in seconds
    - api_url: Base URL for the API
    - output_file: Output file path

    Returns:
    - Exit code (0 for success, 1 for failure)
    """
    url = f"{api_url}/segment"

    with open(audio_file, 'rb') as f:
        files = {'file': (os.path.basename(audio_file), f)}
        data = {
            'start_time': str(start_time),
            'end_time': str(end_time)
        }

        print(f"Extracting segment {start_time}s - {end_time}s from {audio_file}...")
        print(f"Sending request to {url}")

        try:
            response = requests.post(url, files=files, data=data)

            if response.status_code == 200:
                # Save the response content
                with open(output_file, 'wb') as output_file:
                    output_file.write(response.content)

                print(f"Segment extraction completed successfully.")
                print(f"Output saved to: {output_file}")
                return 0
            else:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return 1

        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            return 1

if __name__ == '__main__':
    sys.exit(main())
