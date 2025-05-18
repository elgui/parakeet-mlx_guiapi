# Parakeet-MLX GUI and API Knowledge Base

This document contains important information about the Parakeet-MLX GUI and API project, including architecture, design decisions, and implementation details.

## Project Structure

The project is organized as a Python package with the following structure:

```
parakeet_mlx_guiapi/
├── __init__.py
├── api/
│   ├── __init__.py
│   └── routes.py
├── audio/
│   ├── __init__.py
│   └── processor.py
├── transcription/
│   ├── __init__.py
│   └── transcriber.py
├── ui/
│   ├── __init__.py
│   └── gradio_interface.py
└── utils/
    ├── __init__.py
    ├── config.py
    └── visualization.py
```

- **api/**: Contains API routes and handlers
- **audio/**: Contains audio processing utilities
- **transcription/**: Contains the transcription engine
- **ui/**: Contains the Gradio UI components
- **utils/**: Contains utility functions and configuration

## Key Components

### AudioTranscriber

The `AudioTranscriber` class in `transcription/transcriber.py` is the core component that handles audio transcription. It:

1. Loads the Parakeet-MLX model
2. Preprocesses audio files (resampling, converting to mono)
3. Transcribes audio with timestamps
4. Handles long audio files with chunking

### AudioProcessor

The `AudioProcessor` class in `audio/processor.py` provides utilities for:

1. Preprocessing audio files
2. Extracting segments from audio files
3. Getting audio file information

### Visualization

The visualization utilities in `utils/visualization.py` provide:

1. Timeline visualization of transcription segments
2. Heatmap visualization of speech density

### Configuration

The configuration utilities in `utils/config.py` provide:

1. Default configuration values
2. Loading configuration from environment variables
3. Loading configuration from a config file
4. Saving configuration to a file

## API Endpoints

The API provides the following endpoints:

- `POST /api/transcribe`: Transcribe an audio file
- `POST /api/segment`: Extract a segment from an audio file
- `GET /api/models`: Get available models

## UI Components

The UI is built with Gradio and provides:

1. A transcription interface for uploading and transcribing audio files
2. A segment player for playing specific segments of audio
3. Visualization of transcription results

## Client

The client script (`client.py`) provides a command-line interface for:

1. Transcribing audio files
2. Extracting segments from audio files
3. Saving transcription results in various formats
4. Generating visualizations

## Dependencies

The project depends on:

- **Flask**: Web framework for the API
- **Gradio**: UI framework
- **MLX**: Apple's machine learning framework
- **Parakeet-MLX**: Speech-to-text library
- **Pydub**: Audio processing library
- **Pandas**: Data manipulation library
- **Matplotlib**: Visualization library

## Design Decisions

1. **Modular Architecture**: The project is organized into modules to make it easy to maintain and extend.
2. **Separation of Concerns**: The API, UI, and transcription engine are separated to make it easy to modify one without affecting the others.
3. **Configuration Management**: Configuration is centralized and can be loaded from environment variables or a config file.
4. **Error Handling**: Comprehensive error handling is implemented to make the application robust.
5. **Visualization**: Visualization is provided to make it easy to understand transcription results.

## Future Improvements

1. **Multiple Model Support**: Add support for multiple models and model switching
2. **Batch Processing**: Add support for batch processing of multiple audio files
3. **Speaker Diarization**: Add support for speaker diarization
4. **Real-time Transcription**: Add support for real-time transcription
5. **User Authentication**: Add user authentication for the API
6. **Caching**: Add caching to improve performance
