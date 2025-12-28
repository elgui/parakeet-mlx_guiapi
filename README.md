# Parakeet-MLX GUI and API 🦜✨

This project provides a comprehensive GUI and REST API for the amazing [Parakeet-MLX](https://github.com/mlx-community/parakeet-mlx) speech-to-text library, which is a fantastic implementation of Nvidia's ASR (Automatic Speech Recognition) models for Apple Silicon using MLX.

## Features 🚀

- Web GUI for easy transcription of audio files (Gradio interface) 🌐
- REST API endpoints for seamless integration with other applications 🔌
- Support for multiple output formats (TXT, SRT, VTT, JSON, CSV) 📄
- Word-level timestamp highlighting in subtitle formats ✨
- Chunking of long audio files for better memory management 🧠
- Visualization of transcription results with timeline and heatmap 📊
- Audio segment extraction and playback 🎧
- **Live microphone recording** with direct transcription 🎤
- **Clipboard integration** for quick copy of transcription results 📋
- Comprehensive CLI client with pip-installable commands 💻

## Prerequisites ✅

- Python 3.8 or higher 🐍
- ffmpeg installed (required by Parakeet-MLX) 🛠️
- macOS with Apple Silicon (M1/M2/M3 chip) 🍎
- MLX framework 💪
- Original parakeet-mlx library 📚

## Installation ⬇️

1. Make sure ffmpeg is installed:
   ```bash
   brew install ffmpeg
   ```
   👍

2. Clone the `parakeet-mlx` repository in the **same parent directory** where you plan to clone this repository.
   ```bash
   # Navigate to the desired parent directory
   cd /path/to/your/projects/directory
   git clone https://github.com/mlx-community/parakeet-mlx.git
   ```
   📂

3. Clone this repository (`parakeet-mlx_guiapi`) in the **same parent directory** as `parakeet-mlx`.
   ```bash
   # Assuming you are still in the parent directory from the previous step
   git clone https://github.com/yourusername/parakeet-mlx_guiapi.git
   cd parakeet-mlx_guiapi
   ```
   📁

4. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   📦✨

## Usage ▶️

### Starting the Server 🚀

Run the server with:

```bash
python run.py
```

Or with custom options:

```bash
python run.py --host 127.0.0.1 --port 8000 --debug --model <model_name>
```

Server options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port for the Flask API (default: 5000)
- `--debug`: Enable debug mode
- `--model`: Specify the ASR model to use

The server will start and be accessible at:
- Gradio Web GUI: http://localhost:5001 (port + 1) 🌐
- REST API: http://localhost:5000/api/ 🔌

Note: The Gradio UI runs on port+1 from the specified port (default: 5001).

### Web GUI 🖥️

1. Open your browser and navigate to http://localhost:5001 (Gradio interface)
2. Upload an audio file using the interface ⬆️
3. Configure transcription options:
   - Output Format: json, txt, srt, or vtt 📄
   - Highlight Words: Enable word-level timestamps in SRT/VTT ✨
   - Chunk Duration: Duration in seconds for chunking long audio (0 to disable) ⏱️
   - Overlap Duration: Overlap duration in seconds when using chunking 🔄
4. Click "Transcribe" and view the results 🎉

### REST API 🤖

The following API endpoints are available:

#### Transcribe Audio 🎤➡️📄

```
POST /api/transcribe
```

Parameters (multipart/form-data):
- `file`: The audio file to transcribe (required) ⬆️
- `output_format`: Format for output (json, txt, srt, vtt, csv) (optional, default: json) 📄
- `highlight_words`: Enable word-level timestamps (optional, default: false) ✨
- `chunk_duration`: Duration in seconds for chunking long audio (optional, default: 120) ⏱️
- `overlap_duration`: Overlap duration in seconds when using chunking (optional, default: 15) 🔄

Response:
- For JSON format: JSON object with transcription results and visualizations 📊
- For other formats: File download with the appropriate content type ⬇️

Example cURL request:
```bash
curl -X POST -F "file=@audio.mp3" -F "output_format=json" http://localhost:5000/api/transcribe
```

#### Get Audio Segment ✂️🎧

```
POST /api/segment
```

Parameters (multipart/form-data):
- `file`: The audio file to extract segment from (required) ⬆️
- `start_time`: Start time in seconds (required) ⏱️
- `end_time`: End time in seconds (required) ⏱️

Response:
- WAV audio file containing the requested segment 🎧

Example cURL request:
```bash
curl -X POST -F "file=@audio.mp3" -F "start_time=10" -F "end_time=20" http://localhost:5000/api/segment -o segment.wav
```

#### Get Available Models 🧠

```
GET /api/models
```

Response:
- JSON array of available model names 📜

Example cURL request:
```bash
curl http://localhost:5000/api/models
```

### CLI Client 💻

After installation, you can use the CLI client for file transcription and microphone recording:

#### File Transcription
```bash
# Basic transcription (outputs JSON)
python client.py audio.mp3

# Specify output format
python client.py audio.mp3 --output-format srt

# With chunking for long audio files
python client.py audio.mp3 --output-format json --chunk-duration 120

# Extract a specific segment
python client.py audio.mp3 --segment 10-20 --output-file segment.wav

# Generate visualization
python client.py audio.mp3 --output-format json --visualize
```

#### Microphone Recording 🎤
```bash
# Record from microphone and transcribe (press Enter to stop)
python client.py --mic

# Record and copy transcription to clipboard
python client.py --mic --clipboard

# Record and save to file
python client.py --mic --output-file transcription.txt
```

#### CLI Options Reference

| Option | Description |
|--------|-------------|
| `--mic` | Record from microphone instead of using a file |
| `--clipboard` | Copy transcription result to clipboard |
| `--api-url` | Base URL for the API (default: http://localhost:5000/api) |
| `--output-format` | Output format: json, txt, srt, vtt, csv (default: json) |
| `--highlight-words` | Enable word-level timestamps in SRT/VTT |
| `--chunk-duration` | Chunking duration in seconds (default: 120, 0 to disable) |
| `--overlap-duration` | Overlap duration in seconds (default: 15) |
| `--output-file` | Output file path |
| `--segment` | Extract segment (format: start_time-end_time) |
| `--visualize` | Generate visualization (JSON output only) |

#### Installable Commands

After installing with pip (`pip install -e .`), you can use:
- `parakeet-server` - Start the server
- `parakeet-client` - Run the CLI client

## License 📜

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing 👋

Pull requests are welcome! Feel free to contribute bug fixes or new features. We appreciate your contributions! 🙏

## Acknowledgments 🙌

- This project uses the fantastic [Parakeet-MLX](https://github.com/mlx-community/parakeet-mlx) as its core library
- Thanks to [Nvidia](https://www.nvidia.com/) for training these powerful models
- Thanks to [MLX](https://github.com/ml-explore/mlx) for providing the incredible framework
- Special thanks to [Sam Witteveen](https://github.com/samwit) for his inspirational code and his insightful [YouTube channel](https://www.youtube.com/@samwitteveenai)
