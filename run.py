#!/usr/bin/env python3
"""
Run script for Parakeet-MLX GUI and API.

This script provides a command-line interface for running the Parakeet-MLX GUI and API.
"""

import sys
import os
import argparse

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add sibling parakeet-mlx directory to sys.path
parakeet_mlx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parakeet-mlx')
if os.path.exists(parakeet_mlx_path):
    sys.path.insert(0, parakeet_mlx_path)

# Import app.py from the current directory
from app import app, demo
from parakeet_mlx_guiapi.utils.config import get_config, save_config

# Verify parakeet_mlx is installed
try:
    import parakeet_mlx
except ImportError:
    print("Error: parakeet-mlx is not installed.")
    print("Install it with: pip install parakeet-mlx")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Parakeet-MLX GUI and API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080, since 5000 is used by macOS AirPlay)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--model', type=str, help='Model name to use')

    args = parser.parse_args()

    # Get configuration
    config = get_config()

    # Update configuration from command-line arguments
    if args.debug:
        config["debug"] = True

    if args.model:
        config["model_name"] = args.model

    # Save updated configuration
    save_config(config)

    print(f"Starting Parakeet-MLX GUI and API server on {args.host}:{args.port}")
    print(f"Using model: {config['model_name']}")

    # Launch the Gradio demo on a different port
    gradio_port = int(os.environ.get("GRADIO_SERVER_PORT", args.port + 1))
    print(f"Starting Gradio demo on {args.host}:{gradio_port}")

    # Launch Gradio in non-blocking mode (in a thread)
    import threading
    gradio_thread = threading.Thread(
        target=lambda: demo.launch(
            server_name=args.host,
            server_port=gradio_port,
            debug=config["debug"],
            prevent_thread_lock=True
        ),
        daemon=True
    )
    gradio_thread.start()

    # Run the Flask app (this blocks)
    print(f"Starting Flask app on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=config["debug"])

if __name__ == '__main__':
    main()
