"""
Main application module for Parakeet-MLX GUI and API.

This module provides the main Flask application for the Parakeet-MLX GUI and API.
"""

import os
import sys
from flask import Flask, render_template
from flask_cors import CORS
import gradio as gr

# Add the parakeet-mlx sibling directory to sys.path so we can import it
parakeet_mlx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parakeet-mlx')
if os.path.exists(parakeet_mlx_path):
    sys.path.insert(0, parakeet_mlx_path)

# Import from our package
from parakeet_mlx_guiapi.api import setup_api_routes
from parakeet_mlx_guiapi.ui import create_gradio_interface
from parakeet_mlx_guiapi.utils.config import get_config
from parakeet_mlx_guiapi.live import setup_live_routes

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up API routes
setup_api_routes(app)

# Set up live transcription routes (WebSocket + /live page)
setup_live_routes(app)

# Create Gradio interface
demo = create_gradio_interface()

# Add a route to serve the Gradio interface within an iframe
@app.route('/')
def index():
    # The Gradio demo will be running on port 5001 (args.port + 1)
    gradio_url = "http://localhost:5001" # Assuming localhost for now, will need to make this dynamic if host changes
    return render_template('index.html', gradio_url=gradio_url)


if __name__ == '__main__':
    # Get configuration
    config = get_config()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=config["debug"])
