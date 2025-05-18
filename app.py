"""
Main application module for Parakeet-MLX GUI and API.

This module provides the main Flask application for the Parakeet-MLX GUI and API.
"""

import os
import sys
from flask import Flask, render_template
from flask_cors import CORS
import gradio as gr

# Add the parakeet-mlx directory to sys.path so we can import it
sys.path.append('/projects/parakeet-mlx')

# Import from our package
from parakeet_mlx_guiapi.api import setup_api_routes
from parakeet_mlx_guiapi.ui import create_gradio_interface
from parakeet_mlx_guiapi.utils.config import get_config

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up API routes
setup_api_routes(app)

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
