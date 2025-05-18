"""
Configuration utilities for Parakeet-MLX GUI and API.

This module provides functions for managing configuration.
"""

import os
import tempfile
import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "model_name": "mlx-community/parakeet-tdt-0.6b-v2",
    "upload_folder": os.path.join(tempfile.gettempdir(), 'parakeet_uploads'),
    "result_folder": os.path.join(tempfile.gettempdir(), 'parakeet_results'),
    "default_chunk_duration": 120,
    "default_overlap_duration": 15,
    "max_upload_size_mb": 100,
    "supported_formats": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
    "debug": False
}

# Global configuration
_config = None

def get_config():
    """
    Get the configuration.
    
    Returns:
    - Configuration dictionary
    """
    global _config
    
    if _config is None:
        # Initialize with default config
        _config = DEFAULT_CONFIG.copy()
        
        # Create directories if they don't exist
        os.makedirs(_config["upload_folder"], exist_ok=True)
        os.makedirs(_config["result_folder"], exist_ok=True)
        
        # Load from environment variables if available
        if "PARAKEET_MODEL_NAME" in os.environ:
            _config["model_name"] = os.environ["PARAKEET_MODEL_NAME"]
        
        if "PARAKEET_UPLOAD_FOLDER" in os.environ:
            _config["upload_folder"] = os.environ["PARAKEET_UPLOAD_FOLDER"]
            os.makedirs(_config["upload_folder"], exist_ok=True)
        
        if "PARAKEET_RESULT_FOLDER" in os.environ:
            _config["result_folder"] = os.environ["PARAKEET_RESULT_FOLDER"]
            os.makedirs(_config["result_folder"], exist_ok=True)
        
        if "PARAKEET_DEBUG" in os.environ:
            _config["debug"] = os.environ["PARAKEET_DEBUG"].lower() == "true"
        
        # Try to load from config file if it exists
        config_path = Path.home() / ".parakeet_mlx_guiapi.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                    _config.update(file_config)
            except Exception as e:
                print(f"Error loading config file: {e}")
    
    return _config

def save_config(config):
    """
    Save the configuration to a file.
    
    Parameters:
    - config: Configuration dictionary
    """
    global _config
    
    # Update the global config
    _config = config
    
    # Save to config file
    config_path = Path.home() / ".parakeet_mlx_guiapi.json"
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config file: {e}")

def get_supported_formats():
    """
    Get the list of supported audio formats.
    
    Returns:
    - List of supported file extensions
    """
    config = get_config()
    return config["supported_formats"]
