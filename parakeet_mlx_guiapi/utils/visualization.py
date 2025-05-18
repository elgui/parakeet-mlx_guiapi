"""
Visualization utilities for Parakeet-MLX GUI and API.

This module provides functions for visualizing transcription results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64

def visualize_transcript(df, max_segments=50):
    """
    Create a visualization of the transcript with segment durations.
    
    Parameters:
    - df: DataFrame with Start, End, and Segment columns
    - max_segments: Maximum number of segments to display
    
    Returns:
    - Base64 encoded PNG image
    """
    if df is None or len(df) == 0:
        return None
    
    # Limit the number of segments to display
    if len(df) > max_segments:
        df = df.iloc[:max_segments].copy()
    
    # Calculate segment durations if not already present
    if 'Duration' not in df.columns:
        df['Duration'] = df['End (s)'].astype(float) - df['Start (s)'].astype(float)
    
    # Create the plot
    plt.figure(figsize=(12, max(6, len(df) * 0.25)))
    
    # Plot segments as horizontal bars
    for i, row in df.iterrows():
        start = float(row['Start (s)'])
        duration = float(row['Duration'])
        plt.barh(i, duration, left=start, height=0.8, color='skyblue', alpha=0.7)
        
        # Add text labels for short segments
        if len(row['Segment']) < 30:
            plt.text(start + duration/2, i, row['Segment'],
                    va='center', ha='center', fontsize=9)
    
    # Set labels and title
    plt.yticks(range(len(df)), [f"{i+1}" for i in range(len(df))])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Segment #')
    plt.title('Transcript Timeline')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def create_transcript_heatmap(df):
    """
    Create a heatmap visualization of the transcript showing speech density.
    
    Parameters:
    - df: DataFrame with Start, End, and Segment columns
    
    Returns:
    - Base64 encoded PNG image
    """
    if df is None or len(df) == 0:
        return None
    
    # Get the total duration of the audio
    total_duration = df['End (s)'].max()
    
    # Create a timeline with 1-second bins
    timeline = np.zeros(int(total_duration) + 1)
    
    # Fill the timeline with speech activity
    for _, row in df.iterrows():
        start = int(float(row['Start (s)']))
        end = int(float(row['End (s)']))
        timeline[start:end+1] += 1
    
    # Create the plot
    plt.figure(figsize=(12, 4))
    
    # Plot the heatmap
    plt.imshow([timeline], aspect='auto', cmap='viridis')
    plt.colorbar(label='Speech Density')
    
    # Set labels and title
    plt.xlabel('Time (seconds)')
    plt.title('Speech Density Timeline')
    
    # Set x-axis ticks
    tick_interval = max(1, int(total_duration / 20))
    plt.xticks(np.arange(0, len(timeline), tick_interval), 
               np.arange(0, len(timeline), tick_interval))
    
    # Remove y-axis ticks
    plt.yticks([])
    
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str
