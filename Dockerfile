FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the parakeet-mlx library
COPY /projects/parakeet-mlx /app/parakeet-mlx

# Copy our application files
COPY /projects/parakeet-mlx_guiapi /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create temporary directories for uploads and results
RUN mkdir -p /tmp/parakeet_uploads /tmp/parakeet_results

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "run.py"]
