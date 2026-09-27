# Sentinel Backend Production Dockerfile for Render
FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app \
    PORT=8000

# Install system dependencies: ffmpeg (video transcoding), OpenGL/GLib (OpenCV), curl (health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and pre-install CPU-only PyTorch to avoid massive CUDA wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application modules and root model weights
COPY backend /app/backend
COPY ai /app/ai
COPY database /app/database
COPY yolov8n.pt /app/yolov8n.pt

# Ensure storage directories exist
RUN mkdir -p /app/storage/uploads \
             /app/storage/evidence \
             /app/storage/evidence_playback \
             /app/storage/playback \
             /app/storage/events \
             /app/storage/reports

# Expose default port
EXPOSE 8000

# Start FastAPI backend with dynamic Render $PORT binding
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
