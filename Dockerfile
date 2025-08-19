FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    build-essential \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create python symlinks
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements
COPY requirements.txt .

# Install Python packages (remove torch-related from requirements first)
RUN grep -v "torch" requirements.txt > requirements_filtered.txt || true && \
    pip install --no-cache-dir -r requirements_filtered.txt

# Install WhisperX
RUN pip install git+https://github.com/m-bain/whisperx.git

# Install additional packages
RUN pip install \
    speechbrain \
    webrtcvad \
    pydub \
    redis \
    celery \
    python-multipart

# Create necessary directories
RUN mkdir -p /app/data/uploads \
    /app/data/transcripts \
    /app/data/speaker_clips \
    /app/data/models \
    /app/templates \
    /app/static

# Copy all application files
COPY config.py /app/
COPY database.py /app/
COPY audio_processing.py /app/
COPY models.py /app/
COPY speaker_recognition.py /app/
COPY transcription.py /app/
COPY routes.py /app/
COPY app.py /app/

# Copy templates and static directories
COPY templates/ /app/templates/
COPY static/ /app/static/

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
