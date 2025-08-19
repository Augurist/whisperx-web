#!/bin/bash

# WhisperX Web Platform Setup Script
# For RTX 5090 with Docker and NVIDIA Container Toolkit

set -e

echo "🚀 WhisperX Web Platform Setup"
echo "=============================="

# Check for NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected. Please ensure drivers are installed."
    exit 1
fi

echo "✅ GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Check Docker
if ! docker --version &> /dev/null; then
    echo "❌ Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and back in for group changes."
fi

# Check NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "📦 Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo "✅ NVIDIA Container Toolkit installed"
fi

# Create project structure
echo "📁 Creating project structure..."
mkdir -p whisperx-web/{data,models_cache,templates,static,audio_files}
mkdir -p whisperx-web/data/{uploads,transcripts,speaker_clips}

cd whisperx-web

# Check for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  Hugging Face token not found in environment"
    echo "Please enter your Hugging Face token (for speaker diarization):"
    echo "Get one at: https://huggingface.co/settings/tokens"
    read -p "HF_TOKEN: " HF_TOKEN
    
    # Save to .env file
    echo "HF_TOKEN=$HF_TOKEN" > .env
    echo "✅ Token saved to .env file"
else
    echo "HF_TOKEN=$HF_TOKEN" > .env
    echo "✅ Using existing HF_TOKEN from environment"
fi

# Create requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    cat > requirements.txt << 'EOF'
torch==2.1.0
torchaudio==2.1.0
flask==3.0.0
numpy==1.24.3
ffmpeg-python==0.2.0
pyannote.audio==3.1.1
flask-sqlalchemy==3.1.1
scipy==1.11.4
speechbrain==0.5.16
webrtcvad==2.0.10
pydub==0.25.1
redis==5.0.1
celery==5.3.4
python-multipart==0.0.6
requests==2.31.0
EOF
    echo "✅ Created requirements.txt"
fi

# Download the enhanced app.py if not present
if [ ! -f app.py ]; then
    echo "📥 Please copy the enhanced app.py file to this directory"
    echo "   (The one with database integration and improved speaker clips)"
fi

# Create basic templates if not present
if [ ! -f templates/index.html ]; then
    echo "📥 Please copy the template files to templates/ directory"
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t whisperx-web .

# Create docker-compose.yml if not exists
if [ ! -f docker-compose.yml ]; then
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  whisperx-web:
    image: whisperx-web
    container_name: whisperx-web
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models_cache:/root/.cache
      - ./audio_files:/app/audio_files:ro
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7-alpine
    container_name: whisperx-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

volumes:
  redis_data:
    driver: local
EOF
    echo "✅ Created docker-compose.yml"
fi

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for health check
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check health
if curl -f http://localhost:5000/health &> /dev/null; then
    echo "✅ WhisperX Web Platform is running!"
    echo ""
    echo "🌐 Access the web interface at: http://localhost:5000"
    echo "📊 Features available:"
    echo "   • Audio transcription with speaker diarization"
    echo "   • Speaker identification and management"
    echo "   • Full-text search across transcripts"
    echo "   • Speaker voice samples and cataloging"
    echo ""
    echo "📝 Logs: docker-compose logs -f"
    echo "🛑 Stop: docker-compose down"
else
    echo "❌ Service failed to start. Check logs with: docker-compose logs"
    exit 1
fi
