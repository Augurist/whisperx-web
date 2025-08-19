"""
Configuration module for WhisperX Web
~50 lines
"""
import os
import torch
import sys

class Config:
    """Application configuration"""
    
    # Paths
    UPLOAD_FOLDER = '/app/data/uploads'
    TRANSCRIPTS_FOLDER = '/app/data/transcripts'
    SPEAKER_CLIPS_FOLDER = '/app/data/speaker_clips'
    DATABASE_PATH = '/app/data/speakers.db'
    MODELS_FOLDER = '/app/data/models'
    
    # Limits
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg', 'webm'}
    
    # API Keys
    HF_TOKEN = os.environ.get('HF_TOKEN', '')
    
    # Model settings
    DEFAULT_MODEL_SIZE = 'large'
    DEFAULT_LANGUAGE = 'en'
    COMPUTE_TYPE = 'float16'
    BATCH_SIZE = 16
    
    # Speaker matching
    SIMILARITY_THRESHOLD = 0.6
    EMBEDDING_DIMENSION = 192
    
    # Audio processing
    SAMPLE_RATE = 16000
    MIN_CLIP_DURATION = 3
    MAX_CLIP_DURATION = 15
    IDEAL_CLIP_DURATION = 8
    
    @classmethod
    def check_gpu(cls):
        """Check if GPU is available and exit if not"""
        if not torch.cuda.is_available():
            print("=" * 60)
            print("ERROR: GPU NOT AVAILABLE!")
            print("=" * 60)
            sys.exit(1)
        
        print("=" * 60)
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ PyTorch Version: {torch.__version__}")
        print("=" * 60)
        return True
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        for folder in [cls.UPLOAD_FOLDER, cls.TRANSCRIPTS_FOLDER, 
                      cls.SPEAKER_CLIPS_FOLDER, cls.MODELS_FOLDER]:
            os.makedirs(folder, exist_ok=True)
