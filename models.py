"""
Model loading and management
~120 lines
"""
import torch
import whisperx
import gc
from typing import Optional, Dict, Any
from config import Config

class ModelManager:
    """Manage AI models for transcription and speaker recognition"""
    
    def __init__(self):
        self.model_cache: Dict[str, Any] = {}
    
    def get_whisper_model(self, model_size: str = None) -> Any:
        """Load or retrieve cached WhisperX model - GPU ONLY"""
        if model_size is None:
            model_size = Config.DEFAULT_MODEL_SIZE
            
        if not torch.cuda.is_available():
            raise RuntimeError("GPU lost! CUDA is no longer available.")
        
        if model_size not in self.model_cache:
            print(f"Loading WhisperX model '{model_size}' on CUDA...")
            self.model_cache[model_size] = whisperx.load_model(
                model_size, 
                "cuda",
                compute_type=Config.COMPUTE_TYPE
            )
            print(f"✅ Model loaded on GPU")
        
        return self.model_cache[model_size]
    
    def get_align_model(self, language_code: str) -> tuple:
        """Load alignment model for timestamps"""
        print(f"Loading alignment model for {language_code}...")
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device="cuda"
        )
        return model_a, metadata
    
    def get_speaker_model(self):
        """Load SpeechBrain speaker recognition model"""
        if 'speaker_model' not in self.model_cache:
            try:
                from speechbrain.inference.speaker import SpeakerRecognition
                
                print("Loading SpeechBrain speaker recognition model...")
                self.model_cache['speaker_model'] = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=f"{Config.MODELS_FOLDER}/speaker_recognition",
                    run_opts={"device": "cuda"}
                )
                print("✅ Speaker recognition model loaded")
            except ImportError as e:
                print(f"SpeechBrain not available: {e}")
                return None
            except Exception as e:
                print(f"Error loading speaker model: {e}")
                return None
        
        return self.model_cache.get('speaker_model')
    
    def get_diarization_pipeline(self):
        """Load Pyannote diarization pipeline"""
        if 'diarization_pipeline' not in self.model_cache:
            if not Config.HF_TOKEN:
                print("⚠️ No HF_TOKEN found, diarization disabled")
                return None
            
            try:
                from pyannote.audio import Pipeline
                
                print("Loading diarization pipeline...")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=Config.HF_TOKEN
                )
                pipeline.to(torch.device("cuda"))
                self.model_cache['diarization_pipeline'] = pipeline
                print("✅ Diarization pipeline loaded")
            except Exception as e:
                print(f"Error loading diarization: {e}")
                return None
        
        return self.model_cache.get('diarization_pipeline')
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ GPU memory cleaned")
    
    def clear_cache(self, model_type: str = None):
        """Clear specific or all cached models"""
        if model_type:
            if model_type in self.model_cache:
                del self.model_cache[model_type]
                print(f"✅ Cleared {model_type} from cache")
        else:
            self.model_cache.clear()
            print("✅ Cleared all models from cache")
        
        self.cleanup_memory()

# Global model manager instance
model_manager = ModelManager()
