"""
Speaker recognition and diarization
~200 lines
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cosine
from config import Config
from models import model_manager

class SpeakerRecognition:
    """Handle speaker embedding and matching"""
    
    @staticmethod
    def compute_embedding(audio_path: str, start: float, end: float) -> np.ndarray:
        """Compute speaker embedding using SpeechBrain"""
        try:
            return SpeakerRecognition.compute_embedding_speechbrain(audio_path, start, end)
        except Exception as e:
            print(f"Warning: Could not compute real embedding: {e}")
            # Fallback to random embedding
            return np.random.randn(Config.EMBEDDING_DIMENSION).astype(np.float32)
    
    @staticmethod
    def compute_embedding_speechbrain(audio_path: str, start: float, end: float) -> np.ndarray:
        """Compute speaker embedding using SpeechBrain's speaker verification model"""
        try:
            import torchaudio
            
            model = model_manager.get_speaker_model()
            if model is None:
                raise RuntimeError("Speaker model not available")
            
            # Load and extract audio segment
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Calculate frame positions
            start_frame = int(start * sample_rate)
            end_frame = int(end * sample_rate)
            
            # Extract segment
            if start_frame < waveform.shape[1] and end_frame <= waveform.shape[1]:
                segment = waveform[:, start_frame:end_frame]
            else:
                segment = waveform
            
            # Ensure minimum length (1 second)
            min_samples = sample_rate
            if segment.shape[1] < min_samples:
                padding = min_samples - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, padding))
            
            # Compute embedding
            embeddings = model.encode_batch(segment)
            embedding_array = embeddings.squeeze().cpu().numpy()
            
            # Ensure correct dimension
            if embedding_array.shape[0] != Config.EMBEDDING_DIMENSION:
                if embedding_array.shape[0] > Config.EMBEDDING_DIMENSION:
                    embedding_array = embedding_array[:Config.EMBEDDING_DIMENSION]
                else:
                    padding = np.zeros(Config.EMBEDDING_DIMENSION - embedding_array.shape[0])
                    embedding_array = np.concatenate([embedding_array, padding])
            
            return embedding_array
            
        except Exception as e:
            print(f"Error computing embedding with SpeechBrain: {e}")
            return np.random.randn(Config.EMBEDDING_DIMENSION).astype(np.float32)
    
    @staticmethod
    def match_speakers(segments: List[Dict], audio_path: str, 
                      known_speakers: List[Tuple[str, str, np.ndarray]],
                      audio_processor) -> Tuple[List[Dict], Dict[str, str]]:
        """Match detected speakers to known speakers using embeddings"""
        
        if not known_speakers:
            print("No known speakers to match against")
            return segments, {}
        
        print(f"Found {len(known_speakers)} known speakers with embeddings")
        
        # Group segments by speaker
        speaker_segments = {}
        for segment in segments:
            speaker = segment.get('speaker')
            if speaker and speaker.startswith('SPEAKER_'):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(segment)
        
        speaker_mapping = {}
        
        # Process each unknown speaker
        for speaker_id, speaker_segs in speaker_segments.items():
            print(f"Processing {speaker_id}...")
            
            # Find best segment for embedding
            best_segment = audio_processor.find_best_speaker_segment(
                speaker_segs, speaker_id
            )
            if not best_segment:
                best_segment = speaker_segs[0]
            
            # Compute embedding for this speaker
            embedding = SpeakerRecognition.compute_embedding(
                audio_path,
                best_segment.get('start', 0),
                best_segment.get('end', 0)
            )
            
            if embedding is None:
                continue
            
            # Normalize embedding dimension
            embedding = SpeakerRecognition.normalize_embedding(embedding)
            
            # Compare with all known speakers
            best_match = None
            best_similarity = 0
            similarities = []
            
            for known_id, known_name, known_embedding in known_speakers:
                # Normalize known embedding
                known_emb = SpeakerRecognition.normalize_embedding(known_embedding)
                
                try:
                    # Compute cosine similarity
                    similarity = 1 - cosine(embedding, known_emb)
                    similarities.append((known_name, similarity))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = known_name
                except Exception as e:
                    print(f"  Error comparing with {known_name}: {e}")
                    continue
            
            # Show top matches for debugging
            if similarities:
                similarities.sort(key=lambda x: x[1], reverse=True)
                print(f"  Top matches for {speaker_id}:")
                for name, sim in similarities[:3]:
                    print(f"    - {name}: {sim:.3f}")
            
            # Apply threshold for matching
            if best_similarity > Config.SIMILARITY_THRESHOLD:
                speaker_mapping[speaker_id] = best_match
                print(f"✅ Matched {speaker_id} → {best_match} (similarity: {best_similarity:.3f})")
            else:
                print(f"❌ No match for {speaker_id} (best: {best_similarity:.3f} < {Config.SIMILARITY_THRESHOLD})")
        
        # Apply mapping to all segments
        for segment in segments:
            speaker = segment.get('speaker')
            if speaker in speaker_mapping:
                segment['original_speaker'] = speaker
                segment['speaker'] = speaker_mapping[speaker]
                segment['auto_matched'] = True
                segment['match_confidence'] = best_similarity
        
        return segments, speaker_mapping
    
    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to correct dimension"""
        if len(embedding) == Config.EMBEDDING_DIMENSION:
            return embedding
        
        if len(embedding) > Config.EMBEDDING_DIMENSION:
            return embedding[:Config.EMBEDDING_DIMENSION]
        
        # Pad if too short
        padding = np.zeros(Config.EMBEDDING_DIMENSION - len(embedding))
        return np.concatenate([embedding, padding])
