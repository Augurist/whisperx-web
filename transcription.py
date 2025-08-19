"""
Main transcription logic
~200 lines
"""
import torch
import json
import os
import whisperx
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from config import Config
from models import model_manager
from speaker_recognition import SpeakerRecognition
from audio_processing import AudioProcessor
from database import Database

class TranscriptionService:
    """Main transcription service"""
    
    def __init__(self, db: Database):
        self.db = db
        self.audio_processor = AudioProcessor()
        self.speaker_recognition = SpeakerRecognition()
    
    def _get_gpu_info(self):
        """Safely get GPU information"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except:
            pass
        return None
    
    def transcribe_file(self, filepath: str, unique_id: str, 
                       filename: str) -> Dict[str, Any]:
        """Main transcription pipeline"""
        
        wav_filepath = None
        try:
            print(f"Processing file: {filename}")
            
            # Convert to WAV if needed
            if not filepath.lower().endswith('.wav'):
                wav_filepath = self.audio_processor.convert_to_wav(filepath)
                process_file = wav_filepath if wav_filepath else filepath
            else:
                process_file = filepath
            
            # Load audio
            print("Loading audio...")
            audio = whisperx.load_audio(process_file)
            duration = len(audio) / Config.SAMPLE_RATE
            
            # Transcribe
            result = self._transcribe_audio(audio, process_file)
            
            # Process speakers if diarization enabled
            transcript_id = f"{unique_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            speakers_data = None
            
            if Config.HF_TOKEN:
                speakers_data = self._process_speakers(
                    result, process_file, transcript_id, audio
                )
            
            # Save transcript
            transcript_data = self._save_transcript(
                transcript_id, filename, duration, result, speakers_data
            )
            
            print(f"✅ Transcript saved: {transcript_id}")
            
            # Cleanup
            model_manager.cleanup_memory()
            
            return {
                'success': True,
                'transcript_id': transcript_id,
                'language': transcript_data['language'],
                'duration': transcript_data['duration'],
                'text': transcript_data['text'],
                'segments': transcript_data['segments'],
                'speakers': transcript_data['speakers'],
                'gpu_used': transcript_data.get('gpu_used')
            }
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            raise
        finally:
            # Cleanup temporary files
            if wav_filepath and os.path.exists(wav_filepath):
                os.remove(wav_filepath)
    
    def _transcribe_audio(self, audio: Any, process_file: str) -> Dict:
        """Transcribe audio using WhisperX"""
        # Get model
        model = model_manager.get_whisper_model()
        
        print("Transcribing with large model on GPU...")
        result = model.transcribe(
            audio,
            batch_size=Config.BATCH_SIZE,
            language=Config.DEFAULT_LANGUAGE
        )
        
        print(f"Language: {result['language']}")
        
        # Align timestamps
        print("Aligning timestamps...")
        model_a, metadata = model_manager.get_align_model(result["language"])
        
        aligned_result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            "cuda",
            return_char_alignments=False
        )
        
        result["segments"] = aligned_result["segments"]
        
        # Cleanup alignment model
        del model_a
        model_manager.cleanup_memory()
        
        return result
    
    def _process_speakers(self, result: Dict, process_file: str, 
                         transcript_id: str, audio: Any) -> Optional[Dict]:
        """Process speaker diarization"""
        print("Performing speaker diarization on GPU...")
        
        try:
            pipeline = model_manager.get_diarization_pipeline()
            if not pipeline:
                return None
            
            print(f"Running diarization on: {process_file}")
            diarization = pipeline(process_file)
            
            # Assign speakers to segments
            for segment in result["segments"]:
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", segment_start)
                segment_mid = (segment_start + segment_end) / 2
                
                # Find speaker at segment midpoint
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= segment_mid <= turn.end:
                        segment["speaker"] = speaker
                        break
            
            # Extract unique speakers
            speakers = set()
            for segment in result["segments"]:
                if "speaker" in segment:
                    speakers.add(segment["speaker"])
            
            speakers_data = {
                "enabled": True,
                "speakers": sorted(list(speakers)),
                "speaker_count": len(speakers)
            }
            
            print(f"✅ Found {len(speakers)} speakers")
            
            # Match to known speakers
            known_speakers = self.db.get_speakers_with_embeddings()
            if known_speakers:
                result["segments"], speaker_mapping = self.speaker_recognition.match_speakers(
                    result["segments"],
                    process_file,
                    known_speakers,
                    self.audio_processor
                )
                
                if speaker_mapping:
                    speakers_data["mappings"] = speaker_mapping
            
            # Extract and save speaker clips
            self._save_speaker_clips(
                result["segments"], process_file, transcript_id, speakers
            )
            
            return speakers_data
            
        except Exception as e:
            print(f"Diarization error: {str(e)}")
            return {"enabled": False, "error": str(e)}
    
    def _save_speaker_clips(self, segments: list, process_file: str, 
                           transcript_id: str, speakers: set):
        """Extract and save clips for each speaker"""
        for speaker in speakers:
            best_segment = self.audio_processor.find_best_speaker_segment(
                segments, speaker
            )
            
            if best_segment:
                clip_path = self.audio_processor.extract_speaker_clip(
                    process_file,
                    best_segment.get('start', 0),
                    best_segment.get('end', 0),
                    speaker,
                    transcript_id
                )
                
                if clip_path:
                    embedding = self.speaker_recognition.compute_embedding(
                        process_file,
                        best_segment.get('start', 0),
                        best_segment.get('end', 0)
                    )
                    
                    self.db.save_speaker(
                        speaker,
                        clip_path,
                        best_segment.get('text', ''),
                        embedding
                    )
    
    def _save_transcript(self, transcript_id: str, filename: str, 
                        duration: float, result: Dict, 
                        speakers_data: Optional[Dict]) -> Dict:
        """Save transcript to database and file"""
        # Create full text
        if speakers_data and speakers_data.get("enabled"):
            full_text = self._create_speaker_text(result.get('segments', []))
        else:
            full_text = ' '.join([seg.get('text', '') for seg in result.get('segments', [])])
        
        # Save to file
        transcript_path = os.path.join(
            Config.TRANSCRIPTS_FOLDER,
            f"{transcript_id}.json"
        )
        
        transcript_data = {
            'id': transcript_id,
            'filename': filename,
            'duration': duration,
            'language': result.get('language', 'en'),
            'segments': result.get('segments', []),
            'text': full_text,
            'speakers': speakers_data,
            'processed_at': datetime.now().isoformat(),
            'gpu_used': self._get_gpu_info(),
        }
        
        with open(transcript_path, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        
        return transcript_data
    
    def _create_speaker_text(self, segments: list) -> str:
        """Create text with speaker labels"""
        full_text = []
        current_speaker = None
        
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            text = seg.get('text', '').strip()
            
            if speaker != current_speaker:
                full_text.append(f"\n[{speaker}]: {text}")
                current_speaker = speaker
            else:
                full_text.append(text)
        
        return ' '.join(full_text)
