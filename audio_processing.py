"""
Audio processing utilities
~150 lines
"""
import os
import subprocess
from typing import Optional, List, Dict
from config import Config

class AudioProcessor:
    """Handle audio file operations"""
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def convert_to_wav(input_path: str) -> Optional[str]:
        """Convert audio file to WAV format for processing"""
        output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
        
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', str(Config.SAMPLE_RATE),
                '-ac', '1',
                '-y',
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Converted audio to WAV: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg conversion failed: {e}")
            return None
    
    @staticmethod
    def extract_speaker_clip(audio_path: str, start: float, end: float, 
                           speaker_id: str, transcript_id: str) -> Optional[str]:
        """Extract high-quality audio clip for a speaker"""
        try:
            # Add padding for natural speech
            start = max(0, start - 0.2)
            end = end + 0.2
            duration = end - start
            
            # Ensure reasonable clip length
            if duration > Config.MAX_CLIP_DURATION:
                duration = Config.MAX_CLIP_DURATION
            elif duration < Config.MIN_CLIP_DURATION:
                duration = min(Config.MIN_CLIP_DURATION, duration)
            
            clip_filename = f"{transcript_id}_{speaker_id}_{int(start)}.mp3"
            clip_path = os.path.join(Config.SPEAKER_CLIPS_FOLDER, clip_filename)
            
            # High-quality extraction with normalization
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-ss', str(start),
                '-t', str(duration),
                '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Audio normalization
                '-acodec', 'libmp3lame',
                '-ab', '192k',  # Higher bitrate
                '-ar', '44100',
                '-y', clip_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(clip_path):
                # Verify the file isn't empty
                if os.path.getsize(clip_path) > 1000:  # At least 1KB
                    print(f"✅ Extracted clip for {speaker_id}: {clip_filename} ({duration:.1f}s)")
                    return clip_path
                else:
                    os.remove(clip_path)
                    print(f"❌ Clip too small, removed: {clip_path}")
            
        except Exception as e:
            print(f"❌ Failed to extract clip: {e}")
        
        return None
    
    @staticmethod
    def find_best_speaker_segment(segments: List[Dict], speaker: str) -> Optional[Dict]:
        """Find the best quality segment for a speaker"""
        speaker_segments = [s for s in segments if s.get('speaker') == speaker]
        
        if not speaker_segments:
            return None
        
        # Score segments based on quality criteria
        scored_segments = []
        for seg in speaker_segments:
            duration = seg.get('end', 0) - seg.get('start', 0)
            text = seg.get('text', '').strip()
            words = len(text.split())
            
            # Calculate quality score
            score = 0
            
            # Prefer segments with good length
            if 5 <= duration <= 10:
                score += 10
            elif 3 <= duration <= 15:
                score += 5
            
            # Prefer segments with meaningful text
            if words >= 10:
                score += 5
            if words >= 20:
                score += 5
            
            # Avoid very short or very long segments
            if duration < 2 or duration > 20:
                score -= 10
            
            scored_segments.append((seg, score))
        
        # Sort by score and return best
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        
        if scored_segments:
            return scored_segments[0][0]
        
        return None
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', 
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting duration: {e}")
        return 0.0
