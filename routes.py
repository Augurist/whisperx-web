"""
Flask routes
~280 lines
"""
import os
import torch
import json
import uuid
import requests
from datetime import datetime
from flask import render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from config import Config
from database import Database
from transcription import TranscriptionService
from audio_processing import AudioProcessor

def register_routes(app, db: Database):
    """Register all Flask routes"""
    
    transcription_service = TranscriptionService(db)
    audio_processor = AudioProcessor()
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/speakers')
    def speakers_page():
        return render_template('speakers.html')
    
    @app.route('/search')
    def search_page():
        return render_template('search.html')
    
    @app.route('/admin')
    def admin_page():
        """Admin interface for module testing"""
        return render_template('admin.html')
    
    @app.route('/api/check_updates')
    def check_updates():
        """Check for Docker image updates"""
        try:
            # Get current image from Dockerfile
            current_image = None
            try:
                with open('Dockerfile', 'r') as f:
                    for line in f:
                        if line.startswith('FROM pytorch/pytorch:'):
                            current_image = line.split('FROM ')[1].strip()
                            break
            except:
                current_image = "Unable to read Dockerfile"
            
            # Check Docker Hub for latest images
            url = "https://hub.docker.com/v2/repositories/pytorch/pytorch/tags/"
            params = {"page_size": 10, "ordering": "-last_updated"}
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                latest_images = []
                
                for tag in data['results']:
                    name = tag['name']
                    # Filter for CUDA 12.x runtime images (not release candidates)
                    if 'cuda12' in name and 'runtime' in name and 'rc' not in name:
                        latest_images.append({
                            'name': f"pytorch/pytorch:{name}",
                            'tag': name,
                            'updated': tag['last_updated'][:10],
                            'size_gb': round(tag.get('full_size', 0) / (1024**3), 2)
                        })
                        if len(latest_images) >= 5:
                            break
                
                # Check if update available
                update_available = False
                if latest_images and current_image:
                    if latest_images[0]['name'] != current_image:
                        update_available = True
                
                return jsonify({
                    'current': current_image,
                    'latest': latest_images,
                    'update_available': update_available,
                    'checked_at': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Failed to check Docker Hub'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        import torch
        
        if not torch.cuda.is_available():
            return jsonify({
                'status': 'error',
                'error': 'GPU not available',
                'cuda_available': False
            }), 500
        
        return jsonify({
            'status': 'healthy',
            'cuda_available': True,
            'device': 'cuda',
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            'diarization_available': bool(Config.HF_TOKEN)
        })
    
    @app.route('/transcribe', methods=['POST'])
    def transcribe_audio():
        """Main transcription endpoint"""
        import torch
        
        if not torch.cuda.is_available():
            return jsonify({'error': 'GPU not available! Cannot process.'}), 500
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not audio_processor.allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Add file size validation (e.g., max 500MB)
        if request.content_length > 500 * 1024 * 1024:
            return jsonify({'error': 'File too large. Maximum size is 500MB'}), 400
        
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(Config.UPLOAD_FOLDER, f"{unique_id}_{filename}")
        file.save(filepath)
        
        try:
            result = transcription_service.transcribe_file(
                filepath, unique_id, filename
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    @app.route('/api/speakers', methods=['GET'])
    def api_list_speakers():
        """List all known speakers"""
        speakers = db.get_all_speakers()
        return jsonify(speakers)
    
    @app.route('/api/speakers/<speaker_id>', methods=['GET', 'PUT', 'DELETE'])
    def api_speaker(speaker_id):
        """Get, update, or delete a specific speaker"""
        
        if request.method == 'GET':
            speaker = db.get_speaker(speaker_id)
            if not speaker:
                return jsonify({'error': 'Speaker not found'}), 404
            return jsonify(speaker)
        
        elif request.method == 'PUT':
            data = request.json
            new_name = data.get('name')
            
            if not new_name:
                return jsonify({'error': 'Name is required'}), 400
            
            success = db.update_speaker_name(speaker_id, new_name)
            if success:
                return jsonify({'success': True, 'name': new_name})
            return jsonify({'error': 'Update failed'}), 500
        
        elif request.method == 'DELETE':
            # TODO: Implement delete_speaker in database.py
            return jsonify({'success': True})
    
    @app.route('/api/speakers/merge', methods=['POST'])
    def merge_speakers():
        """Merge two speakers together"""
        data = request.json
        from_speaker = data.get('from')
        to_speaker = data.get('to')
        
        if not from_speaker or not to_speaker:
            return jsonify({'error': 'Both from and to speakers required'}), 400
        
        if from_speaker == to_speaker:
            return jsonify({'error': 'Cannot merge a speaker with itself'}), 400
        
        try:
            # Get the target speaker's name
            target_speaker = db.get_speaker(to_speaker)
            if not target_speaker:
                return jsonify({'error': f'Target speaker {to_speaker} not found'}), 404
            
            # Update the from_speaker to have the same name as to_speaker
            # This effectively merges them for display purposes
            success = db.update_speaker_name(from_speaker, target_speaker['name'])
            
            if success:
                # TODO: Implement actual merge in database.py to combine segments
                # TODO: Update transcript files to reflect the merge
                
                return jsonify({
                    'success': True, 
                    'message': f'Successfully merged {from_speaker} into {target_speaker["name"]}'
                })
            else:
                return jsonify({'error': 'Failed to update speaker name in database'}), 500
                
        except Exception as e:
            print(f"Error merging speakers: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Merge failed: {str(e)}'}), 500
    
    @app.route('/api/speaker_clip/<speaker_id>')
    def get_speaker_clip(speaker_id):
        """Get audio clip for a speaker"""
        # First try to get speaker from database
        speaker = db.get_speaker(speaker_id)
        
        if speaker and speaker.get('clip_path'):
            clip_path = speaker['clip_path']
            if os.path.exists(clip_path):
                return send_file(clip_path, mimetype='audio/mpeg')
        
        # Fallback to clips directory - speaker_id might be a filename
        clips_dir = Config.SPEAKER_CLIPS_FOLDER
        
        # If speaker_id looks like a filename, use it directly
        if speaker_id.endswith('.mp3'):
            clip_path = os.path.join(clips_dir, speaker_id)
            if os.path.exists(clip_path):
                return send_file(clip_path, mimetype='audio/mpeg')
        
        # Otherwise search for clips containing the speaker_id
        if os.path.exists(clips_dir):
            for filename in os.listdir(clips_dir):
                if speaker_id in filename and filename.endswith('.mp3'):
                    clip_path = os.path.join(clips_dir, filename)
                    return send_file(clip_path, mimetype='audio/mpeg')
        
        return jsonify({'error': 'Clip file not found'}), 404
    
    @app.route('/api/all_speaker_clips')
    def get_all_speaker_clips():
        """Get all speaker clips information"""
        clips = []
        clips_dir = Config.SPEAKER_CLIPS_FOLDER
        
        if os.path.exists(clips_dir):
            for filename in os.listdir(clips_dir):
                if filename.endswith('.mp3'):
                    # Parse filename format: {transcript_id}_{YYYY}{MM}{DD}_{HHMMSS}_SPEAKER_XX_{timestamp}.mp3
                    parts = filename.replace('.mp3', '').split('_')
                    
                    # Find the SPEAKER_XX part
                    speaker_id = None
                    for i, part in enumerate(parts):
                        if part == 'SPEAKER' and i + 1 < len(parts):
                            speaker_id = f"SPEAKER_{parts[i+1]}"
                            break
                    
                    if speaker_id:
                        clip_info = {
                            'filename': filename,
                            'path': os.path.join(clips_dir, filename),
                            'speaker': speaker_id,
                            'timestamp': parts[-1] if parts else ''
                        }
                        clips.append(clip_info)
        
        return jsonify(clips)
    
    @app.route('/api/speaker_texts')
    def get_speaker_texts():
        """Get text samples for all speakers from database"""
        speakers = db.get_all_speakers()
        speaker_texts = {}
        
        for speaker in speakers:
            if speaker.get('sample_text'):
                speaker_texts[speaker['id']] = {
                    'name': speaker.get('name', speaker.get('id', 'Unknown')),
                    'text': speaker['sample_text']
                }
        
        return jsonify(speaker_texts)
    
    @app.route('/transcripts')
    def list_transcripts():
        """List all transcripts from JSON files"""
        transcripts = []
        transcript_dir = Config.TRANSCRIPTS_FOLDER
        
        # Add pagination
        page = request.args.get('page', 1, type=int)
        per_page = 50
        
        if os.path.exists(transcript_dir):
            files = [f for f in os.listdir(transcript_dir) if f.endswith('.json')]
            # Sort by modification time
            files.sort(key=lambda x: os.path.getmtime(os.path.join(transcript_dir, x)), reverse=True)
            
            # Paginate
            start = (page - 1) * per_page
            end = start + per_page
            
            for filename in files[start:end]:
                filepath = os.path.join(transcript_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        transcripts.append({
                            'id': data.get('id'),
                            'filename': data.get('filename'),
                            'duration': data.get('duration'),
                            'language': data.get('language'),
                            'speakers': data.get('speakers'),
                            'processed_at': data.get('processed_at'),
                            'preview': data.get('text', '')[:100] + '...'
                        })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        return jsonify({
            'transcripts': transcripts,
            'page': page,
            'per_page': per_page,
            'total': len(files) if 'files' in locals() else 0
        })
    
    @app.route('/transcript/<transcript_id>')
    def get_transcript(transcript_id):
        """Get full transcript details"""
        transcript_path = os.path.join(
            Config.TRANSCRIPTS_FOLDER,
            f"{transcript_id}.json"
        )
        
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                return jsonify(json.load(f))
        
        return jsonify({'error': 'Transcript not found'}), 404
