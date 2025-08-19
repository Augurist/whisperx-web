# Add these methods to database.py

def merge_speakers(self, from_speaker_id, to_speaker_id):
    """Merge one speaker into another"""
    try:
        # Update all segments from from_speaker to to_speaker
        self.conn.execute("""
            UPDATE segments 
            SET speaker_id = ? 
            WHERE speaker_id = ?
        """, (to_speaker_id, from_speaker_id))
        
        # Delete the from_speaker
        self.conn.execute("DELETE FROM speakers WHERE id = ?", (from_speaker_id,))
        
        self.conn.commit()
        return True
    except Exception as e:
        print(f"Error merging speakers: {e}")
        self.conn.rollback()
        return False

def delete_speaker(self, speaker_id):
    """Delete a speaker and all associated data"""
    try:
        self.conn.execute("DELETE FROM segments WHERE speaker_id = ?", (speaker_id,))
        self.conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
        self.conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting speaker: {e}")
        self.conn.rollback()
        return False

def cleanup_old_clips(self, days=30):
    """Remove clips older than specified days"""
    import os
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days)
    clips_dir = Config.SPEAKER_CLIPS_FOLDER
    
    if os.path.exists(clips_dir):
        for filename in os.listdir(clips_dir):
            filepath = os.path.join(clips_dir, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff_date:
                os.remove(filepath)
                print(f"Removed old clip: {filename}")
