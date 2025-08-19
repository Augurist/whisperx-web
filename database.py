"""
Database operations module
~180 lines
"""
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class Database:
    """Database operations for speakers and transcripts"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create speakers table
        c.execute('''CREATE TABLE IF NOT EXISTS speakers
                     (id TEXT PRIMARY KEY,
                      name TEXT,
                      embedding BLOB,
                      clip_path TEXT,
                      transcript_sample TEXT,
                      created_at TIMESTAMP,
                      appearance_count INTEGER DEFAULT 1)''')
        
        # Create transcripts table
        c.execute('''CREATE TABLE IF NOT EXISTS transcripts
                     (id TEXT PRIMARY KEY,
                      filename TEXT,
                      duration REAL,
                      language TEXT,
                      full_text TEXT,
                      segments TEXT,
                      speakers TEXT,
                      created_at TIMESTAMP)''')
        
        # Create speaker_appearances table
        c.execute('''CREATE TABLE IF NOT EXISTS speaker_appearances
                     (transcript_id TEXT,
                      speaker_id TEXT,
                      segments_count INTEGER,
                      FOREIGN KEY(transcript_id) REFERENCES transcripts(id),
                      FOREIGN KEY(speaker_id) REFERENCES speakers(id))''')
        
        # Create FTS5 virtual table for full-text search
        try:
            c.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts 
                        USING fts5(id, filename, full_text, tokenize='porter')''')
        except sqlite3.OperationalError:
            pass  # Table already exists
        
        conn.commit()
        conn.close()
    
    def save_speaker(self, speaker_id: str, clip_path: str = None, 
                    transcript_sample: str = None, embedding: np.ndarray = None) -> bool:
        """Save or update speaker in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            # Check if speaker exists
            c.execute("SELECT id FROM speakers WHERE id = ?", (speaker_id,))
            exists = c.fetchone()
            
            if exists:
                c.execute("""UPDATE speakers 
                           SET appearance_count = appearance_count + 1
                           WHERE id = ?""", (speaker_id,))
            else:
                embedding_blob = embedding.tobytes() if embedding is not None else None
                c.execute("""INSERT INTO speakers (id, name, embedding, clip_path, 
                           transcript_sample, created_at, appearance_count)
                           VALUES (?, ?, ?, ?, ?, ?, 1)""",
                        (speaker_id, speaker_id, embedding_blob, clip_path, 
                         transcript_sample, datetime.now()))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving speaker: {e}")
            return False
        finally:
            conn.close()
    
    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        """Get speaker details"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT id, name, clip_path, transcript_sample, 
                           created_at, appearance_count, embedding
                    FROM speakers WHERE id = ?""", (speaker_id,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            'id': row[0],
            'name': row[1],
            'clip_path': row[2],
            'sample_text': row[3],
            'created_at': row[4],
            'appearance_count': row[5],
            'has_embedding': row[6] is not None
        }
    
    def get_all_speakers(self) -> List[Dict]:
        """Get all speakers"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT id, name, clip_path, transcript_sample, 
                           created_at, appearance_count
                    FROM speakers
                    ORDER BY appearance_count DESC""")
        
        speakers = []
        for row in c.fetchall():
            speakers.append({
                'id': row[0],
                'name': row[1],
                'has_clip': bool(row[2]),
                'sample_text': row[3],
                'created_at': row[4],
                'appearance_count': row[5]
            })
        
        conn.close()
        return speakers
    
    def get_speakers_with_embeddings(self) -> List[Tuple[str, str, np.ndarray]]:
        """Get speakers with real names and embeddings"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT id, name, embedding 
                    FROM speakers 
                    WHERE embedding IS NOT NULL 
                    AND name NOT LIKE 'SPEAKER_%'
                    ORDER BY appearance_count DESC""")
        
        speakers = []
        for row in c.fetchall():
            if row[2]:
                try:
                    embedding = np.frombuffer(row[2], dtype=np.float32)
                    speakers.append((row[0], row[1], embedding))
                except Exception as e:
                    print(f"Error loading embedding for {row[1]}: {e}")
        
        conn.close()
        return speakers
    
    def update_speaker_name(self, speaker_id: str, new_name: str) -> bool:
        """Update speaker name"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute("SELECT id FROM speakers WHERE id = ?", (speaker_id,))
            exists = c.fetchone()
            
            if exists:
                c.execute("UPDATE speakers SET name = ? WHERE id = ?", 
                         (new_name, speaker_id))
            else:
                c.execute("""INSERT INTO speakers (id, name, created_at, appearance_count)
                           VALUES (?, ?, ?, 1)""",
                         (speaker_id, new_name, datetime.now()))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating speaker: {e}")
            return False
        finally:
            conn.close()
