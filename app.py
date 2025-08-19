"""
WhisperX Web - Main Application
Clean entry point that imports all modules
~50 lines
"""
from flask import Flask
from config import Config
from database import Database
from routes import register_routes

def create_app():
    """Create and configure Flask application"""
    
    # Check GPU availability
    Config.check_gpu()
    
    # Ensure all directories exist
    Config.ensure_directories()
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure Flask
    app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    
    # Initialize database
    db = Database(Config.DATABASE_PATH)
    
    # Register routes
    register_routes(app, db)
    
    # Print startup info
    if not Config.HF_TOKEN:
        print("⚠️  WARNING: No HF_TOKEN environment variable found.")
        print("   Speaker diarization will be disabled.")
    else:
        print("✅ Hugging Face token found. Speaker diarization enabled!")
    
    print("=" * 60)
    print("✅ Server starting (models will load on first transcription)")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)
