"""
Flask Application Factory
"""
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Initialize SocketIO with proper CORS
socketio = SocketIO(
    cors_allowed_origins="*",
    cors_credentials=True,
    async_mode='eventlet'
)

def create_app(config_name='development'):
    """Create and configure the Flask application"""
    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['REDIS_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    # Enable CORS for all origins (development mode)
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Initialize SocketIO with app and CORS settings
    socketio.init_app(
        app,
        cors_allowed_origins="*",
        cors_credentials=True,
        async_mode='eventlet',
        logger=True,
        engineio_logger=True
    )

    # Register blueprints
    from app.routes import video, chat, search
    app.register_blueprint(video.bp)
    app.register_blueprint(chat.bp)
    app.register_blueprint(search.bp)

    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'multimodal-video-analysis'}

    return app
