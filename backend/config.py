"""
Configuration settings for the application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from backend/)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Config:
    """Base configuration"""
    # Paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    TEMP_FOLDER = BASE_DIR / 'temp'

    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TTL = 86400  # 24 hours

    # Video Processing
    FRAME_SAMPLE_RATE = int(os.getenv('FRAME_SAMPLE_RATE', 1))
    MAX_FRAMES = int(os.getenv('MAX_FRAMES', 100))
    MAX_VIDEO_DURATION = int(os.getenv('MAX_VIDEO_DURATION', 3600))
    VIDEO_QUALITY = 'worst[ext=mp4]/worst'  # Fast downloads

    # AI Models
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')  # Fast & cheap
    # Alternative options:
    # - 'gpt-4o-mini': Best balance (recommended)
    # - 'gpt-3.5-turbo': Cheapest, faster
    # - 'gpt-4o': Best quality, more expensive
    # - 'gpt-4-turbo-preview': Original (expensive)

    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')  # Multimodal with reasoning
    # Alternative options:
    # - 'gemini-2.5-flash': Best for video analysis with reasoning (recommended)
    # - 'gemini-2.5-flash-lite': Fastest & cheapest, limited reasoning
    # - 'gemini-1.5-flash': Older model, less capable

    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    CLIP_MODEL = 'openai/clip-vit-base-patch16'  # Balance between accuracy and speed

    # Processing
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    MAX_WORKERS = 4

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
