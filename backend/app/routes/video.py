"""
Video processing routes
"""
import asyncio
import logging
from flask import Blueprint, request, jsonify
from pydantic import BaseModel, HttpUrl, ValidationError

from app.services.video_processor import processor
from app.utils.cache import cache
from app import socketio

logger = logging.getLogger(__name__)

bp = Blueprint('video', __name__, url_prefix='/api/video')

class ProcessVideoRequest(BaseModel):
    url: HttpUrl
    force_reprocess: bool = False

@bp.route('/process', methods=['POST'])
def process_video():
    """Process a YouTube video"""
    try:
        # Validate request
        data = request.get_json()
        req = ProcessVideoRequest(**data)

        # Extract video ID
        video_id = processor.youtube_service.extract_video_id(str(req.url))
        cache_key = cache.get_cache_key(video_id)

        # Check cache
        if not req.force_reprocess and cache.exists(cache_key):
            cached_data = cache.get(cache_key)
            if cached_data:
                return jsonify({
                    'video_id': video_id,
                    'status': 'completed',
                    'message': 'Video already processed',
                    'data': cached_data
                }), 200

        # Start background processing
        socketio.start_background_task(
            process_video_background,
            str(req.url),
            video_id
        )

        return jsonify({
            'video_id': video_id,
            'status': 'processing',
            'message': 'Video processing started'
        }), 202

    except ValidationError as e:
        return jsonify({'error': 'Invalid request', 'details': e.errors()}), 400
    except Exception as e:
        logger.error(f"Process video error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def process_video_background(url: str, video_id: str):
    """Background task for video processing"""
    try:
        # Emit progress updates
        socketio.emit('processing_status', {
            'video_id': video_id,
            'status': 'started',
            'message': 'Starting video processing...'
        })

        # Create event loop for async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Process video
        result = loop.run_until_complete(processor.process_video(url))

        # Cache the result
        cache_key = cache.get_cache_key(video_id)
        cache.set(cache_key, result.to_dict())

        # Emit completion
        socketio.emit('processing_status', {
            'video_id': video_id,
            'status': 'completed',
            'message': 'Video processed successfully',
            'data': result.to_dict()
        })

        logger.info(f"Video {video_id} processed successfully")

    except Exception as e:
        logger.error(f"Background processing error: {e}", exc_info=True)

        # Cache error
        error_key = cache.get_cache_key(video_id, 'error')
        cache.set(error_key, {'error': str(e)}, ttl=3600)

        # Emit error
        socketio.emit('processing_status', {
            'video_id': video_id,
            'status': 'error',
            'message': str(e)
        })

@bp.route('/<video_id>', methods=['GET'])
def get_video_info(video_id: str):
    """Get processed video information"""
    try:
        cache_key = cache.get_cache_key(video_id)
        cached_data = cache.get(cache_key)

        if not cached_data:
            # Check for error
            error_key = cache.get_cache_key(video_id, 'error')
            error_data = cache.get(error_key)
            if error_data:
                return jsonify({'error': error_data['error']}), 500

            return jsonify({'error': 'Video not found or still processing'}), 404

        return jsonify(cached_data), 200

    except Exception as e:
        logger.error(f"Get video error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/<video_id>/status', methods=['GET'])
def get_processing_status(video_id: str):
    """Get video processing status"""
    try:
        cache_key = cache.get_cache_key(video_id)
        if cache.exists(cache_key):
            return jsonify({'status': 'completed'}), 200

        error_key = cache.get_cache_key(video_id, 'error')
        if cache.exists(error_key):
            error_data = cache.get(error_key)
            return jsonify({'status': 'error', 'error': error_data.get('error')}), 200

        return jsonify({'status': 'processing'}), 200

    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/<video_id>', methods=['DELETE'])
def delete_video(video_id: str):
    """Delete processed video data"""
    try:
        # Delete from cache
        keys = [
            cache.get_cache_key(video_id),
            cache.get_cache_key(video_id, 'embeddings'),
            cache.get_cache_key(video_id, 'error')
        ]

        for key in keys:
            cache.delete(key)

        # Clean up files
        from config import Config
        video_path = Config.UPLOAD_FOLDER / f"{video_id}.mp4"
        embeddings_path = Config.TEMP_FOLDER / f"{video_id}_embeddings.npz"

        if video_path.exists():
            video_path.unlink()
        if embeddings_path.exists():
            embeddings_path.unlink()

        return jsonify({'message': 'Video data deleted successfully'}), 200

    except Exception as e:
        logger.error(f"Delete video error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
