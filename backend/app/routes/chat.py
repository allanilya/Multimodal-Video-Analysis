"""
Chat routes
"""
import asyncio
import logging
from flask import Blueprint, request, jsonify
from pydantic import BaseModel, ValidationError

from app.services.chat_service import chat_service

logger = logging.getLogger(__name__)

bp = Blueprint('chat', __name__, url_prefix='/api/chat')

class ChatRequest(BaseModel):
    video_id: str
    message: str
    use_visual_context: bool = True

@bp.route('/', methods=['POST'])
def chat():
    """Chat with video content"""
    try:
        # Validate request
        data = request.get_json()
        req = ChatRequest(**data)

        # Create event loop for async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Get response
        result = loop.run_until_complete(
            chat_service.chat_with_video(
                req.video_id,
                req.message,
                req.use_visual_context
            )
        )

        return jsonify(result), 200

    except ValidationError as e:
        return jsonify({'error': 'Invalid request', 'details': e.errors()}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to generate response'}), 500
