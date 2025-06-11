"""
Search routes
"""
import asyncio
import logging
from flask import Blueprint, request, jsonify
from pydantic import BaseModel, ValidationError

from app.services.search_service import search_service

logger = logging.getLogger(__name__)

bp = Blueprint('search', __name__, url_prefix='/api/search')

class SearchRequest(BaseModel):
    video_id: str
    query: str
    search_type: str = 'both'  # 'visual', 'text', or 'both'

@bp.route('/', methods=['POST'])
def search():
    """Search video content"""
    try:
        # Validate request
        data = request.get_json()
        req = SearchRequest(**data)

        # Validate search type
        if req.search_type not in ['visual', 'text', 'both']:
            return jsonify({'error': 'Invalid search_type'}), 400

        # Create event loop for async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Perform search
        result = loop.run_until_complete(
            search_service.search(
                req.video_id,
                req.query,
                req.search_type
            )
        )

        return jsonify(result), 200

    except ValidationError as e:
        return jsonify({'error': 'Invalid request', 'details': e.errors()}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({'error': 'Search failed'}), 500
