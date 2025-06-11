"""
Caching utilities using Redis with fallback to in-memory
"""
import json
import logging
from typing import Optional, Dict, Any
import redis
from config import Config

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching with Redis and in-memory fallback"""

    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                Config.REDIS_URL,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        # Fallback to memory
        return self.memory_cache.get(key)

    def set(self, key: str, value: Any, ttl: int = None):
        """Set data in cache with optional TTL"""
        ttl = ttl or Config.CACHE_TTL
        serialized = json.dumps(value)

        # Try Redis
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.error(f"Redis set error: {e}")

        # Always store in memory as backup
        self.memory_cache[key] = value

    def delete(self, key: str):
        """Delete data from cache"""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")

        self.memory_cache.pop(key, None)

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if self.redis_client:
            try:
                return self.redis_client.exists(key) > 0
            except Exception as e:
                logger.error(f"Redis exists error: {e}")

        return key in self.memory_cache

    def get_cache_key(self, video_id: str, suffix: str = "") -> str:
        """Generate cache key for video data"""
        return f"video:{video_id}:{suffix}" if suffix else f"video:{video_id}"

# Global cache instance
cache = CacheManager()
