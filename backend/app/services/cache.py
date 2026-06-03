"""
Redis async caching service.
Caches search results and scraped content to avoid redundant calls.
Gracefully falls back to no-cache if Redis is unavailable.
"""

import json
import hashlib
import logging
from typing import Optional, Any

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None
_redis_available: bool = True


async def get_redis() -> Optional[aioredis.Redis]:
    """Get the Redis client, initializing if needed."""
    global _redis_client, _redis_available

    if not _redis_available:
        return None

    if _redis_client is None:
        try:
            settings = get_settings()
            _redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await _redis_client.ping()
            logger.info("Redis connected at %s", settings.redis_url)
        except Exception as e:
            logger.warning("Redis unavailable, running without cache: %s", e)
            _redis_available = False
            _redis_client = None
            return None

    return _redis_client


def _cache_key(prefix: str, data: str) -> str:
    """Generate a deterministic cache key."""
    h = hashlib.md5(data.encode()).hexdigest()
    return f"research:{prefix}:{h}"


async def cache_get(prefix: str, key_data: str) -> Optional[Any]:
    """
    Get a cached value.

    Args:
        prefix: Cache namespace (e.g., 'search', 'scrape').
        key_data: Data to hash for the cache key.

    Returns:
        Cached value (parsed from JSON) or None.
    """
    client = await get_redis()
    if client is None:
        return None

    try:
        key = _cache_key(prefix, key_data)
        value = await client.get(key)
        if value:
            logger.debug("Cache hit: %s", key)
            return json.loads(value)
        return None
    except Exception as e:
        logger.warning("Cache get error: %s", e)
        return None


async def cache_set(prefix: str, key_data: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Store a value in cache.

    Args:
        prefix: Cache namespace.
        key_data: Data to hash for the cache key.
        value: Value to cache (must be JSON-serializable).
        ttl: Time-to-live in seconds (default from settings).
    """
    client = await get_redis()
    if client is None:
        return

    try:
        settings = get_settings()
        key = _cache_key(prefix, key_data)
        serialized = json.dumps(value, default=str)
        await client.set(key, serialized, ex=ttl or settings.cache_ttl)
        logger.debug("Cache set: %s (ttl=%ds)", key, ttl or settings.cache_ttl)
    except Exception as e:
        logger.warning("Cache set error: %s", e)


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis_client, _redis_available
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
    _redis_available = True
    logger.info("Redis connection closed")
