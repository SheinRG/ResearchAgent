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
                socket_connect_timeout=5,  # seconds
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


async def cache_get_many(prefix: str, key_datas: list[str]) -> dict[str, Any]:
    """
    Batch-get multiple cached values in a single round trip via MGET.

    Args:
        prefix: Cache namespace (e.g., 'scrape').
        key_datas: List of raw key strings to look up.

    Returns:
        Dict mapping each input key_data to its cached value (only hits included).
    """
    if not key_datas:
        return {}

    client = await get_redis()
    if client is None:
        return {}

    try:
        keys = [_cache_key(prefix, k) for k in key_datas]
        values = await client.mget(keys)
        hits: dict[str, Any] = {}
        for key_data, value in zip(key_datas, values):
            if value:
                try:
                    hits[key_data] = json.loads(value)
                except json.JSONDecodeError:
                    continue
        return hits
    except Exception as e:
        logger.warning("Cache mget error: %s", e)
        return {}


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


async def hash_set(key: str, field: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Store one JSON-serializable field in a Redis hash, best-effort.

    Used by the semantic cache's vector index, where one hash holds every
    embedding for a bucket and the field name is the answer's cache key.
    """
    client = await get_redis()
    if client is None:
        return

    try:
        await client.hset(key, field, json.dumps(value, default=str))
        if ttl:
            # Refreshed on every write, so an index in active use never expires
            # while an abandoned one is reclaimed instead of leaking.
            await client.expire(key, ttl)
    except Exception as e:
        logger.warning("Hash set error for %s: %s", key, e)


async def hash_get_all(key: str) -> dict[str, Any]:
    """Read a whole hash, dropping any field whose JSON no longer parses."""
    client = await get_redis()
    if client is None:
        return {}

    try:
        raw = await client.hgetall(key)
        result: dict[str, Any] = {}
        for field, value in (raw or {}).items():
            try:
                result[field] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
        return result
    except Exception as e:
        logger.warning("Hash read error for %s: %s", key, e)
        return {}


async def hash_delete(key: str, fields: list[str]) -> None:
    """Remove fields from a hash, best-effort."""
    if not fields:
        return

    client = await get_redis()
    if client is None:
        return

    try:
        await client.hdel(key, *fields)
    except Exception as e:
        logger.warning("Hash delete error for %s: %s", key, e)


def _counter_key(name: str) -> str:
    """Key for a monotonic counter. Not hashed — these are read by humans."""
    return f"research:counter:{name}"


async def counter_incr(name: str) -> None:
    """
    Increment a named counter by one, best-effort.

    Deliberately has no TTL: these are lifetime process-independent tallies
    (e.g. answer-cache hits vs misses) whose whole value is the ratio between
    them over time. Silently does nothing when Redis is unavailable, exactly
    like the rest of this module.
    """
    client = await get_redis()
    if client is None:
        return

    try:
        await client.incr(_counter_key(name))
    except Exception as e:
        logger.warning("Counter incr error for %s: %s", name, e)


async def counters_get(names: list[str]) -> dict[str, int]:
    """
    Read several counters in one round trip. Missing counters read as 0.

    Returns an all-zero mapping when Redis is down, so callers can render the
    numbers unconditionally.
    """
    if not names:
        return {}

    client = await get_redis()
    if client is None:
        return {name: 0 for name in names}

    try:
        values = await client.mget([_counter_key(n) for n in names])
        result: dict[str, int] = {}
        for name, value in zip(names, values):
            try:
                result[name] = int(value) if value is not None else 0
            except (TypeError, ValueError):
                result[name] = 0
        return result
    except Exception as e:
        logger.warning("Counter read error: %s", e)
        return {name: 0 for name in names}


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis_client, _redis_available
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
    _redis_available = True
    logger.info("Redis connection closed")
