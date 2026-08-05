"""
Redis async caching service.
Caches search results and scraped content to avoid redundant calls.
Gracefully falls back to no-cache if Redis is unavailable.
"""

import json
import hashlib
import logging
import time
from typing import Optional, Any

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None
# A failed connect puts Redis in a cooldown rather than disabling it for the
# life of the process. This used to be a one-way `_redis_available = False`
# latch: a single blip on a free-tier instance turned the cache off until the
# next restart, silently taking the answer cache, the rate limiter's Redis path
# and the health counters with it.
_redis_retry_at: float = 0.0
_redis_backoff: float = 0.0

_BACKOFF_START = 5.0    # seconds before the first reconnect attempt
_BACKOFF_MAX = 300.0    # cap, so a long outage doesn't reconnect on every request


async def get_redis() -> Optional[aioredis.Redis]:
    """
    Get the Redis client, connecting if needed.

    Returns None while Redis is unreachable, so every caller degrades to
    no-cache — but retries on an exponential backoff instead of giving up
    permanently.
    """
    global _redis_client, _redis_retry_at, _redis_backoff

    if _redis_client is not None:
        return _redis_client

    # Still cooling down from a recent failure.
    if time.monotonic() < _redis_retry_at:
        return None

    try:
        settings = get_settings()
        client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,  # seconds
        )
        await client.ping()
        _redis_client = client
        _redis_backoff = 0.0
        _redis_retry_at = 0.0
        logger.info("Redis connected at %s", settings.redis_url)
        return _redis_client
    except Exception as e:
        _redis_backoff = (
            _BACKOFF_START if _redis_backoff == 0.0
            else min(_redis_backoff * 2, _BACKOFF_MAX)
        )
        _redis_retry_at = time.monotonic() + _redis_backoff
        logger.warning(
            "Redis unavailable, running without cache (retry in %.0fs): %s",
            _redis_backoff, e,
        )
        _redis_client = None
        return None


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
    """Close the Redis connection and clear any reconnect backoff."""
    global _redis_client, _redis_retry_at, _redis_backoff
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
    _redis_retry_at = 0.0
    _redis_backoff = 0.0
    logger.info("Redis connection closed")
