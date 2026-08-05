"""
Redis reconnect behaviour.

Regression tests for a one-way `_redis_available = False` latch: a single failed
connect used to disable the cache for the life of the process, silently taking
the answer cache, the Redis rate-limit path and the health counters with it
until the next restart.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import cache


@pytest.fixture(autouse=True)
def reset_cache_state():
    """Each test starts with a fresh, unconnected module state."""
    cache._redis_client = None
    cache._redis_retry_at = 0.0
    cache._redis_backoff = 0.0
    yield
    cache._redis_client = None
    cache._redis_retry_at = 0.0
    cache._redis_backoff = 0.0


def _working_client():
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    return client


@pytest.mark.asyncio
async def test_failed_connect_returns_none_and_does_not_raise():
    with patch.object(cache.aioredis, "from_url", side_effect=OSError("refused")):
        assert await cache.get_redis() is None


@pytest.mark.asyncio
async def test_second_call_within_cooldown_does_not_reconnect():
    """The backoff exists so a dead Redis isn't dialled on every request."""
    with patch.object(
        cache.aioredis, "from_url", side_effect=OSError("refused")
    ) as from_url:
        assert await cache.get_redis() is None
        assert await cache.get_redis() is None
        assert from_url.call_count == 1


@pytest.mark.asyncio
async def test_reconnects_once_the_cooldown_expires():
    """The actual regression: a failure must not be permanent."""
    with patch.object(cache.aioredis, "from_url", side_effect=OSError("refused")):
        assert await cache.get_redis() is None

    assert cache._redis_backoff == cache._BACKOFF_START

    # Jump past the cooldown rather than sleeping through it.
    with patch.object(cache.time, "monotonic", return_value=cache._redis_retry_at + 1):
        client = _working_client()
        with patch.object(cache.aioredis, "from_url", return_value=client):
            assert await cache.get_redis() is client

    # A successful connect clears the backoff.
    assert cache._redis_backoff == 0.0
    assert cache._redis_retry_at == 0.0


@pytest.mark.asyncio
async def test_backoff_doubles_and_is_capped():
    now = 1000.0
    with patch.object(cache.aioredis, "from_url", side_effect=OSError("refused")):
        for _ in range(20):
            with patch.object(cache.time, "monotonic", return_value=now):
                assert await cache.get_redis() is None
            now = cache._redis_retry_at + 1

    assert cache._redis_backoff == cache._BACKOFF_MAX


@pytest.mark.asyncio
async def test_existing_client_is_reused():
    client = _working_client()
    with patch.object(cache.aioredis, "from_url", return_value=client) as from_url:
        assert await cache.get_redis() is client
        assert await cache.get_redis() is client
        assert from_url.call_count == 1
