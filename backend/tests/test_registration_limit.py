"""
Unit tests for the per-IP registration limit (auth.check_registration_limit).

The hole this closes is denial of service, not budget bypass: the spend ceilings
in ``services.budget`` are global, so minting accounts cannot get past them by
construction. What an unlimited /register *could* do is burn the shared daily
allowance in a loop and drop every legitimate visitor into degraded mode.

The properties worth pinning:
  1. Redis-down must not remove the limit — same fail-closed contract as the
     per-user query limiter it sits beside.
  2. Attempts are counted, not successes: a script hammering addresses that
     already exist never creates a row, so counting rows would count nothing.
  3. Counters are per-IP, so one abuser cannot lock everyone else out.
  4. The day-long signup window and the hour-long query window share one
     fallback dict without expiring each other.

All external services are faked — no live Redis.
"""

import time

import pytest
from unittest.mock import AsyncMock, patch

import app.routers.auth as auth
from app.config import get_settings


# --- Fakes -----------------------------------------------------------------


class _FakePipeline:
    """Mimics redis.asyncio pipeline: sync queue methods + async execute()."""

    def __init__(self, store):
        self._store = store
        self._key = None

    def incr(self, key):
        self._key = key
        return self

    def expire(self, key, ttl, nx=False):
        return self

    async def execute(self):
        self._store[self._key] = self._store.get(self._key, 0) + 1
        return [self._store[self._key], True]


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self):
        return _FakePipeline(self.store)


class _BrokenRedis:
    def pipeline(self):
        raise RuntimeError("connection reset")


@pytest.fixture(autouse=True)
def _clear_local_buckets():
    auth._local_buckets.clear()
    yield
    auth._local_buckets.clear()


def _limit() -> int:
    return get_settings().registrations_per_ip_per_day


# --- The limit is enforced -------------------------------------------------


@pytest.mark.asyncio
async def test_allows_up_to_the_limit_then_blocks():
    limit = _limit()
    redis = _FakeRedis()
    with patch.object(auth, "get_redis", new=AsyncMock(return_value=redis)):
        for _ in range(limit):
            await auth.check_registration_limit("1.2.3.4")

        with pytest.raises(auth.HTTPException) as exc:
            await auth.check_registration_limit("1.2.3.4")

    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_counters_are_per_ip():
    """One abuser must not lock out everyone else."""
    limit = _limit()
    redis = _FakeRedis()
    with patch.object(auth, "get_redis", new=AsyncMock(return_value=redis)):
        for _ in range(limit + 1):
            try:
                await auth.check_registration_limit("1.2.3.4")
            except auth.HTTPException:
                pass

        # A different address is unaffected.
        await auth.check_registration_limit("5.6.7.8")


@pytest.mark.asyncio
async def test_message_does_not_leak_whether_an_email_exists():
    """
    Registration returns a deliberately vague 409 on a duplicate address. The
    429 must not become the side channel that 409 avoids being.
    """
    limit = _limit()
    redis = _FakeRedis()
    with patch.object(auth, "get_redis", new=AsyncMock(return_value=redis)):
        for _ in range(limit):
            await auth.check_registration_limit("1.2.3.4")
        with pytest.raises(auth.HTTPException) as exc:
            await auth.check_registration_limit("1.2.3.4")

    detail = str(exc.value.detail).lower()
    assert "email" not in detail and "account exists" not in detail


# --- Fail closed -----------------------------------------------------------


@pytest.mark.asyncio
async def test_redis_down_still_enforces_the_limit():
    limit = _limit()
    with patch.object(auth, "get_redis", new=AsyncMock(return_value=None)):
        for _ in range(limit):
            await auth.check_registration_limit("1.2.3.4")
        with pytest.raises(auth.HTTPException):
            await auth.check_registration_limit("1.2.3.4")


@pytest.mark.asyncio
async def test_redis_error_mid_flight_falls_back_to_local_limit():
    limit = _limit()
    with patch.object(auth, "get_redis", new=AsyncMock(return_value=_BrokenRedis())):
        for _ in range(limit):
            await auth.check_registration_limit("1.2.3.4")
        with pytest.raises(auth.HTTPException):
            await auth.check_registration_limit("1.2.3.4")


# --- Configuration ---------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_disables_the_limit():
    # Patched on the instance, not the class: pydantic v2 fields are not class
    # attributes. get_settings() is lru_cached, so this is the object the
    # limiter reads.
    settings = get_settings()
    with patch.object(auth, "get_redis", new=AsyncMock(return_value=None)), \
         patch.object(settings, "registrations_per_ip_per_day", 0):
        for _ in range(50):
            await auth.check_registration_limit("1.2.3.4")


# --- The two windows share one fallback dict -------------------------------


def test_signup_bucket_is_not_pruned_by_the_hourly_window():
    """
    Both limiters use _local_buckets with different windows. Pruning against a
    single global window would expire a day-long signup bucket after an hour and
    silently reset the count the limit depends on.
    """
    auth._hit_local_window("signup:ip:1.2.3.4", auth._REGISTRATION_WINDOW_SECONDS)

    # Age the bucket past the hourly window, but not past the daily one.
    start, count, window = auth._local_buckets["signup:ip:1.2.3.4"]
    auth._local_buckets["signup:ip:1.2.3.4"] = (
        start - auth._RATE_WINDOW_SECONDS - 60, count, window,
    )

    # Force a prune by exceeding the size threshold with cheap filler entries.
    now = time.time()
    for i in range(10_001):
        auth._local_buckets[f"filler:{i}"] = (now, 1, auth._RATE_WINDOW_SECONDS)

    auth._hit_local_window("ratelimit:someone", auth._RATE_WINDOW_SECONDS)

    assert auth._local_buckets["signup:ip:1.2.3.4"][1] == count


def test_expired_bucket_rolls_the_window_over():
    key = "signup:ip:9.9.9.9"
    auth._hit_local_window(key, auth._REGISTRATION_WINDOW_SECONDS)
    start, count, window = auth._local_buckets[key]
    auth._local_buckets[key] = (start - window - 1, count, window)

    assert auth._hit_local_window(key, auth._REGISTRATION_WINDOW_SECONDS) == 1
