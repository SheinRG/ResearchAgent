"""
Unit tests for the anonymous free-query allowance (app.services.anonymous).

Two counters, doing two different jobs: a small per-visitor cap (the product
decision about when to ask for a signup) and a looser per-IP cap (a backstop
against clearing cookies). Neither is the budget guard — that lives in
services.budget and is what actually protects the free tier.

Also covers search-credit accounting, since the ceiling is only as honest as
the numbers it reads: a cached search is billed nothing and must not be counted.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.services import anonymous
from app.services.usage import (
    UsageAccumulator,
    record_search,
    search_credits_for,
    usage_scope,
)


# --- Fakes ------------------------------------------------------------------


class _FakePipeline:
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

    async def get(self, key):
        return self.store.get(key)


def _request(ip="203.0.113.5", cookie=None):
    """A stand-in for the bits of starlette.Request this module touches."""
    headers = {"x-forwarded-for": ip} if ip else {}
    return SimpleNamespace(
        headers=headers,
        cookies={anonymous.ANON_COOKIE: cookie} if cookie else {},
        client=SimpleNamespace(host="127.0.0.1"),
    )


@pytest.fixture(autouse=True)
def _clear_local_buckets():
    anonymous._local_buckets.clear()
    yield
    anonymous._local_buckets.clear()


@pytest.fixture
def _settings():
    return SimpleNamespace(
        anon_free_queries=3,
        anon_free_queries_per_ip=10,
        environment="development",
    )


# --- The allowance ----------------------------------------------------------


@pytest.mark.asyncio
async def test_peek_does_not_spend_a_query(_settings):
    """
    A visitor who is turned away must not be charged for it. The endpoint peeks
    before doing any work, so peeking has to be genuinely read-only.
    """
    redis = _FakeRedis()
    with patch.object(anonymous, "get_redis", AsyncMock(return_value=redis)), \
         patch.object(anonymous, "get_settings", return_value=_settings):
        first = await anonymous.peek_quota(_request(), "a" * 32)
        second = await anonymous.peek_quota(_request(), "a" * 32)

    assert first.remaining == 3
    assert second.remaining == 3
    assert redis.store == {}


@pytest.mark.asyncio
async def test_allowance_counts_down_and_then_walls(_settings):
    redis = _FakeRedis()
    with patch.object(anonymous, "get_redis", AsyncMock(return_value=redis)), \
         patch.object(anonymous, "get_settings", return_value=_settings):
        for expected_used in (1, 2, 3):
            quota = await anonymous.consume_quota(_request(), "a" * 32)
            assert quota.used == expected_used

        with pytest.raises(HTTPException) as exc:
            await anonymous.consume_quota(_request(), "a" * 32)

    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == "demo_quota_exhausted"


@pytest.mark.asyncio
async def test_new_cookie_does_not_reset_the_ip_allowance(_settings):
    """
    Clearing cookies buys a fresh visitor id but not a fresh allowance — the
    per-IP counter keeps counting. It is deliberately looser (offices share an
    address), so this blunts casual resets rather than preventing them.
    """
    redis = _FakeRedis()
    with patch.object(anonymous, "get_redis", AsyncMock(return_value=redis)), \
         patch.object(anonymous, "get_settings", return_value=_settings):
        # Ten queries across ten "fresh" visitors on one address.
        for i in range(10):
            await anonymous.consume_quota(_request(), f"{i:032d}")

        with pytest.raises(HTTPException):
            await anonymous.consume_quota(_request(), "f" * 32)


@pytest.mark.asyncio
async def test_peek_reports_the_cap_that_will_actually_bite(_settings):
    """
    A visitor on a busy office IP must not be promised three queries and then
    refused on their first.
    """
    redis = _FakeRedis()
    with patch.object(anonymous, "get_redis", AsyncMock(return_value=redis)), \
         patch.object(anonymous, "get_settings", return_value=_settings):
        for i in range(9):
            await anonymous.consume_quota(_request(), f"{i:032d}")

        quota = await anonymous.peek_quota(_request(), "f" * 32)

    # Nine of ten IP queries gone, so one left — not the three this brand new
    # cookie would otherwise be told it has.
    assert quota.remaining == 1


@pytest.mark.asyncio
async def test_redis_down_still_counts(_settings):
    """
    Fails closed, matching the signed-in rate limiter: a Redis outage bounds
    abuse per-instance instead of removing the limit.
    """
    with patch.object(anonymous, "get_redis", AsyncMock(return_value=None)), \
         patch.object(anonymous, "get_settings", return_value=_settings):
        for _ in range(3):
            await anonymous.consume_quota(_request(), "a" * 32)

        with pytest.raises(HTTPException):
            await anonymous.consume_quota(_request(), "a" * 32)


# --- Visitor identity -------------------------------------------------------


def test_cookie_value_is_validated_before_use():
    """The cookie becomes a Redis key suffix, and it is attacker-controlled."""
    assert anonymous.read_anon_id(_request(cookie="a" * 32)) == "a" * 32
    assert anonymous.read_anon_id(_request(cookie="short")) == ""
    assert anonymous.read_anon_id(_request(cookie="../../etc/passwd")) == ""
    assert anonymous.read_anon_id(_request(cookie="x" * 4096)) == ""
    assert anonymous.read_anon_id(_request()) == ""


def test_new_ids_are_distinct_and_well_formed():
    first, second = anonymous.new_anon_id(), anonymous.new_anon_id()

    assert first != second
    assert anonymous.read_anon_id(_request(cookie=first)) == first


def test_client_ip_prefers_the_forwarded_header():
    """Render terminates TLS, so request.client is the proxy, not the visitor."""
    assert anonymous.client_ip(_request(ip="198.51.100.7")) == "198.51.100.7"
    assert (
        anonymous.client_ip(_request(ip="198.51.100.7, 10.0.0.1")) == "198.51.100.7"
    )
    assert anonymous.client_ip(_request(ip=None)) == "127.0.0.1"


# --- Search-credit accounting -----------------------------------------------


def test_credits_are_priced_by_search_depth():
    assert search_credits_for("basic") == 1
    assert search_credits_for("advanced") == 2
    # An unrecognised depth is priced at 1 rather than 0: a guard that counts
    # an unknown search as free is a guard that stops working on a config typo.
    assert search_credits_for("nonsense") == 1


def test_searches_accumulate_within_a_request():
    with usage_scope() as usage:
        record_search("basic")
        record_search("basic")
        record_search("advanced")

    assert usage.searches == 3
    assert usage.search_credits == 4
    assert usage.as_dict()["search_credits"] == 4


def test_recording_outside_a_scope_is_a_no_op():
    """Scripts, tests and the startup health check have nowhere to report to."""
    record_search("basic")  # must not raise


def test_a_run_that_searched_nothing_costs_nothing():
    """
    An answer-cache hit reports real zeros rather than omitting the field —
    that is what makes the cache's saving visible, and what keeps a replay from
    drawing down the ceiling.
    """
    assert UsageAccumulator().as_dict()["search_credits"] == 0
