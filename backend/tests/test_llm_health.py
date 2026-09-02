"""
Tests for GroqClient.health_check (app/services/llm.py).

These exist because of a specific outage, twice. On 2026-08-22 (`72f8650`)
Groq decommissioned both Llama models; every call 404'd, every layer degraded
politely, and the demo answered nothing while reporting itself healthy. On
2026-09-01 a Render instance with a stale `GROQ_MODEL` env var did it again,
and the startup log still read "LLM client is healthy (model:
llama-3.1-8b-instant)" on the line immediately after warning that the model
did not exist.

The cause both times was that health_check asked "is Groq up?" and not "can I
serve a request?". It logged the missing model and then set the cache to True
regardless. There was no test on this method at all.

So: reachability alone is not health, and a missing model must not be
survivable by the health check even though it is survivable by the service.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.llm import GroqClient


def models(*ids):
    """Shape of `client.models.list()` — an object with a `.data` list."""
    return SimpleNamespace(data=[SimpleNamespace(id=i) for i in ids])


@pytest.fixture
def client(monkeypatch):
    c = GroqClient.__new__(GroqClient)  # skip __init__: no API key needed here
    c.model = "openai/gpt-oss-20b"
    c._health_cache = None
    c._health_cache_at = 0.0
    c.client = SimpleNamespace(models=SimpleNamespace(list=AsyncMock()))
    return c


@pytest.mark.asyncio
async def test_healthy_when_the_configured_model_is_served(client):
    client.client.models.list.return_value = models("openai/gpt-oss-20b", "whisper-large-v3")
    assert await client.health_check() is True


@pytest.mark.asyncio
async def test_unhealthy_when_the_model_is_missing(client):
    """
    The regression. Groq answers 200, the model is gone, every call will 404.

    Before the fix this returned True and the startup banner said "healthy".
    """
    client.client.models.list.return_value = models("openai/gpt-oss-20b", "groq/compound")
    client.model = "llama-3.1-8b-instant"
    assert await client.health_check() is False


@pytest.mark.asyncio
async def test_unhealthy_when_the_api_is_unreachable(client):
    client.client.models.list.side_effect = RuntimeError("connection refused")
    assert await client.health_check() is False


@pytest.mark.asyncio
async def test_missing_model_is_logged_as_an_error_with_what_is_available(client, caplog):
    """
    A warning is what this was, and a warning is skimmable. The operator needs
    the available list to fix it, so it goes in the message.
    """
    client.client.models.list.return_value = models("openai/gpt-oss-20b")
    client.model = "llama-3.1-8b-instant"
    with caplog.at_level("ERROR"):
        await client.health_check()
    assert "llama-3.1-8b-instant" in caplog.text
    assert "openai/gpt-oss-20b" in caplog.text


@pytest.mark.asyncio
async def test_result_is_cached_for_ttl(client):
    """Repeated polls must not burn Groq rate limit on the models endpoint."""
    client.client.models.list.return_value = models("openai/gpt-oss-20b")
    assert await client.health_check() is True
    assert await client.health_check() is True
    assert client.client.models.list.await_count == 1


@pytest.mark.asyncio
async def test_cache_expires(client):
    client.client.models.list.return_value = models("openai/gpt-oss-20b")
    assert await client.health_check(ttl=0) is True
    assert await client.health_check(ttl=0) is True
    assert client.client.models.list.await_count == 2


@pytest.mark.asyncio
async def test_a_recovered_model_flips_the_cached_answer_back(client):
    """
    The cache must not pin an unhealthy verdict past the fix.

    Changing GROQ_MODEL and restarting is the documented remedy; if a stale
    False survived, the operator would think the fix had not worked.
    """
    client.client.models.list.return_value = models("openai/gpt-oss-20b")
    client.model = "llama-3.1-8b-instant"
    assert await client.health_check(ttl=0) is False

    client.model = "openai/gpt-oss-20b"
    assert await client.health_check(ttl=0) is True
