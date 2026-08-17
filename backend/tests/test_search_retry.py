"""
Retry behaviour for the search providers.

Regression tests for a silent failure mode: both Serper and Tavily used to
return ``[]`` on the first timeout or 5xx, which the pipeline cannot tell apart
from "the web has nothing on this". A single blip produced a confident answer
with no sources and no visible error.

Two properties are being pinned here, and the second is the one that is easy to
get wrong:

1. Transient failures are retried; permanent ones (401, 404) are not, and an
   exhausted budget still degrades to ``[]`` rather than failing the run.
2. **Every attempt is counted as a billable search.** Retries spend real
   provider credits, and the budget guard is the load-bearing cost control on a
   free tier. A retry loop that bills three lookups while the guard counts one
   is a slow leak with no alarm on it.

No sleeping: ``base_delay`` is set to zero everywhere, so these run instantly.
"""

import httpx
import pytest

from app.services import search as search_service
from app.services import tavily as tavily_service
from app.services.retry import is_retryable, with_retries
from app.services.usage import usage_scope


# --- helpers ---------------------------------------------------------------

def _http_error(status: int) -> httpx.HTTPStatusError:
    """An HTTPStatusError carrying a real response, as httpx raises it."""
    request = httpx.Request("POST", "https://example.test/search")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


def _serper_payload() -> dict:
    return {"organic": [{"link": "https://example.test/a", "title": "A", "snippet": "s"}]}


def _tavily_payload() -> dict:
    return {"results": [{"url": "https://example.test/a", "title": "A", "content": "s"}]}


class _Responder:
    """An async callable that replays a scripted sequence of outcomes."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    """Stands in for the module-level httpx.AsyncClient."""

    def __init__(self, responder):
        self.post = responder


# --- is_retryable ----------------------------------------------------------

@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status):
    assert is_retryable(_http_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_statuses_are_not_retryable(status):
    """A bad API key is not a blip. Retrying it just fails three times."""
    assert is_retryable(_http_error(status)) is False


def test_transport_failures_are_retryable():
    assert is_retryable(httpx.TimeoutException("timed out")) is True
    assert is_retryable(httpx.ConnectError("refused")) is True


def test_unrelated_exceptions_are_not_retryable():
    assert is_retryable(ValueError("bad json")) is False


# --- with_retries ----------------------------------------------------------

@pytest.mark.asyncio
async def test_succeeds_without_retrying():
    op = _Responder("ok")
    result = await with_retries(op, label="t", attempts=3, base_delay=0)
    assert result == "ok"
    assert op.calls == 1


@pytest.mark.asyncio
async def test_retries_transient_failure_then_succeeds():
    op = _Responder(_http_error(503), "ok")
    result = await with_retries(op, label="t", attempts=3, base_delay=0)
    assert result == "ok"
    assert op.calls == 2


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried():
    op = _Responder(_http_error(401))
    with pytest.raises(httpx.HTTPStatusError):
        await with_retries(op, label="t", attempts=3, base_delay=0)
    assert op.calls == 1


@pytest.mark.asyncio
async def test_exhausted_attempts_raise_the_last_error():
    """Retrying exhausted is a failure. Swallowing it here is the original bug."""
    op = _Responder(_http_error(503))
    with pytest.raises(httpx.HTTPStatusError):
        await with_retries(op, label="t", attempts=3, base_delay=0)
    assert op.calls == 3


@pytest.mark.asyncio
async def test_attempts_of_one_disables_retrying():
    op = _Responder(_http_error(503))
    with pytest.raises(httpx.HTTPStatusError):
        await with_retries(op, label="t", attempts=1, base_delay=0)
    assert op.calls == 1


@pytest.mark.asyncio
async def test_on_attempt_fires_once_per_attempt_including_retries():
    """The billing hook. One call per provider round-trip, not one per success."""
    op = _Responder(_http_error(503), _http_error(503), "ok")
    seen = []
    await with_retries(
        op, label="t", attempts=3, base_delay=0, on_attempt=lambda: seen.append(1)
    )
    assert op.calls == 3
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_on_attempt_failure_does_not_break_the_operation():
    """Accounting must never be able to fail a search."""
    def boom():
        raise RuntimeError("accounting exploded")

    result = await with_retries(
        _Responder("ok"), label="t", attempts=2, base_delay=0, on_attempt=boom
    )
    assert result == "ok"


@pytest.mark.asyncio
async def test_deadline_stops_a_further_attempt():
    """
    A 20s provider timeout times three attempts is a 60s wall on the slowest
    step in the product. The deadline is what makes the worst case wall-clock
    bounded instead of arithmetic on the timeout.
    """
    op = _Responder(_http_error(503))
    with pytest.raises(httpx.HTTPStatusError):
        await with_retries(
            op, label="t", attempts=5, base_delay=10.0, deadline=1.0
        )
    # The first backoff alone (10s) blows a 1s budget, so no retry is started.
    assert op.calls == 1


# --- search_web (Serper) ---------------------------------------------------

@pytest.mark.asyncio
async def test_serper_retries_and_counts_every_attempt(monkeypatch):
    responder = _Responder(
        _http_error(503),
        httpx.Response(200, json=_serper_payload(), request=httpx.Request("POST", "https://x.test")),
    )
    monkeypatch.setattr(search_service, "_get_client", lambda: _FakeClient(responder))
    monkeypatch.setattr(
        search_service.get_settings(), "search_retry_base_delay", 0, raising=False
    )

    with usage_scope() as usage:
        results = await search_service.search_web("q", max_results=5)

    assert responder.calls == 2
    assert len(results) == 1
    # Two round-trips reached Serper, so two credits were spent — even though
    # only one of them returned anything.
    assert usage.searches == 2


@pytest.mark.asyncio
async def test_serper_returns_empty_after_exhausting_retries(monkeypatch):
    """Still best-effort: one dead sub-query degrades a run, it doesn't fail it."""
    responder = _Responder(_http_error(503))
    monkeypatch.setattr(search_service, "_get_client", lambda: _FakeClient(responder))
    monkeypatch.setattr(
        search_service.get_settings(), "search_retry_base_delay", 0, raising=False
    )

    with usage_scope() as usage:
        results = await search_service.search_web("q", max_results=5)

    assert results == []
    assert responder.calls == usage.searches == 3


# --- tavily_search ---------------------------------------------------------

@pytest.mark.asyncio
async def test_tavily_retries_and_counts_every_attempt(monkeypatch):
    responder = _Responder(
        httpx.TimeoutException("slow"),
        httpx.Response(200, json=_tavily_payload(), request=httpx.Request("POST", "https://x.test")),
    )
    monkeypatch.setattr(tavily_service, "_get_client", lambda: _FakeClient(responder))
    monkeypatch.setattr(
        tavily_service.get_settings(), "search_retry_base_delay", 0, raising=False
    )

    with usage_scope() as usage:
        results = await tavily_service.tavily_search("q", max_results=5, search_depth="basic")

    assert responder.calls == 2
    assert len(results) == 1
    assert usage.searches == 2


@pytest.mark.asyncio
async def test_tavily_bad_key_is_not_retried(monkeypatch):
    """One attempt, one credit — not three of each."""
    responder = _Responder(_http_error(401))
    monkeypatch.setattr(tavily_service, "_get_client", lambda: _FakeClient(responder))

    with usage_scope() as usage:
        results = await tavily_service.tavily_search("q", max_results=5, search_depth="basic")

    assert results == []
    assert responder.calls == usage.searches == 1
