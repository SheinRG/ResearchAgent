"""
Langfuse tracing — per-stage latency and token cost for the research pipeline.

Everything here is optional and fails soft. With no Langfuse keys configured the
module is a set of no-ops: the context managers still work, they just yield a
null observation that discards whatever you tell it. That is the same contract
``services.cache`` uses for Redis and ``main._init_sentry`` uses for Sentry — a
deployment without the vendor behaves exactly as it did before.

Three rules this module holds to, because tracing sits on the request path:

- **It can never fail a request.** Every call into the SDK is wrapped. A broken
  exporter, an expired key, or an SDK version whose signature moved produces a
  log line, not a 500.
- **It never blocks token delivery.** The SDK batches and exports on a
  background thread; nothing here awaits the network.
- **It sends no more content than it must.** Payloads are truncated, and
  ``langfuse_capture_content=false`` reduces traces to timings, token counts and
  counts-of-things, with no prompts or answers at all.

The langfuse import is deliberately lazy — inside :func:`init_tracing` rather
than at module scope — so an install without the package still boots.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from app.config import get_settings
from app.services.usage import price_for

logger = logging.getLogger(__name__)

# The live client, or None when tracing is disabled/unavailable. Everything in
# this module treats None as "do nothing".
_client: Any = None

# Resolved once at init so the hot path never re-reads settings.
_capture_content: bool = True
_max_content_chars: int = 2000

# propagate_attributes moved around across SDK majors; resolved at init and left
# as None when this version does not expose it.
_propagate_attributes: Any = None


# ---------------------------------------------------------------------------
# Content shaping
# ---------------------------------------------------------------------------

def preview(value: Any) -> Any:
    """
    Shape a value for sending to Langfuse.

    Returns None when content capture is off, so callers can pass prompts and
    answers unconditionally and let configuration decide. Strings are truncated:
    a synthesizer prompt carries the full text of every scraped source, which is
    both expensive to ship and useless to read in a trace viewer.
    """
    if not _capture_content:
        return None
    if isinstance(value, str) and len(value) > _max_content_chars:
        return value[:_max_content_chars] + f"… [+{len(value) - _max_content_chars} chars]"
    return value


def usage_details(usage: Any) -> Optional[dict]:
    """Map a :class:`~app.services.usage.TokenUsage` to Langfuse's usage shape."""
    if usage is None:
        return None
    return {
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "total": usage.total_tokens,
    }


def cost_details(usage: Any) -> Optional[dict]:
    """
    Cost for a call, in Langfuse's cost shape.

    Sent explicitly rather than left to Langfuse to derive, because its built-in
    price table does not cover Groq's model catalogue — without this every trace
    would report $0 and the cost dashboard would be decorative. The numbers come
    from the same table that feeds the figure shown in the UI, so the two agree
    by construction.

    Returns None for a model we have no price for, so an unpriced model shows as
    "unknown cost" rather than a confident zero.
    """
    if usage is None:
        return None

    prompt_price, completion_price = price_for(usage.model)
    if not (prompt_price or completion_price):
        return None

    input_cost = usage.prompt_tokens * prompt_price / 1_000_000
    output_cost = usage.completion_tokens * completion_price / 1_000_000
    return {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost,
    }


# ---------------------------------------------------------------------------
# Observation handles
# ---------------------------------------------------------------------------

class _NullObservation:
    """Stands in for a span when tracing is off. Swallows everything."""

    def update(self, **_kwargs: Any) -> None:
        return

    def score(self, **_kwargs: Any) -> None:
        return


class _Observation:
    """Thin wrapper that keeps SDK failures away from the caller."""

    __slots__ = ("_raw",)

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def update(self, **kwargs: Any) -> None:
        """Attach output/usage/level to the observation. Never raises."""
        try:
            self._raw.update(**kwargs)
        except Exception as e:
            logger.debug("Tracing update failed (ignored): %s", e)


NULL_OBSERVATION = _NullObservation()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Whether traces are actually being recorded."""
    return _client is not None


def init_tracing() -> None:
    """
    Start the Langfuse client when both keys are configured.

    A no-op otherwise, and a no-op if anything at all goes wrong — a wrong key
    or a missing package must not stop the app from serving research.
    """
    global _client, _capture_content, _max_content_chars, _propagate_attributes

    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.info("Langfuse tracing disabled (no keys configured)")
        return

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            sample_rate=settings.langfuse_sample_rate,
            # Keeps local experiments out of the production dashboard.
            environment=settings.environment,
        )

        _capture_content = settings.langfuse_capture_content
        _max_content_chars = settings.langfuse_max_content_chars
        _client = client

        try:
            from langfuse import propagate_attributes

            _propagate_attributes = propagate_attributes
        except ImportError:
            # Older/newer SDK without this helper: user and session still ride
            # along in span metadata, so nothing is lost but the grouping.
            _propagate_attributes = None
            logger.debug("langfuse.propagate_attributes unavailable in this SDK version")

        logger.info(
            "Langfuse tracing enabled (host=%s, sample_rate=%.2f, content=%s)",
            settings.langfuse_host,
            settings.langfuse_sample_rate,
            "on" if _capture_content else "off",
        )
    except Exception as e:
        _client = None
        logger.warning("Langfuse init failed, continuing without tracing: %s", e)


def shutdown_tracing() -> None:
    """Flush buffered observations on shutdown. Never raises."""
    global _client
    if _client is None:
        return
    try:
        _client.shutdown()
        logger.info("Langfuse tracing flushed and shut down")
    except Exception as e:
        logger.warning("Langfuse shutdown failed (ignored): %s", e)
    finally:
        _client = None


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------

def _start(as_type: str, name: str, **kwargs: Any) -> tuple[Any, Any]:
    """
    Open an observation, returning (context_manager, observation).

    Returns (None, NULL_OBSERVATION) when tracing is off or the SDK refuses —
    callers treat that as "carry on without a trace".
    """
    client = _client
    if client is None:
        return None, NULL_OBSERVATION

    try:
        cm = client.start_as_current_observation(as_type=as_type, name=name, **kwargs)
        return cm, _Observation(cm.__enter__())
    except Exception as e:
        logger.debug("Could not start %s observation '%s' (ignored): %s", as_type, name, e)
        return None, NULL_OBSERVATION


def _finish(cm: Any, exc: Optional[BaseException]) -> None:
    """Close an observation, swallowing exporter failures."""
    if cm is None:
        return
    try:
        if exc is None:
            cm.__exit__(None, None, None)
        else:
            cm.__exit__(type(exc), exc, exc.__traceback__)
    except Exception as e:
        logger.debug("Closing observation failed (ignored): %s", e)


@contextmanager
def span(name: str, **kwargs: Any) -> Iterator[Any]:
    """
    A traced unit of work.

    An exception raised inside the block marks the span as errored and then
    propagates unchanged — tracing observes, it never swallows.
    """
    cm, observation = _start("span", name, **kwargs)
    try:
        yield observation
    except Exception as e:
        observation.update(level="ERROR", status_message=str(e)[:500])
        _finish(cm, e)
        raise
    else:
        _finish(cm, None)


@contextmanager
def generation(name: str, *, model: str, **kwargs: Any) -> Iterator[Any]:
    """
    A traced LLM call. Same contract as :func:`span`, plus a model name.

    Report tokens on the way out with
    ``observation.update(output=..., usage_details=tracing.usage_details(usage))``.
    """
    cm, observation = _start("generation", name, model=model, **kwargs)
    try:
        yield observation
    except Exception as e:
        observation.update(level="ERROR", status_message=str(e)[:500])
        _finish(cm, e)
        raise
    else:
        _finish(cm, None)


@contextmanager
def request_scope(
    *,
    user_id: str = "",
    session_id: str = "",
    tags: Optional[list[str]] = None,
) -> Iterator[None]:
    """
    Attach user/session/tags to every observation created inside the block.

    ``user_id`` is the internal account UUID, never an email — traces carry the
    user's questions, and there is no reason to also hand over their identity.
    """
    if _client is None or _propagate_attributes is None:
        yield
        return

    try:
        cm = _propagate_attributes(
            user_id=user_id or None,
            session_id=session_id or None,
            tags=tags or None,
        )
        cm.__enter__()
    except Exception as e:
        logger.debug("propagate_attributes failed (ignored): %s", e)
        yield
        return

    try:
        yield
    except Exception as e:
        _finish(cm, e)
        raise
    else:
        _finish(cm, None)
