"""
Transient-failure retry for outbound HTTP calls.

One shared implementation, because the alternative is three subtly different
ones. ``services.llm`` had the only copy; the search providers had none at all
and returned ``[]`` on any failure, which is indistinguishable from "the web has
nothing on this" — a single blip on Tavily produced a confident, sourceless
answer instead of an error anyone could see.

Two things make this more than a ``for`` loop with a ``sleep``:

**A deadline, not just an attempt count.** Tavily's timeout is 20s. Three
attempts of a hanging provider is a 60s wall, and the research run is already
the slowest thing the product does. The deadline stops the *next* retry once the
budget is spent, so worst case stays bounded by wall clock rather than by
arithmetic on the timeout.

**A per-attempt hook.** Every attempt that reaches a search provider is billed
by that provider, whether or not we get a response back. ``on_attempt`` is how
the caller counts them — see ``services.usage.record_search``. Without it,
retrying is a way to spend a month's credits while the budget guard watches a
third of them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional, TypeVar

import httpx

logger = logging.getLogger(__name__)

__all__ = ["RETRYABLE_STATUS", "is_retryable", "with_retries"]

T = TypeVar("T")

# HTTP statuses worth retrying — transient server/throttling errors only.
# 4xx like 400/401/404 are permanent: retrying a bad API key just spends the
# delay three times and fails anyway.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Transport-level failures that mean "try again", not "this request was wrong".
_RETRYABLE_EXC = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


def is_retryable(exc: Exception) -> bool:
    """Whether an httpx/SDK exception represents a transient, retryable failure."""
    if isinstance(exc, _RETRYABLE_EXC):
        return True
    # SDK errors expose .status_code; some wrap an httpx response instead.
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    return status in RETRYABLE_STATUS


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    label: str,
    attempts: int,
    base_delay: float,
    deadline: Optional[float] = None,
    on_attempt: Optional[Callable[[], None]] = None,
) -> T:
    """
    Run ``operation`` with exponential backoff on transient failures.

    Args:
        operation: Zero-argument async callable. Called once per attempt, so it
            must be safe to repeat — build the request inside it, not around it.
        label: What is being retried, for the log line ("Tavily search '...'").
        attempts: Total tries, including the first. 1 disables retrying.
        base_delay: Seconds before the first retry, doubled each time.
        deadline: Give up rather than start a further attempt once this many
            seconds have elapsed. None = bounded only by ``attempts``.
        on_attempt: Called immediately before each attempt, including retries.
            Used to count billable calls. Never allowed to fail the operation.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        The final attempt's exception. Retrying exhausted is still a failure —
        swallowing it here is what produced the silent empty results.
    """
    started = time.monotonic()
    last_attempt = max(attempts, 1) - 1

    for attempt in range(last_attempt + 1):
        if on_attempt is not None:
            try:
                on_attempt()
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("on_attempt hook failed for %s (ignored): %s", label, e)

        try:
            return await operation()
        except Exception as e:
            if attempt >= last_attempt or not is_retryable(e):
                raise

            delay = base_delay * (2 ** attempt)
            if deadline is not None and (time.monotonic() - started) + delay >= deadline:
                logger.warning(
                    "%s failed on attempt %d/%d (%s); retry budget of %.0fs spent, giving up",
                    label, attempt + 1, last_attempt + 1, e, deadline,
                )
                raise

            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                label, attempt + 1, last_attempt + 1, e, delay,
            )
            await asyncio.sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise AssertionError(f"with_retries fell through for {label}")  # pragma: no cover
