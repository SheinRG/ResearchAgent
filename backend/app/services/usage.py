"""
Per-request token and cost accounting.

``services.llm`` reports a :class:`TokenUsage` for every call it makes, tagged
with the pipeline stage that spent the tokens. That report previously went only
to the logs, so nothing could say what a single question cost. This module gives
it somewhere to land: a per-request accumulator held in a ContextVar, so every
LLM call made anywhere inside one research run — triage, synthesis, follow-ups,
chat — adds to the same total without any node having to thread an accounting
object through the graph state.

``TokenUsage`` itself lives here rather than in ``services.llm`` so the
accounting module owns the accounting type and depends on nothing: the LLM
client imports it, not the other way round. ``services.llm`` re-exports it, so
``from app.services.llm import TokenUsage`` still resolves.

A ContextVar is the right tool because the accumulator must survive the hop into
``asyncio.create_task``: the graph runs in a task spawned by the research
endpoint, and asyncio copies the current context into new tasks. The task
therefore sees the same accumulator *instance* the endpoint created — the
contextvar is never rebound inside the task, only the object is mutated, so
totals are visible to both sides.

Cost is derived from a static price table and is a good-faith estimate for
display, not a billing figure. See :data:`MODEL_PRICES`.

The same accumulator also counts **search credits**. Tokens are not the scarce
resource here — Groq's free tier throttles and recovers, while Tavily's 1,000
credits per month do not come back until the month rolls over. The credit count
is therefore what the spend ceiling in ``services.budget`` actually gates on,
and it is counted rather than inferred from ``len(sub_queries)`` because a
search served from the Redis cache is billed nothing.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting for a single LLM call, attributed to a pipeline stage."""

    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# A usage observer. Kept synchronous on purpose: it is invoked from inside the
# streaming loop, so it must never await or it would stall token delivery.
UsageCallback = Callable[[TokenUsage], None]

# USD per 1,000,000 tokens, as (prompt, completion) — Groq's published list
# prices for the models this app runs.
#
# These are hardcoded on purpose: the alternative is an extra API call per
# request to price a number that only decorates the UI. They will drift. An
# unknown model is priced at 0.0 rather than guessed, and the miss is logged
# once so swapping GROQ_SYNTH_MODEL cannot silently zero the cost line without
# leaving a trace.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}

_PRICE_UNIT_TOKENS = 1_000_000

# Tavily bills per search, by depth. One sub-query = one search, so a four
# sub-query research run costs four credits at basic depth and eight at
# advanced. Serper's free tier is counted the same way (its 2,500 lifetime
# queries are scarcer still per unit).
SEARCH_CREDITS_PER_CALL: dict[str, int] = {
    "basic": 1,
    "advanced": 2,
}


def search_credits_for(depth: str) -> int:
    """Credits billed by one search at ``depth``. Unknown depths cost 1."""
    return SEARCH_CREDITS_PER_CALL.get(depth, 1)

# Models already warned about, so a missing price logs once per process rather
# than once per request.
_unpriced_models: set[str] = set()


def price_for(model: str) -> tuple[float, float]:
    """Return (prompt, completion) USD per 1M tokens for ``model``, or (0, 0)."""
    price = MODEL_PRICES.get(model)
    if price is not None:
        return price

    if model and model not in _unpriced_models:
        _unpriced_models.add(model)
        logger.warning(
            "No price known for model '%s' — its tokens will be counted but "
            "costed at $0. Add it to usage.MODEL_PRICES.",
            model,
        )
    return (0.0, 0.0)


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost of one call. Unknown models cost 0."""
    prompt_price, completion_price = price_for(model)
    return (
        prompt_tokens * prompt_price + completion_tokens * completion_price
    ) / _PRICE_UNIT_TOKENS


@dataclass
class StageUsage:
    """Totals for a single pipeline stage within one request."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class UsageAccumulator:
    """Running token/cost totals for one request, broken down by stage."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    # Billable search-provider calls this request made, and what they cost in
    # provider credits. Both stay zero for a cached search, a document-only
    # question, or an answer-cache hit — all of which are genuinely free.
    searches: int = 0
    search_credits: int = 0
    by_stage: dict[str, StageUsage] = field(default_factory=dict)

    def add_search(self, credits: int) -> None:
        """Record one billable search costing ``credits``. Never raises."""
        self.searches += 1
        self.search_credits += max(0, int(credits))

    def add(self, usage: TokenUsage) -> None:
        """Fold one call's usage into the totals. Never raises."""
        call_cost = cost_of(
            usage.model, usage.prompt_tokens, usage.completion_tokens
        )

        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.cost_usd += call_cost

        stage = self.by_stage.setdefault(usage.stage or "unknown", StageUsage())
        stage.calls += 1
        stage.prompt_tokens += usage.prompt_tokens
        stage.completion_tokens += usage.completion_tokens
        stage.total_tokens += usage.total_tokens
        stage.cost_usd += call_cost

    def as_dict(self) -> dict:
        """Serialize for the SSE ``done`` event."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            # 6dp: a cheap 8B triage call costs ~$0.00002, and rounding that to
            # zero would make the per-stage breakdown useless.
            "cost_usd": round(self.cost_usd, 6),
            "searches": self.searches,
            "search_credits": self.search_credits,
            "by_stage": {name: s.as_dict() for name, s in self.by_stage.items()},
        }


_current: ContextVar[Optional[UsageAccumulator]] = ContextVar(
    "current_usage_accumulator", default=None
)


@contextmanager
def usage_scope() -> Iterator[UsageAccumulator]:
    """
    Collect token usage for everything that runs inside this block.

    Enter the scope *before* spawning the task that runs the graph, so the task
    inherits the accumulator when asyncio copies the context.
    """
    accumulator = UsageAccumulator()
    token = _current.set(accumulator)
    try:
        yield accumulator
    finally:
        try:
            _current.reset(token)
        except ValueError:
            # The scope wraps `yield`s inside an SSE async generator. If the
            # client disconnects, that generator can be closed from a different
            # task than the one that entered the scope, and a Token cannot be
            # reset from a foreign context. Nothing leaks — each request runs in
            # its own task with its own context — so this is cleanup noise, not
            # a failure worth propagating into the response path.
            logger.debug("Usage scope closed from a different context; ignoring.")


def current_usage() -> Optional[UsageAccumulator]:
    """The accumulator for the request in flight, or None outside a scope."""
    return _current.get()


def record_usage(usage: TokenUsage) -> None:
    """
    Ambient observer called by ``services.llm`` for every LLM call.

    A no-op outside a :func:`usage_scope` — LLM calls made from scripts, tests,
    or the startup health check have nowhere to report to, and that is fine.
    Never raises: token accounting must not be able to fail a request.
    """
    accumulator = _current.get()
    if accumulator is None:
        return
    try:
        accumulator.add(usage)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Usage accounting failed (ignored): %s", e)


def record_search(depth: str = "basic") -> None:
    """
    Ambient observer for one *billable* search — call it only on a cache miss.

    Same contract as :func:`record_usage`: a no-op outside a scope, and never
    raises. Credit accounting must not be able to fail a research run; the
    budget guard treats an under-count as spend it will notice on the next
    request, which is strictly better than a 500 on the search path.
    """
    accumulator = _current.get()
    if accumulator is None:
        return
    try:
        accumulator.add_search(search_credits_for(depth))
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Search credit accounting failed (ignored): %s", e)


def empty_usage() -> dict:
    """
    A zeroed usage payload, for runs that spent nothing.

    An answer-cache hit is the case that matters: it must report real zeros
    rather than omitting the field, so the UI can show that the answer was free.
    """
    return UsageAccumulator().as_dict()
