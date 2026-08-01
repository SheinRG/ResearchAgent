"""
Exact-match answer cache — short-circuits an identical repeat question before it
reaches the graph.

The existing caches in ``services.cache`` sit *inside* the pipeline: they save
the search and scrape round-trips for a repeated sub-query, but the triage call,
the rerank, and the expensive synthesis call all still run. This cache sits in
front of the whole graph, so a repeat question costs one Redis GET.

Deliberately EXACT match, not semantic. The key covers everything that changes
the answer — the normalized question, the conversation history that will enter
the prompt, the user's preferred name, and the synthesis model. A semantic layer
that also catches near-duplicate phrasings belongs on top of this seam, reusing
``build_answer_key`` as the exact-hit fast path.

Two limits are structural rather than accidental:

- **Staleness is bounded by TTL, not by understanding.** The cache is consulted
  before triage runs, so nothing here knows whether the question is time
  sensitive. ``answer_cache_ttl`` is set short for that reason.
- **A hit must replay the whole SSE sequence**, not just the answer text. The UI
  populates its Sources and Images tabs from separate events, so returning only
  the answer would render a cited answer with an empty Sources tab.
"""

import json
import hashlib
import logging
from typing import Any, Optional

from app.services.cache import cache_get, cache_set
from app.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "answer"

# Bump to invalidate every cached answer at once — required whenever a prompt,
# the source-formatting, or the event contract below changes, since old entries
# would otherwise be replayed under new rendering rules.
_CACHE_VERSION = 1

# Mirror format_history's defaults so the key covers exactly the history that
# actually reaches the prompt. Keying on more history than the model sees would
# miss hits that are genuinely identical from the model's point of view.
_HISTORY_TURNS = 4
_HISTORY_ANSWER_CHARS = 600


def build_answer_key(
    query: str,
    history: list[dict] | None,
    user_name: str,
    model: str,
) -> str:
    """
    Build the cache key for a question plus everything that changes its answer.

    Args:
        query: The user's question.
        history: Prior {query, answer} turns, as passed to the graph.
        user_name: The user's preferred name (it appears in the prompt).
        model: The synthesis model (a model swap invalidates old answers).

    Returns:
        A hex digest identifying this exact question-in-context.
    """
    payload = {
        "v": _CACHE_VERSION,
        "q": " ".join((query or "").lower().split()),
        "u": (user_name or "").strip(),
        "m": model,
        "h": [
            {
                "q": " ".join((turn.get("query") or "").lower().split()),
                "a": (turn.get("answer") or "").strip()[:_HISTORY_ANSWER_CHARS],
            }
            for turn in (history or [])[-_HISTORY_TURNS:]
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def is_cacheable(final_state: dict, has_documents: bool) -> bool:
    """
    Whether a completed run may be stored.

    Excluded on purpose:
    - runs with uploaded documents (the answer is specific to that upload, and
      the document text is not part of the key);
    - failed runs and empty answers;
    - chat replies, which have no sources. They are cheap to regenerate, and
      replaying a canned reply to "hi" makes the assistant feel broken.
    """
    if has_documents:
        return False
    if final_state.get("error"):
        return False
    if not (final_state.get("draft_answer") or "").strip():
        return False
    if not final_state.get("all_sources"):
        return False
    return True


async def get_cached_answer(key: str) -> Optional[dict]:
    """Look up a stored answer payload. Returns None on miss or malformed entry."""
    payload = await cache_get(_PREFIX, key)
    if not isinstance(payload, dict):
        return None
    if not (payload.get("answer") or "").strip():
        return None
    return payload


async def store_answer(key: str, final_state: dict) -> None:
    """Store a completed run's user-visible output under ``key``."""
    settings = get_settings()
    payload = {
        "answer": final_state.get("draft_answer", ""),
        "sources": final_state.get("all_sources", []),
        "images": final_state.get("images", []),
        "sub_queries": final_state.get("sub_queries", []),
        "follow_ups": final_state.get("follow_up_suggestions", []),
        "citations": final_state.get("citations", []),
        "confidence": final_state.get("confidence", 0.0),
    }
    await cache_set(_PREFIX, key, payload, ttl=settings.answer_cache_ttl)
    logger.info(
        "Answer cached (ttl=%ds, %d sources)",
        settings.answer_cache_ttl,
        len(payload["sources"]),
    )


def iter_replay_events(payload: dict) -> list[tuple[str, dict]]:
    """
    Rebuild the SSE event sequence a live run would have emitted.

    Order matters and mirrors the real pipeline: sub-queries, then the
    authoritative source list, images, the answer, then follow-ups. The answer
    is emitted as a single token event — the frontend appends tokens, so one
    large token renders identically to thousands of small ones, and arriving
    instantly is the point.

    Returns:
        A list of (event_type, data) tuples for the caller to serialize.
    """
    events: list[tuple[str, dict]] = []

    sub_queries = payload.get("sub_queries") or []
    if sub_queries:
        events.append(("sub_queries", {"queries": sub_queries}))

    sources = payload.get("sources") or []
    if sources:
        # `replace` marks this as the authoritative citation-ordered list, the
        # same contract the synthesizer uses: index i here is marker [i].
        events.append(("sources", {"sources": sources, "replace": True}))

    images = payload.get("images") or []
    if images:
        events.append(("images", {"images": images}))

    events.append(("phase", {"phase": "writing", "message": "Synthesizing your answer..."}))
    events.append(("token", {"token": payload.get("answer", "")}))

    follow_ups = payload.get("follow_ups") or []
    if follow_ups:
        events.append(("follow_up", {"suggestions": follow_ups}))

    return events


def cached_state_for_save(payload: dict) -> dict[str, Any]:
    """
    Shape a cached payload like a graph final state, so a cache hit is persisted
    to the user's session history the same way a live run is.
    """
    return {
        "draft_answer": payload.get("answer", ""),
        "all_sources": payload.get("sources", []),
        "sub_queries": payload.get("sub_queries", []),
        "citations": payload.get("citations", []),
        "confidence": payload.get("confidence", 0.0),
        "follow_up_suggestions": payload.get("follow_ups", []),
        "iteration": 1,
    }
