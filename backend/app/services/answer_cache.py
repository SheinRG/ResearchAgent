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

A second, semantic layer sits behind the exact one: on an exact miss the
question is embedded and compared against previously answered questions in the
same bucket. It is off by default and requires an embedding key, because unlike
every other optional integration here it can serve a user an answer to a
question they did not ask.

Three limits are structural rather than accidental:

- **A hit must replay the whole SSE sequence**, not just the answer text. The UI
  populates its Sources, Images and Trace tabs from separate events, so
  returning only the answer would render a cited answer with empty tabs.
- **Staleness is bounded by what triage understood.** Triage labels each answer
  time-sensitive or not as it is produced, and that label rides along with the
  cached entry — so a definition can be reused for hours while a price expires
  in minutes. The label describes the answer that was *stored*, which is the
  only thing known at lookup time.
- **Semantic matching is confined to first turns.** The exact key covers the
  conversation history for a reason: "how does its pricing work?" means
  different things in different threads. Meaning-based matching is only safe on
  a question that stands alone.
"""

import hashlib
import json
import logging
import time
from typing import Any, Optional

from app.config import get_settings
from app.services.cache import (
    cache_get,
    cache_set,
    counter_incr,
    counters_get,
    hash_delete,
    hash_get_all,
    hash_set,
)
from app.services.embeddings import cosine
from app.utils.semantic_guards import rejection_reason

logger = logging.getLogger(__name__)

_PREFIX = "answer"
_INDEX_PREFIX = "research:answer_index"

# Hit/miss tallies. Without these the cache is unfalsifiable: it either saves a
# lot of money or none at all, and the logs alone never say which.
_HITS = "answer_cache_hits"
_MISSES = "answer_cache_misses"
_SEMANTIC_HITS = "semantic_cache_hits"
_SEMANTIC_REJECTED = "semantic_cache_rejected"

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


def build_bucket_key(user_name: str, model: str) -> str:
    """
    The vector index a question belongs to.

    Everything that must match EXACTLY lives in this key, so cosine similarity
    is only ever asked to judge the question itself. Two consequences follow:
    a user with a preferred name can never be served an answer addressed to
    someone else, and changing the synthesis model does not resurrect answers
    written by the old one.
    """
    payload = f"{_CACHE_VERSION}|{(user_name or '').strip()}|{model}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{_INDEX_PREFIX}:{digest}"


def ttl_for(time_sensitive: bool) -> int:
    """
    How long an answer may live, given what triage understood about it.

    Before triage emitted this flag every answer expired on the short TTL,
    because nothing could tell a stock price from a definition. Now only the
    answers that actually go stale pay that price.
    """
    settings = get_settings()
    return (
        settings.answer_cache_ttl
        if time_sensitive
        else settings.answer_cache_evergreen_ttl
    )


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


async def _fetch_answer(key: str) -> Optional[dict]:
    """Read a stored payload without tallying anything."""
    payload = await cache_get(_PREFIX, key)
    if isinstance(payload, dict) and (payload.get("answer") or "").strip():
        return payload
    return None


async def get_cached_answer(key: str) -> Optional[dict]:
    """
    Look up a stored answer payload. Returns None on miss or malformed entry.

    Every lookup is tallied, so the denominator is "requests eligible for the
    cache" — uploads and disabled-cache runs never reach here and correctly do
    not count against the hit rate. The semantic layer reads through
    :func:`_fetch_answer` instead, so resolving one of its candidates does not
    register as a second exact-match lookup.
    """
    payload = await _fetch_answer(key)
    await counter_incr(_HITS if payload else _MISSES)
    return payload


async def find_similar_answer(
    query: str,
    embedding: list[float],
    bucket_key: str,
) -> Optional[dict]:
    """
    Find a stored answer to a differently-worded version of the same question.

    Returns the cached payload with ``matched_query`` and ``similarity``
    attached, or None. Candidates are scored by cosine similarity, then every
    one above the threshold must additionally clear the guards in
    ``utils.semantic_guards`` — the score alone rates "best X" and "worst X" as
    near-identical.

    Ordering matters: candidates are tried best-first and the first one that
    passes wins, so a very close but rejected match does not block a slightly
    weaker but legitimate one.
    """
    settings = get_settings()
    entries = await hash_get_all(bucket_key)
    if not entries:
        return None

    scored: list[tuple[float, str, dict]] = []
    for key, entry in entries.items():
        score = cosine(embedding, entry.get("e") or [])
        if score >= settings.semantic_similarity_threshold:
            scored.append((score, key, entry))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    now = int(time.time())
    stale_fields: list[str] = []

    for score, key, entry in scored:
        candidate_query = entry.get("q", "")

        reason = rejection_reason(
            query, candidate_query, settings.semantic_min_token_overlap
        )
        if reason:
            logger.info(
                "Semantic candidate rejected (%.3f): %r vs %r — %s",
                score, query[:60], candidate_query[:60], reason,
            )
            await counter_incr(_SEMANTIC_REJECTED)
            continue

        # A near-match to a time-sensitive answer has to be much fresher than an
        # exact repeat would: it is not even the same question, so any staleness
        # in the stored answer compounds with the difference in intent.
        age = now - int(entry.get("t", 0))
        if entry.get("ts", True) and age > settings.semantic_time_sensitive_max_age:
            logger.info(
                "Semantic candidate too stale (%.3f, %ds old, time-sensitive): %r",
                score, age, candidate_query[:60],
            )
            continue

        payload = await _fetch_answer(key)
        if not payload:
            # The answer expired while its vector lingered. Drop the vector so
            # the next lookup does not pay for it again.
            stale_fields.append(key)
            continue

        if stale_fields:
            await hash_delete(bucket_key, stale_fields)

        await counter_incr(_SEMANTIC_HITS)
        logger.info(
            "Semantic cache HIT (%.3f): %r served from %r",
            score, query[:60], candidate_query[:60],
        )
        return {**payload, "matched_query": candidate_query, "similarity": round(score, 4)}

    if stale_fields:
        await hash_delete(bucket_key, stale_fields)
    return None


async def cache_stats() -> dict:
    """Lifetime tallies plus derived rates, for the health endpoint."""
    counts = await counters_get([_HITS, _MISSES, _SEMANTIC_HITS, _SEMANTIC_REJECTED])
    hits = counts.get(_HITS, 0)
    misses = counts.get(_MISSES, 0)
    semantic_hits = counts.get(_SEMANTIC_HITS, 0)
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 3) if total else 0.0,
        "semantic_hits": semantic_hits,
        "semantic_rejected": counts.get(_SEMANTIC_REJECTED, 0),
        # Share of otherwise-missed questions rescued by meaning-based matching.
        "semantic_rescue_rate": round(semantic_hits / misses, 3) if misses else 0.0,
    }


async def store_answer(
    key: str,
    final_state: dict,
    *,
    query: str = "",
    embedding: Optional[list[float]] = None,
    bucket_key: str = "",
) -> None:
    """
    Store a completed run's user-visible output under ``key``.

    When an embedding is supplied the question is also added to the bucket's
    vector index, making it findable by later questions that mean the same
    thing. Without one the entry is still cached, just exact-match only.
    """
    time_sensitive = bool(final_state.get("time_sensitive", True))
    ttl = ttl_for(time_sensitive)

    payload = {
        "answer": final_state.get("draft_answer", ""),
        "sources": final_state.get("all_sources", []),
        "images": final_state.get("images", []),
        "sub_queries": final_state.get("sub_queries", []),
        "follow_ups": final_state.get("follow_up_suggestions", []),
        "citations": final_state.get("citations", []),
        "invalid_citations": final_state.get("invalid_citations", 0),
        "confidence": final_state.get("confidence", 0.0),
        "trace": final_state.get("trace") or {},
        # Carried so a later semantic hit knows what it is reusing, and can say
        # so to the user.
        "query": query,
        "time_sensitive": time_sensitive,
    }
    await cache_set(_PREFIX, key, payload, ttl=ttl)
    logger.info(
        "Answer cached (ttl=%ds, time_sensitive=%s, %d sources)",
        ttl, time_sensitive, len(payload["sources"]),
    )

    if embedding and bucket_key and query:
        await _index_embedding(
            bucket_key=bucket_key,
            key=key,
            query=query,
            embedding=embedding,
            time_sensitive=time_sensitive,
            ttl=ttl,
        )


async def _index_embedding(
    *,
    bucket_key: str,
    key: str,
    query: str,
    embedding: list[float],
    time_sensitive: bool,
    ttl: int,
) -> None:
    """Add one question's vector to its bucket, pruning the bucket if it grew."""
    await hash_set(
        bucket_key,
        key,
        {
            # Rounded to keep the index small: 5 decimal places is far below the
            # precision at which a similarity threshold could notice.
            "e": [round(v, 5) for v in embedding],
            "q": query,
            "t": int(time.time()),
            "ts": time_sensitive,
        },
        # The bucket outlives its longest entry, then expires on its own if the
        # deployment goes quiet.
        ttl=max(ttl, get_settings().answer_cache_evergreen_ttl),
    )
    await _prune_index(bucket_key)


async def _prune_index(bucket_key: str) -> None:
    """
    Keep a bucket under its entry cap, dropping the oldest first.

    Every lookup scores the whole bucket in-process, so an unbounded index would
    turn a cache hit into the slow path — the exact thing this feature exists to
    avoid.
    """
    settings = get_settings()
    limit = settings.semantic_max_index_entries

    entries = await hash_get_all(bucket_key)
    if len(entries) <= limit:
        return

    by_age = sorted(entries.items(), key=lambda kv: kv[1].get("t", 0))
    doomed = [field for field, _ in by_age[: len(entries) - limit]]
    await hash_delete(bucket_key, doomed)
    logger.info("Semantic index pruned: dropped %d oldest entries", len(doomed))


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

    # The live pipeline emits the trace right after the answer; a replay keeps
    # the same order so the Trace tab fills identically on a cache hit.
    trace = payload.get("trace") or {}
    if trace.get("sources") or trace.get("sub_queries"):
        events.append(("trace", trace))

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
        # Entries cached before this field existed read as 0, which is the same
        # thing a healthy answer reports.
        "invalid_citations": payload.get("invalid_citations", 0),
        "trace": payload.get("trace") or {},
        "time_sensitive": payload.get("time_sensitive", True),
        "confidence": payload.get("confidence", 0.0),
        "follow_up_suggestions": payload.get("follow_ups", []),
        "iteration": 1,
    }
