"""
Research router — SSE streaming research endpoint and session persistence helper.
"""

import json
import uuid
import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from app.config import get_settings
from app.models.schemas import ResearchRequest
from app.models.database import User, ResearchSession, ResearchQuery, get_session_factory
from app.agents.graph import get_research_graph
from app.routers.auth import Principal, get_principal, check_rate_limit
from app.services import budget
from app.services.anonymous import (
    consume_quota,
    peek_quota,
    read_anon_id,
    set_anon_cookie,
)
from app.services.answer_cache import (
    build_answer_key,
    build_bucket_key,
    cached_state_for_save,
    find_similar_answer,
    get_cached_answer,
    is_cacheable,
    iter_replay_events,
    store_answer,
)
from app.services import embeddings, tracing
from app.services.usage import current_usage, empty_usage, usage_scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["research"])

# Strong references to detached background tasks. asyncio only holds a weak
# reference to a running task, so without this the garbage collector is free to
# cancel a session write mid-flight.
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    """Run a coroutine detached from the request, keeping it alive until done."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _sse(event_type: str, data: dict) -> str:
    """Serialize one SSE frame."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _save_session(
    session_id: str,
    query: str,
    final_state: dict,
    user_id: str = "",
    documents: list | None = None,
    usage: dict | None = None,
) -> None:
    try:
        factory = get_session_factory()
        async with factory() as db:
            existing = await db.get(ResearchSession, session_id)
            if existing is None:
                db.add(ResearchSession(id=session_id, user_id=user_id or None))
            sources = final_state.get("all_sources", final_state.get("search_results", []))
            research_query = ResearchQuery(
                session_id=session_id,
                user_id=user_id or None,
                query=query,
                sub_queries=final_state.get("sub_queries", []),
                answer=final_state.get("draft_answer", ""),
                sources=sources[:20],
                citations=final_state.get("citations", []),
                confidence=final_state.get("confidence", 0.0),
                iterations=final_state.get("iteration", 1),
                follow_up_suggestions=final_state.get("follow_up_suggestions", []),
                documents=documents or [],
                invalid_citations=final_state.get("invalid_citations", 0),
                total_tokens=(usage or {}).get("total_tokens", 0),
                cost_usd=(usage or {}).get("cost_usd", 0.0),
                # Ground truth for the spend ceiling. Written here, on the same
                # row as the answer, so the budget can never be read from a
                # store that is allowed to evict it.
                search_credits=(usage or {}).get("search_credits", 0),
                trace=final_state.get("trace") or {},
            )
            db.add(research_query)
            await db.commit()
            logger.info("Saved session %s to database", session_id)
    except Exception as e:
        logger.error("Database save failed for session %s: %s", session_id, e)


@router.post("/research")
async def research(
    request: ResearchRequest,
    http_request: Request,
    principal: Principal = Depends(get_principal),
):
    user_id = principal.user_id
    logger.info("Research request from %s: %s", principal.label, request.query[:100])

    if principal.is_anonymous:
        # Checked before anything is done, and read-only: a visitor who is
        # turned away here must not have a free query deducted for it. The
        # spend ceiling is checked later, on a cache miss, because a cached
        # answer costs nothing and should be served even when live runs are
        # paused.
        quota = await peek_quota(http_request, principal.anon_id)
        if quota.exhausted:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "demo_quota_exhausted",
                    "limit": quota.limit,
                    "message": (
                        f"You've used your {quota.limit} free queries. "
                        "Sign in to keep going — it's free."
                    ),
                },
            )
    else:
        await check_rate_limit(user_id)

    # Personalization: how the user wants the agent to address them (if set).
    # Anonymous visitors have no profile, and looking one up by "" would scan
    # for a user that cannot exist.
    user_name = ""
    if user_id:
        try:
            factory = get_session_factory()
            async with factory() as db:
                db_user = await db.get(User, user_id)
                if db_user:
                    user_name = db_user.preferred_name or ""
        except Exception as e:
            logger.warning("Could not load preferred_name for %s: %s", user_id, e)

    # Traces stay attributable without carrying an identity: a demo visitor is
    # a truncated random id, which groups one visitor's turns together and says
    # nothing about who they are.
    trace_actor = user_id or f"anon:{principal.anon_id[:8]}"

    async def event_stream() -> AsyncGenerator[str, None]:
        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.monotonic()
        settings = get_settings()
        event_queue: asyncio.Queue = asyncio.Queue()

        async def spend_free_query() -> dict | None:
            """
            Deduct one demo query at the moment we commit to serving.

            Returns an SSE payload if the visitor raced past the pre-flight
            check in another tab, otherwise None. A cache hit spends one too:
            the per-visitor allowance is a product decision about when to ask
            for a signup, and the visitor got an answer either way. What the
            run *costs* is the budget guard's business, and a replay costs
            nothing, so it never touches the credit pool.
            """
            if not principal.is_anonymous:
                return None
            try:
                await consume_quota(http_request, principal.anon_id)
                return None
            except HTTPException as e:
                detail = e.detail if isinstance(e.detail, dict) else {}
                return {
                    "code": "demo_quota_exhausted",
                    "limit": detail.get("limit", get_settings().anon_free_queries),
                    "message": detail.get(
                        "message", "You've used your free queries. Sign in to keep going."
                    ),
                }

        history = [{"query": h.query, "answer": h.answer} for h in request.history]
        stored_documents = [
            {"name": d.name, "file_id": d.file_id, "mime": d.mime, "size": d.size}
            for d in request.documents
            if d.file_id
        ]

        # --- Exact-match answer cache -------------------------------------
        # Consulted before the graph runs, so a repeat question skips triage,
        # search, rerank AND synthesis. Uploads bypass entirely: the document
        # text is not part of the key, so a hit could answer from the wrong file.
        cache_key: str | None = None
        bucket_key = ""
        question_embedding: list[float] | None = None
        cached = None
        cache_kind = "exact"

        if settings.answer_cache_enabled and not request.documents:
            cache_key = build_answer_key(
                query=request.query,
                history=history,
                user_name=user_name,
                model=settings.groq_synth_model,
            )
            bucket_key = build_bucket_key(user_name, settings.groq_synth_model)

            with tracing.span(
                "answer-cache-lookup", input=tracing.preview(request.query)
            ) as lookup_span:
                cached = await get_cached_answer(cache_key)
                lookup_span.update(output={"hit": bool(cached)})

            # --- Semantic layer -------------------------------------------
            # Only on an exact miss, and only for a standalone question: the
            # exact key covers conversation history because "how does its
            # pricing work?" means different things in different threads, and
            # meaning-based matching cannot preserve that.
            semantic_eligible = (
                settings.semantic_cache_enabled
                and not cached
                and not history
                and embeddings.is_configured()
            )
            if semantic_eligible:
                with tracing.span(
                    "semantic-cache-lookup", input=tracing.preview(request.query)
                ) as semantic_span:
                    question_embedding = await embeddings.embed(request.query)
                    if question_embedding:
                        cached = await find_similar_answer(
                            request.query, question_embedding, bucket_key
                        )
                    if cached:
                        cache_kind = "semantic"
                    semantic_span.update(output={
                        "hit": bool(cached),
                        "embedded": bool(question_embedding),
                        "similarity": (cached or {}).get("similarity"),
                        "matched_query": tracing.preview((cached or {}).get("matched_query", "")),
                    })

            if cached:
                blocked = await spend_free_query()
                if blocked:
                    yield _sse("limit", blocked)
                    return
                logger.info(
                    "Answer cache HIT (%s) for: %s", cache_kind, request.query[:80]
                )
                # A hit still gets a `research` observation, so cached and live
                # turns sit side by side in the dashboard and the hit rate is
                # visible there as well as on /api/health. Opened and closed
                # before any yield: the SSE generator can be finalized from a
                # different task, and a span must not straddle that.
                with tracing.request_scope(
                    user_id=trace_actor, session_id=session_id, tags=["research", "cached"]
                ):
                    with tracing.span(
                        "research",
                        input=tracing.preview(request.query),
                        metadata={"cached": True, "cache_kind": cache_kind},
                    ) as root_span:
                        root_span.update(output={
                            "answer": tracing.preview(cached.get("answer", "")),
                            "sources": len(cached.get("sources", [])),
                            "invalid_citations": cached.get("invalid_citations", 0),
                            "cost_usd": 0.0,
                        })
                for event_type, data in iter_replay_events(cached):
                    yield _sse(event_type, data)

                # Persist the turn so a cached answer still lands in the user's
                # history — detached, so it never delays the response.
                _spawn(_save_session(
                    session_id=session_id,
                    query=request.query,
                    final_state=cached_state_for_save(cached),
                    user_id=user_id,
                    documents=stored_documents,
                    usage=empty_usage(),
                ))

                yield _sse("done", {
                    "session_id": session_id,
                    "total_sources": len(cached.get("sources", [])),
                    "iterations": 1,
                    "confidence": cached.get("confidence", 0.0),
                    "invalid_citations": cached.get("invalid_citations", 0),
                    "model": settings.groq_synth_model,
                    "latency_ms": int((time.monotonic() - start_time) * 1000),
                    "cached": True,
                    # "exact" or "semantic". A semantic hit answers a question
                    # the user did not literally ask, so the UI says so and
                    # shows the question it actually reused.
                    "cache_kind": cache_kind,
                    "matched_query": cached.get("matched_query", ""),
                    "similarity": cached.get("similarity"),
                    # A replayed answer makes no LLM calls at all. Reporting
                    # real zeros (rather than omitting the field) is the point:
                    # it is what makes the cache's saving visible.
                    "usage": empty_usage(),
                })
                return

        # --- Spend ceiling ------------------------------------------------
        # Only reached on a cache miss, i.e. only when this turn is about to
        # cost real search credits. A breach degrades to cache-only rather than
        # returning 503: the visitor still gets every answer already in the
        # cache, and an honest "paused" is a far better failure than a demo
        # that falls over during the one spike that matters.
        budget_status = await budget.check_budget(anonymous=principal.is_anonymous)
        if not budget_status.allowed:
            logger.warning(
                "Budget ceiling reached (%s); refusing live run for %s",
                budget_status.reason, principal.label,
            )
            yield _sse("paused", {
                "code": budget_status.reason,
                "message": budget_status.message,
                # True when signing in would actually help — a demo sub-pool
                # breach, not a global one. Saying "sign in to continue" when
                # the whole month is spent would be a lie.
                "signin_helps": budget_status.reason == "anon_daily",
            })
            yield _sse("done", {
                "session_id": session_id,
                "total_sources": 0,
                "iterations": 0,
                "confidence": 0.0,
                "invalid_citations": 0,
                "model": settings.groq_synth_model,
                "latency_ms": int((time.monotonic() - start_time) * 1000),
                "cached": False,
                "cache_kind": "",
                "paused": True,
                "usage": empty_usage(),
            })
            return

        blocked = await spend_free_query()
        if blocked:
            yield _sse("limit", blocked)
            return

        # Book the worst case before the run starts. Reserving what it *might*
        # cost rather than what it did is what stops a burst of concurrent
        # requests from all reading the same 60s-old total and all being let in.
        await budget.reserve(anonymous=principal.is_anonymous)

        async def queue_callback(event_type: str, data: dict):
            await event_queue.put((event_type, data))

        async def run_agent():
            try:
                graph = get_research_graph()
                initial_state = {
                    "query": request.query,
                    "max_iterations": request.max_iterations,
                    "history": history,
                    "user_name": user_name,
                    "iteration": 0,
                    "sub_queries": [],
                    "documents": [d.model_dump() for d in request.documents],
                    "document_chunks": [],
                    "search_results": [],
                    "scraped_content": [],
                    "ranked_chunks": [],
                    "draft_answer": "",
                    "citations": [],
                    "all_sources": [],
                    "reflection": {},
                    "confidence": 0.0,
                    "follow_up_suggestions": [],
                    "phase": "starting",
                    "error": "",
                    # Router overwrites this; default True so a no-doc run is
                    # never accidentally blocked from searching the web.
                    "needs_web": True,
                    "sse_callback": queue_callback,
                }
                accumulated: dict = {}
                # The root span lives HERE, inside the task, rather than around
                # the SSE generator: the generator's cleanup can run in a
                # different asyncio context when a client disconnects, and an
                # OpenTelemetry span cannot be closed from a foreign context.
                # The whole graph runs within this one task, so every stage span
                # nests under this root automatically.
                with tracing.request_scope(
                    user_id=trace_actor, session_id=session_id, tags=["research"]
                ):
                    with tracing.span(
                        "research",
                        input=tracing.preview(request.query),
                        metadata={
                            "cached": False,
                            "has_documents": bool(request.documents),
                            "history_turns": len(history),
                        },
                    ) as root_span:
                        async for state in graph.astream(initial_state):
                            if isinstance(state, dict):
                                for node_output in state.values():
                                    if isinstance(node_output, dict):
                                        accumulated.update(node_output)

                        spent = current_usage()
                        root_span.update(output={
                            "answer": tracing.preview(accumulated.get("draft_answer", "")),
                            "mode": accumulated.get("mode", "research"),
                            "sources": len(accumulated.get("all_sources") or []),
                            "invalid_citations": accumulated.get("invalid_citations", 0),
                            "confidence": accumulated.get("confidence", 0.0),
                            # The cost of the whole turn, on the trace that owns
                            # it — the per-stage split is on the child spans.
                            "cost_usd": spent.as_dict()["cost_usd"] if spent else 0.0,
                        })

                await event_queue.put(("_final_state", accumulated))
            except Exception as e:
                logger.error("Agent execution failed: %s", e)
                await event_queue.put(("error", {"message": str(e)}))
                await event_queue.put(("_final_state", {"error": str(e)}))

        # Collect token usage for every LLM call this request makes. Entered
        # BEFORE the graph task is spawned: asyncio copies the current context
        # into new tasks, so the task inherits this accumulator and the totals
        # it adds from inside the graph are visible out here.
        with usage_scope() as request_usage:
            agent_task = asyncio.create_task(run_agent())
            try:
                while True:
                    event_type, data = await asyncio.wait_for(event_queue.get(), timeout=300)
                    if event_type == "_final_state":
                        final = data
                        usage = request_usage.as_dict()
                        done_data = {
                            "session_id": session_id,
                            "total_sources": len(final.get("all_sources", final.get("search_results", []))),
                            "iterations": final.get("iteration", 1),
                            "confidence": final.get("confidence", 0.0),
                            "invalid_citations": final.get("invalid_citations", 0),
                            "model": settings.groq_synth_model,
                            "latency_ms": int((time.monotonic() - start_time) * 1000),
                            "cached": False,
                            "cache_kind": "",
                            "usage": usage,
                        }
                        logger.info(
                            "Research complete: %d tokens, $%.6f, %dms",
                            usage["total_tokens"], usage["cost_usd"],
                            done_data["latency_ms"],
                        )

                        # Persist detached: a slow managed-Postgres write used to sit
                        # between the answer finishing and the user seeing "done".
                        _spawn(_save_session(
                            session_id=session_id,
                            query=request.query,
                            final_state=final,
                            user_id=user_id,
                            documents=stored_documents,
                            usage=usage,
                        ))

                        if cache_key and is_cacheable(final, bool(request.documents)):
                            # The embedding was computed during the (missed)
                            # semantic lookup, so indexing this answer costs
                            # nothing extra. When semantic matching is off it is
                            # None and the entry is stored exact-match only.
                            _spawn(store_answer(
                                cache_key,
                                final,
                                query=request.query,
                                embedding=question_embedding,
                                bucket_key=bucket_key,
                            ))

                        yield _sse("done", done_data)
                        break
                    elif event_type == "error":
                        yield _sse("error", data)
                    else:
                        yield _sse(event_type, data)
            except asyncio.TimeoutError:
                logger.error("Research timed out after 5 minutes")
                yield _sse("error", {"message": "Research timed out"})
            finally:
                if not agent_task.done():
                    agent_task.cancel()

    response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
    # A visitor who lands straight on /api/research without calling
    # /api/demo/status first still needs their id pinned, or every query would
    # mint a fresh identity and the allowance would never count down.
    if principal.is_anonymous and not read_anon_id(http_request):
        set_anon_cookie(response, principal.anon_id)
    return response
