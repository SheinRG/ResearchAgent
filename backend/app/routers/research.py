"""
Research router — SSE streaming research endpoint and session persistence helper.
"""

import json
import uuid
import asyncio
import logging
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from app.config import get_settings
from app.models.schemas import ResearchRequest
from app.models.database import User, ResearchSession, ResearchQuery, get_session_factory
from app.agents.graph import get_research_graph
from app.routers.auth import get_current_user, check_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["research"])


async def _save_session(session_id: str, query: str, final_state: dict, user_id: str = "", documents: list | None = None) -> None:
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
            )
            db.add(research_query)
            await db.commit()
            logger.info("Saved session %s to database", session_id)
    except Exception as e:
        logger.error("Database save failed for session %s: %s", session_id, e)


@router.post("/research")
async def research(request: ResearchRequest, user: dict = Depends(get_current_user)):
    user_id = user.get("sub", "")
    logger.info("Research request from %s: %s", user.get("email"), request.query[:100])
    await check_rate_limit(user_id)

    # Personalization: how the user wants the agent to address them (if set).
    user_name = ""
    try:
        factory = get_session_factory()
        async with factory() as db:
            db_user = await db.get(User, user_id)
            if db_user:
                user_name = db_user.preferred_name or ""
    except Exception as e:
        logger.warning("Could not load preferred_name for %s: %s", user_id, e)

    async def event_stream() -> AsyncGenerator[str, None]:
        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.monotonic()
        event_queue: asyncio.Queue = asyncio.Queue()

        async def queue_callback(event_type: str, data: dict):
            await event_queue.put((event_type, data))

        async def run_agent():
            try:
                graph = get_research_graph()
                initial_state = {
                    "query": request.query,
                    "max_iterations": request.max_iterations,
                    "history": [{"query": h.query, "answer": h.answer} for h in request.history],
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
                async for state in graph.astream(initial_state):
                    if isinstance(state, dict):
                        for node_output in state.values():
                            if isinstance(node_output, dict):
                                accumulated.update(node_output)
                await event_queue.put(("_final_state", accumulated))
            except Exception as e:
                logger.error("Agent execution failed: %s", e)
                await event_queue.put(("error", {"message": str(e)}))
                await event_queue.put(("_final_state", {"error": str(e)}))

        agent_task = asyncio.create_task(run_agent())
        try:
            while True:
                event_type, data = await asyncio.wait_for(event_queue.get(), timeout=300)
                if event_type == "_final_state":
                    final = data
                    settings = get_settings()
                    done_data = {
                        "session_id": session_id,
                        "total_sources": len(final.get("all_sources", final.get("search_results", []))),
                        "iterations": final.get("iteration", 1),
                        "confidence": final.get("confidence", 0.0),
                        "model": settings.groq_synth_model,
                        "latency_ms": int((time.monotonic() - start_time) * 1000),
                    }
                    try:
                        await _save_session(
                            session_id=session_id,
                            query=request.query,
                            final_state=final,
                            user_id=user_id,
                            documents=[
                                {"name": d.name, "file_id": d.file_id, "mime": d.mime, "size": d.size}
                                for d in request.documents
                                if d.file_id
                            ],
                        )
                    except Exception as e:
                        logger.error("Failed to save session: %s", e)
                    yield f"event: done\ndata: {json.dumps(done_data)}\n\n"
                    break
                elif event_type == "error":
                    yield f"event: error\ndata: {json.dumps(data)}\n\n"
                else:
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        except asyncio.TimeoutError:
            logger.error("Research timed out after 5 minutes")
            yield f"event: error\ndata: {json.dumps({'message': 'Research timed out'})}\n\n"
        finally:
            if not agent_task.done():
                agent_task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
