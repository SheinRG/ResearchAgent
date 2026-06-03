"""
FastAPI application — CORS, SSE endpoints, session management.
Main entry point for the AI Research Agent backend.
"""

import json
import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.config import get_settings
from app.models.schemas import (
    ResearchRequest,
    SessionResponse,
    SessionListItem,
    SearchResult,
    DoneEvent,
)
from app.models.database import (
    init_db,
    close_db,
    get_db,
    ResearchSession,
    ResearchQuery,
)
from app.agents.graph import get_research_graph
from app.services.llm import get_llm_client
from app.services.cache import close_redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("Starting AI Research Agent backend...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Database init failed (will retry on first request): %s", e)

    # Check Ollama health
    try:
        llm = get_llm_client()
        healthy = await llm.health_check()
        if healthy:
            logger.info("Ollama is healthy (model: %s)", llm.model)
        else:
            logger.warning("Ollama model not found — ensure it's pulled")
    except Exception as e:
        logger.warning("Ollama health check failed: %s", e)

    yield

    # Shutdown
    await close_redis()
    await close_db()
    logger.info("Backend shutdown complete")


# --- FastAPI App ---
app = FastAPI(
    title="AI Research Agent",
    description="Autonomous research agent with cited answers — 100% local, zero API keys",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health Check ---
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    llm = get_llm_client()
    ollama_ok = await llm.health_check()
    return {
        "status": "healthy",
        "ollama": "connected" if ollama_ok else "disconnected",
        "model": llm.model,
    }


# --- SSE Research Endpoint ---
@app.post("/api/research")
async def research(request: ResearchRequest):
    """
    Start a research session. Streams SSE events as the agent works.

    Events streamed:
        - phase: Current phase (planning, searching, reading, writing, reflecting)
        - sub_queries: Generated sub-queries
        - sources: Found web sources
        - token: Individual answer tokens (streamed)
        - follow_up: Suggested follow-up questions
        - done: Final session metadata
    """
    logger.info("Research request: %s", request.query[:100])

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events from the research agent."""
        session_id = request.session_id or str(uuid.uuid4())
        collected_events = []

        async def sse_callback(event_type: str, data: dict):
            """Callback for agent nodes to emit SSE events."""
            event_str = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            collected_events.append((event_type, data))
            return event_str

        # We need a different approach: collect and yield
        # Use an async queue for real-time streaming
        import asyncio
        event_queue: asyncio.Queue = asyncio.Queue()

        async def queue_callback(event_type: str, data: dict):
            """Put events into the queue for streaming."""
            await event_queue.put((event_type, data))

        async def run_agent():
            """Run the research graph and signal completion."""
            try:
                graph = get_research_graph()
                initial_state = {
                    "query": request.query,
                    "max_iterations": request.max_iterations,
                    "iteration": 0,
                    "sub_queries": [],
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
                    "sse_callback": queue_callback,
                }

                # Run the graph
                final_state = None
                async for state in graph.astream(initial_state):
                    # Each yield is a partial state update from a node
                    if isinstance(state, dict):
                        for node_name, node_output in state.items():
                            if isinstance(node_output, dict):
                                final_state = node_output

                # Signal completion
                await event_queue.put(("_final_state", final_state or {}))

            except Exception as e:
                logger.error("Agent execution failed: %s", e)
                await event_queue.put(("error", {"message": str(e)}))
                await event_queue.put(("_final_state", {"error": str(e)}))

        # Start agent in background task
        import asyncio
        agent_task = asyncio.create_task(run_agent())

        # Stream events from the queue
        try:
            while True:
                event_type, data = await asyncio.wait_for(
                    event_queue.get(), timeout=300  # 5 minute timeout
                )

                if event_type == "_final_state":
                    # Save to database and emit done event
                    final = data
                    done_data = {
                        "session_id": session_id,
                        "total_sources": len(final.get("all_sources", final.get("search_results", []))),
                        "iterations": final.get("iteration", 1),
                        "confidence": final.get("confidence", 0.0),
                    }

                    # Save to database asynchronously
                    try:
                        await _save_session(
                            session_id=session_id,
                            query=request.query,
                            final_state=final,
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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Session Endpoints ---
@app.get("/api/sessions")
async def list_sessions(limit: int = 20):
    """List recent research sessions."""
    try:
        from app.models.database import get_session_factory
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(ResearchQuery)
                .order_by(desc(ResearchQuery.created_at))
                .limit(limit)
            )
            queries = result.scalars().all()

            return [
                SessionListItem(
                    id=q.session_id,
                    query=q.query,
                    confidence=q.confidence or 0.0,
                    created_at=q.created_at.isoformat() if q.created_at else "",
                ).model_dump()
                for q in queries
            ]
    except Exception as e:
        logger.error("Failed to list sessions: %s", e)
        return []


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve a past research session."""
    try:
        from app.models.database import get_session_factory
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(ResearchQuery)
                .where(ResearchQuery.session_id == session_id)
                .order_by(desc(ResearchQuery.created_at))
                .limit(1)
            )
            query = result.scalar_one_or_none()

            if not query:
                raise HTTPException(status_code=404, detail="Session not found")

            return SessionResponse(
                id=query.session_id,
                query=query.query,
                answer=query.answer,
                sources=[SearchResult(**s) for s in (query.sources or [])],
                citations=query.citations or [],
                confidence=query.confidence or 0.0,
                iterations=query.iterations or 1,
                follow_up_suggestions=query.follow_up_suggestions or [],
                created_at=query.created_at.isoformat() if query.created_at else "",
            ).model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


async def _save_session(session_id: str, query: str, final_state: dict) -> None:
    """Save a completed research session to the database."""
    try:
        from app.models.database import get_session_factory
        factory = get_session_factory()
        async with factory() as db:
            # Create session
            session = ResearchSession(id=session_id)
            db.add(session)

            # Create query record
            sources = final_state.get("all_sources", final_state.get("search_results", []))
            research_query = ResearchQuery(
                session_id=session_id,
                query=query,
                sub_queries=final_state.get("sub_queries", []),
                answer=final_state.get("draft_answer", ""),
                sources=sources[:20],  # Limit stored sources
                citations=final_state.get("citations", []),
                confidence=final_state.get("confidence", 0.0),
                iterations=final_state.get("iteration", 1),
                follow_up_suggestions=final_state.get("follow_up_suggestions", []),
            )
            db.add(research_query)
            await db.commit()

            logger.info("Saved session %s to database", session_id)

    except Exception as e:
        logger.error("Database save failed for session %s: %s", session_id, e)
