"""
Sessions router — list sessions, get a single session thread.
"""

import logging

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select, desc, delete

from app.models.schemas import (
    SessionListItem,
    SessionTurn,
    SessionThreadResponse,
    SearchResult,
)
from app.models.database import ResearchQuery, ResearchSession, get_session_factory
from app.routers.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["sessions"])


def _safe_search_result(raw: dict) -> SearchResult | None:
    if not isinstance(raw, dict):
        return None
    return SearchResult(
        url=raw.get("url", ""),
        title=raw.get("title", ""),
        domain=raw.get("domain", raw.get("url", "")),
        favicon=raw.get("favicon", ""),
        snippet=raw.get("snippet", ""),
    )


@router.get("/sessions")
async def list_sessions(limit: int = 20, user: dict = Depends(get_current_user)):
    user_id = user.get("sub", "")
    try:
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(ResearchQuery)
                .where(ResearchQuery.user_id == user_id)
                .order_by(desc(ResearchQuery.created_at))
                .limit(200)
            )
            rows = result.scalars().all()
        sessions: dict[str, list] = {}
        for row in rows:
            sessions.setdefault(row.session_id, []).append(row)
        items = []
        for sid, turns in sessions.items():
            turns_asc = sorted(turns, key=lambda r: r.created_at or "")
            first = turns_asc[0]
            last = turns_asc[-1]
            title = first.query or ""
            created_at = first.created_at.isoformat() if first.created_at else ""
            updated_at = last.created_at.isoformat() if last.created_at else created_at
            confidence = last.confidence or 0.0
            turn_count = len(turns_asc)
            items.append(
                SessionListItem(
                    id=sid,
                    query=title,
                    title=title,
                    confidence=confidence,
                    turn_count=turn_count,
                    created_at=created_at,
                    updated_at=updated_at,
                ).model_dump()
            )
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return items[:limit]
    except Exception as e:
        logger.error("Failed to list sessions: %s", e)
        return []


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a history thread owned by the current user (and all its turns)."""
    user_id = user.get("sub", "")
    try:
        factory = get_session_factory()
        async with factory() as db:
            # Ownership check: only the owner's rows for this session may be deleted.
            owned = await db.execute(
                select(ResearchQuery.id)
                .where(ResearchQuery.session_id == session_id)
                .where(ResearchQuery.user_id == user_id)
                .limit(1)
            )
            if owned.scalar_one_or_none() is None:
                raise HTTPException(status_code=404, detail="Thread not found")

            # Remove the turns, then the parent session row (if it belongs here).
            await db.execute(
                delete(ResearchQuery)
                .where(ResearchQuery.session_id == session_id)
                .where(ResearchQuery.user_id == user_id)
            )
            await db.execute(
                delete(ResearchSession)
                .where(ResearchSession.id == session_id)
                .where(ResearchSession.user_id == user_id)
            )
            await db.commit()
        logger.info("Deleted session %s for user %s", session_id, user_id)
        return {"deleted": True, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail="Failed to delete thread")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    # Public endpoint — the UUID is the only authorization needed for read access.
    try:
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(ResearchQuery)
                .where(ResearchQuery.session_id == session_id)
                .order_by(ResearchQuery.created_at)
            )
            rows = result.scalars().all()
        if not rows:
            raise HTTPException(status_code=404, detail="Session not found")
        turns = []
        for row in rows:
            sources = [
                sr
                for raw in (row.sources or [])
                if (sr := _safe_search_result(raw)) is not None
            ]
            turns.append(
                SessionTurn(
                    query=row.query or "",
                    answer=row.answer or "",
                    sources=sources,
                    citations=row.citations or [],
                    confidence=row.confidence or 0.0,
                    iterations=row.iterations or 1,
                    follow_up_suggestions=row.follow_up_suggestions or [],
                    documents=row.documents or [],
                    created_at=row.created_at.isoformat() if row.created_at else "",
                )
            )
        first = rows[0]
        return SessionThreadResponse(
            id=session_id,
            title=first.query or "",
            created_at=first.created_at.isoformat() if first.created_at else "",
            turns=turns,
        ).model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve session")
