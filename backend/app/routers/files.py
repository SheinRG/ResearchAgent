"""Serve raw bytes of previously uploaded files for the document viewer."""
import logging
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select

from app.models.database import UploadedFile, get_session_factory
from app.routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["files"])

# Anything that could terminate or inject a header, plus quotes/backslashes that
# would break out of the quoted filename.
_UNSAFE_FILENAME_CHARS = re.compile(r'[\r\n"\\]')


def _safe_filename(name: str) -> str:
    return _UNSAFE_FILENAME_CHARS.sub("", name or "").strip() or "file"


@router.get("/files/{file_id}")
async def get_file(file_id: str, user: dict = Depends(get_current_user)):
    """
    Serve an uploaded file's bytes to its owner.

    Deliberately NOT public-by-UUID any more. These bytes are whatever the user
    uploaded — resumes, contracts, notes — and the old unauthenticated route
    also let an attacker host attacker-controlled content on the API origin,
    which is same-origin with the refresh-token cookie.

    Served as an attachment with nosniff so the browser never renders it as a
    document; the viewer fetches it and builds its own blob URL.
    """
    try:
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(UploadedFile).where(UploadedFile.id == file_id)
            )
            row = result.scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="File not found")
        # 404 rather than 403 on a mismatch: a 403 would confirm the id exists.
        # Rows with no owner (pre-auth uploads) belong to nobody and stay closed.
        if not row.user_id or row.user_id != user.get("sub"):
            logger.warning(
                "Blocked cross-user file access: file_id=%s requested by user=%s",
                file_id, user.get("sub"),
            )
            raise HTTPException(status_code=404, detail="File not found")
        return Response(
            content=row.content,
            media_type=row.mime or "application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{_safe_filename(row.filename)}"',
                "X-Content-Type-Options": "nosniff",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to serve file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve file")
