"""Serve raw bytes of previously uploaded files for the document viewer."""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from sqlalchemy import select

from app.models.database import UploadedFile, get_session_factory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["files"])


@router.get("/files/{file_id}")
async def get_file(file_id: str):
    # Public-by-id, matching the sessions endpoint: the unguessable UUID is the
    # capability, so shared/restored sessions can render their attached files.
    try:
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(UploadedFile).where(UploadedFile.id == file_id)
            )
            row = result.scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="File not found")
        # `inline` so PDFs/text render in the iframe instead of downloading.
        safe_name = (row.filename or "file").replace('"', "")
        return Response(
            content=row.content,
            media_type=row.mime or "application/octet-stream",
            headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to serve file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve file")
