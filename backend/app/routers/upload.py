"""File upload and text extraction endpoint."""
import logging
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.services.file_processor import extract_text, SUPPORTED_EXTENSIONS
from app.routers.auth import get_current_user
from app.models.database import UploadedFile, get_session_factory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Upload a file and extract its text content.

    Supports .txt, .md, .pdf, .docx files up to 5MB.
    Returns the extracted text (truncated at 4000 chars), filename, char count,
    file_id (server-side storage id), mime type, and byte size.
    """
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max size is 5MB.")

    try:
        text = extract_text(file.filename or "", content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("File extraction error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to extract text from file.")

    mime = file.content_type or ""
    file_id = ""

    try:
        file_id = str(uuid.uuid4())
        factory = get_session_factory()
        async with factory() as db:
            db.add(UploadedFile(
                id=file_id,
                user_id=user.get("sub") or None,
                filename=file.filename or "",
                mime=mime,
                size=len(content),
                content=content,
            ))
            await db.commit()
        logger.info(
            "File uploaded and persisted by user %s: %s (%d chars extracted, file_id=%s)",
            user.get("sub"),
            file.filename,
            len(text),
            file_id,
        )
    except Exception as e:
        logger.warning("Failed to persist uploaded file bytes (file_id will be empty): %s", e)
        file_id = ""

    return {
        "text": text,
        "filename": file.filename,
        "chars": len(text),
        "file_id": file_id,
        "mime": mime,
        "size": len(content),
    }
