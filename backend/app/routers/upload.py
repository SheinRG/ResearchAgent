"""File upload and text extraction endpoint."""
import logging
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.services.file_processor import (
    extract_text,
    mime_for_filename,
    SUPPORTED_EXTENSIONS,
)
from app.routers.auth import get_current_user
from app.models.database import UploadedFile, get_session_factory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Read granularity for the size guard below. Peak memory per upload is bounded
# at MAX_FILE_SIZE + CHUNK_SIZE rather than "whatever the client chose to send".
CHUNK_SIZE = 64 * 1024


async def _read_within_limit(file: UploadFile, limit: int) -> bytes:
    """
    Read an upload into memory, aborting as soon as it exceeds `limit`.

    The obvious version — `await file.read()` then check `len()` — has already
    materialised the whole body before it can reject it, so a 500MB POST costs
    500MB of RAM on an instance that has far less. Reading in chunks lets us
    stop at the first byte past the limit instead of after the last one.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413, detail="File too large. Max size is 5MB."
            )
        chunks.append(chunk)
    return b"".join(chunks)


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
    content = await _read_within_limit(file, MAX_FILE_SIZE)

    try:
        text = extract_text(file.filename or "", content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("File extraction error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to extract text from file.")

    # Derived from the validated extension, never from file.content_type — the
    # client controls that header and this value is served back verbatim.
    mime = mime_for_filename(file.filename or "")
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
