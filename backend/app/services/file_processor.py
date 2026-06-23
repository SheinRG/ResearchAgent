"""Extract plain text from uploaded files (txt, md, pdf, docx)."""
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_CHARS = 12000


def extract_text(filename: str, content: bytes) -> str:
    """
    Extract plain text from file bytes. Returns up to MAX_CHARS characters.
    Raises ValueError for unsupported types.
    Raises RuntimeError if extraction fails.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if ext in (".txt", ".md"):
            text = content.decode("utf-8", errors="replace")

        elif ext == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            text = "\n".join(pages)

        elif ext == ".docx":
            from docx import Document

            doc = Document(io.BytesIO(content))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            text = "\n".join(paragraphs)

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {filename}: {e}") from e

    text = text.strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n\n[...truncated at {MAX_CHARS} chars]"
    return text
