"""
Recursive text chunker for splitting scraped content into
overlapping chunks suitable for re-ranking.
"""

import logging

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks using recursive character splitting.

    Tries to split on paragraph boundaries first, then sentences, then
    arbitrary positions. Each chunk overlaps with the next by `chunk_overlap`
    characters for context continuity.

    Args:
        text: The full text to chunk.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]

    # Split into paragraphs first
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    chunks = _recursive_split(text, separators, chunk_size, chunk_overlap)

    # Filter out very short chunks
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

    logger.debug("Chunked %d chars → %d chunks (size=%d, overlap=%d)",
                 len(text), len(chunks), chunk_size, chunk_overlap)

    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if len(text) <= chunk_size:
        return [text]

    # Find the best separator
    best_sep = None
    for sep in separators:
        if sep in text:
            best_sep = sep
            break

    if best_sep is None:
        # Force split at chunk_size boundaries with overlap
        return _force_split(text, chunk_size, chunk_overlap)

    # Split on the best separator
    parts = text.split(best_sep)
    chunks = []
    current_chunk = ""

    for part in parts:
        candidate = current_chunk + best_sep + part if current_chunk else part

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # If this single part is too large, recursively split it
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(best_sep) + 1:]
                if remaining_seps:
                    sub_chunks = _recursive_split(part, remaining_seps, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    sub_chunks = _force_split(part, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
            else:
                current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    # Add overlaps
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:] if len(prev) >= chunk_overlap else prev
            overlapped.append(overlap_text + " " + chunks[i])
        return overlapped

    return chunks


def _force_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Force-split text at fixed boundaries when no separator works."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap  # Step back by overlap amount

    return chunks
