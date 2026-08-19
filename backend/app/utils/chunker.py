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

    # Split into paragraphs first. The splitters below return *non-overlapping*
    # chunks, and the overlap is stitched on once afterwards. Applying it inside
    # the recursion instead re-glued a neighbour tail that a nested split had
    # already added, so oversized paragraphs came back with the join text
    # duplicated inside the chunk.
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    chunks = _recursive_split(text, separators, chunk_size)
    chunks = _add_overlap(chunks, chunk_overlap)

    # Drop very short chunks (under 30 chars) to avoid noise
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

    logger.debug("Chunked %d chars → %d chunks (size=%d, overlap=%d)",
                 len(text), len(chunks), chunk_size, chunk_overlap)

    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
) -> list[str]:
    """
    Recursively split text using a hierarchy of separators.

    Returns non-overlapping chunks. Overlap is a presentation concern applied
    once by _add_overlap, because this function calls itself and any overlap
    added here would be re-applied by every enclosing level.
    """
    if len(text) <= chunk_size:
        return [text]

    # Find the best separator
    best_sep = None
    for sep in separators:
        if sep in text:
            best_sep = sep
            break

    if best_sep is None:
        # No separator left to respect: cut at fixed boundaries. Overlap is 0
        # here for the same reason as above — the caller stitches it on.
        return _force_split(text, chunk_size, 0)

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
                current_chunk = ""

            # If this single part is too large, recursively split it
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(best_sep) + 1:]
                if remaining_seps:
                    chunks.extend(_recursive_split(part, remaining_seps, chunk_size))
                else:
                    chunks.extend(_force_split(part, chunk_size, 0))
            else:
                current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _add_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    """
    Prepend each chunk with the tail of the one before it.

    Applied exactly once, over the final flat chunk list, so a chunk carries a
    single neighbour tail no matter how many levels of recursion produced it.
    """
    if chunk_overlap <= 0 or len(chunks) < 2:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        tail = prev[-chunk_overlap:] if len(prev) >= chunk_overlap else prev
        overlapped.append(tail + " " + chunks[i])
    return overlapped


def _force_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Force-split text at fixed boundaries when no separator works."""
    chunks = []
    start = 0

    # Each iteration must move `start` forward, otherwise the loop never ends.
    # Config drift (an overlap set at or above the chunk size) would otherwise
    # hang the whole request on a long page, so clamp the stride to >= 1 char.
    stride = max(1, chunk_size - chunk_overlap)

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += stride

    return chunks
