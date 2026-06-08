"""
Citation extraction and formatting utilities.
Extracts [1], [2], etc. markers from generated text and maps them to source URLs.
"""

import re
import logging
from typing import Optional

from app.models.schemas import Citation, SearchResult

logger = logging.getLogger(__name__)

# Pattern to match citation markers like [1], [2], [3]
CITATION_PATTERN = re.compile(r'\[(\d+)\]')


def extract_citations(
    text: str,
    sources: list[SearchResult],
) -> list[Citation]:
    """
    Extract citation markers from text and map them to source URLs.

    Args:
        text: The generated answer text containing [1], [2] markers.
        sources: The list of source search results (1-indexed mapping).

    Returns:
        List of Citation objects for all valid markers found.
    """
    if not text or not sources:
        return []

    # Find all citation indices in the text
    matches = CITATION_PATTERN.findall(text)
    seen_indices = set()
    citations = []

    for match in matches:
        try:
            idx = int(match)
            if idx < 1 or idx > len(sources):
                continue
            if idx in seen_indices:
                continue
            seen_indices.add(idx)

            source = sources[idx - 1]  # Convert 1-indexed to 0-indexed

            # Extract the surrounding context for the claim
            claim = _extract_claim_context(text, idx)

            citations.append(Citation(
                index=idx,
                source_url=source.url,
                source_title=source.title,
                source_domain=source.domain,
                claim=claim,
            ))
        except (ValueError, IndexError):
            continue

    logger.info("Extracted %d unique citations from answer", len(citations))
    return citations


def _extract_claim_context(text: str, citation_idx: int) -> str:
    """
    Extract the sentence or clause containing a citation marker.

    Args:
        text: The full text.
        citation_idx: The citation index (e.g., 1 for [1]).

    Returns:
        The surrounding context string.
    """
    marker = f"[{citation_idx}]"
    pos = text.find(marker)
    if pos == -1:
        return ""

    # Find sentence boundaries around the marker
    # Look backwards for sentence start (up to 300 characters before the marker)
    start = pos
    for i in range(pos - 1, max(pos - 300, -1), -1):
        if i < 0:
            start = 0
            break
        # Split on standard punctuation or newlines
        if text[i] in '.!?\n' and i < pos - 2:
            start = i + 1
            break
    else:
        start = max(0, pos - 150)

    # Look forwards for sentence end (up to 300 characters after the marker)
    end = pos + len(marker)
    for i in range(end, min(end + 300, len(text))):
        if text[i] in '.!?\n':
            end = i + 1
            break
    else:
        end = min(len(text), pos + 200)

    claim = text[start:end].strip()
    # Remove all citation markers from the claim text to keep it clean
    claim = re.sub(r'\[\d+\]', '', claim).strip()

    return claim[:300]  # Limit claim length to 300 characters to keep citations concise


def format_sources_for_prompt(sources: list[SearchResult]) -> str:
    """
    Format source search results into a numbered list for the LLM prompt.

    Args:
        sources: List of search results.

    Returns:
        Formatted string with numbered sources and their snippets.
    """
    if not sources:
        return "No sources available."

    lines = []
    for i, source in enumerate(sources, 1):
        lines.append(f"[{i}] {source.title} ({source.domain})")
        if source.snippet:
            lines.append(f"    {source.snippet[:200]}")
        lines.append(f"    URL: {source.url}")
        lines.append("")

    return "\n".join(lines)


def format_chunks_for_prompt(
    chunks: list,
    max_chunks: int = 10,
) -> str:
    """
    Format ranked chunks into a context string for the LLM.

    Args:
        chunks: List of RankedChunk objects.
        max_chunks: Maximum number of chunks to include.

    Returns:
        Formatted context string with source attribution.
    """
    if not chunks:
        return "No source content available."

    lines = []
    seen_sources = {}
    source_idx = 0

    for chunk in chunks[:max_chunks]:
        url = chunk.source_url
        if url not in seen_sources:
            source_idx += 1
            seen_sources[url] = source_idx

        idx = seen_sources[url]
        lines.append(f"--- Source [{idx}]: {chunk.source_title} ({chunk.source_domain}) ---")
        lines.append(chunk.text)
        lines.append("")

    return "\n".join(lines)
