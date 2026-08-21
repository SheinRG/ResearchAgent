"""
Citation extraction and formatting utilities.
Extracts [1], [2], etc. markers from generated text and maps them to source URLs.
"""

import re
import logging
from urllib.parse import urlparse
from typing import Optional

from app.models.schemas import Citation, SearchResult

logger = logging.getLogger(__name__)

# Pattern to match citation markers like [1], [2], [3]
CITATION_PATTERN = re.compile(r'\[(\d+)\]')
MAX_CLAIM_LENGTH = 300


def select_evidence(
    ranked_chunks: list[dict],
    max_sources: int = 8,
    max_chunks: int = 12,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Choose which chunks become the numbered evidence, and whose they are.

    Split out of :func:`build_cited_context` so the exact text behind marker
    ``[i]`` can be recovered without re-deriving these rules. The eval harness
    needs that to pair an answer sentence with the evidence the model actually
    read for the source it cited (see ``evals/capture.py``): the full chunk text
    is never persisted anywhere, so it only exists while the graph is running.
    A second implementation of this loop would drift from this one, and the
    resulting dataset would be quietly mislabelled rather than obviously wrong.

    Ordering is relevance order (the order chunks come back from re-ranking),
    deduplicated by URL. The caps interact deliberately: once ``max_sources``
    distinct sources are held, later chunks from *new* sources are skipped but
    chunks from sources already in the list keep filling, up to ``max_chunks``
    in total. That keeps the evidence concentrated on the best sources rather
    than spreading it thin.

    Args:
        ranked_chunks: Re-ranked chunk dicts (text + source_url/title/domain).
        max_sources: Maximum distinct sources to surface and number.
        max_chunks: Maximum total chunks across all sources.

    Returns:
        Tuple of (source URLs in citation order, {url: [chunk text, ...]}).
        The URL at index ``i`` is the source cited as ``[i + 1]``.
    """
    order: list[str] = []          # source URLs in citation order
    chunks_by_url: dict[str, list[str]] = {}
    total_chunks = 0

    for chunk in ranked_chunks:
        if total_chunks >= max_chunks:
            break
        url = chunk.get("source_url", "")
        text = (chunk.get("text") or "").strip()
        if not url or not text:
            continue
        if url not in chunks_by_url:
            if len(order) >= max_sources:
                continue  # source cap reached; skip new sources but keep filling existing
            order.append(url)
            chunks_by_url[url] = []
        chunks_by_url[url].append(text)
        total_chunks += 1

    return order, chunks_by_url


def build_cited_context(
    ranked_chunks: list[dict],
    search_results: list[dict],
    max_sources: int = 8,
    max_chunks: int = 12,
) -> tuple[list[dict], str]:
    """
    Build a canonical source list and matching prompt context from ranked chunks.

    This is the single source of truth for citation numbering: source ``[i]`` in
    the returned context corresponds exactly to ``cited_sources[i - 1]``. The same
    list is sent to the UI and used to resolve citations, so a ``[1]`` marker in
    the answer always points at the same source on screen.

    Sources are ordered by relevance (the order chunks appear after re-ranking),
    deduplicated by URL, and capped at ``max_sources``. Each source carries every
    one of its top chunks (within the global ``max_chunks`` budget) so the model
    sees consolidated evidence per source.

    Args:
        ranked_chunks: Re-ranked chunk dicts (text + source_url/title/domain).
        search_results: Original search results, used to enrich favicon/snippet.
        max_sources: Maximum distinct sources to surface and number.
        max_chunks: Maximum total chunks to include across all sources.

    Returns:
        Tuple of (cited_sources, context_string).
    """
    enrich = {r.get("url"): r for r in search_results if r.get("url")}
    order, chunks_by_url = select_evidence(ranked_chunks, max_sources, max_chunks)

    cited_sources: list[dict] = []
    context_blocks: list[str] = []

    for idx, url in enumerate(order, 1):
        sample = next((c for c in ranked_chunks if c.get("source_url") == url), {})
        enriched = enrich.get(url, {})
        domain = enriched.get("domain") or sample.get("source_domain") or urlparse(url).netloc.replace("www.", "")
        title = enriched.get("title") or sample.get("source_title") or domain
        # Uploaded-file sources use a file:// URL; Google's favicon service would
        # 404 on domain "Uploaded file", so leave their favicon blank instead.
        if url.startswith("file://"):
            favicon = enriched.get("favicon") or ""
        else:
            favicon = enriched.get("favicon") or f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
        snippet = enriched.get("snippet") or (chunks_by_url[url][0][:200] if chunks_by_url[url] else "")

        cited_sources.append({
            "url": url,
            "title": title,
            "domain": domain,
            "favicon": favicon,
            "snippet": snippet,
        })

        evidence = "\n".join(chunks_by_url[url])
        context_blocks.append(f"[{idx}] {title} ({domain})\n{evidence}")

    context = "\n\n".join(context_blocks) if context_blocks else "No source content available."
    return cited_sources, context


def _source_field(source, field: str) -> str:
    """Read a field from a source that may be a dict or a SearchResult."""
    if isinstance(source, dict):
        return source.get(field, "")
    return getattr(source, field, "")


def extract_citations_detailed(
    text: str,
    sources: list,
) -> tuple[list[Citation], list[int]]:
    """
    Extract citation markers from text and map them to source URLs, reporting
    any markers that point at a source the model was never given.

    The ``sources`` list must use the same ordering that was presented to the
    model (see :func:`build_cited_context`), so marker ``[i]`` resolves to
    ``sources[i - 1]``.

    A marker outside that range — a ``[7]`` in an answer built from 4 sources —
    is the clearest fabrication signal available, so it is returned to the
    caller rather than only logged. Callers surface the count to the user and
    persist it with the answer.

    Args:
        text: The generated answer text containing [1], [2] markers.
        sources: Canonical ordered sources (dicts or SearchResult objects).

    Returns:
        Tuple of (citations for valid markers, sorted unique invalid markers).
    """
    if not text or not sources:
        return [], []

    # Find all citation indices in the text
    matches = CITATION_PATTERN.findall(text)
    seen_indices = set()
    citations = []
    # Markers pointing at a source number that was never given to the model.
    # These are dropped from the returned citations, but they are the clearest
    # signal available that the model invented a reference, so they are counted
    # and returned rather than discarded silently.
    invalid_indices: list[int] = []

    for match in matches:
        try:
            idx = int(match)
            if idx < 1 or idx > len(sources):
                invalid_indices.append(idx)
                continue
            if idx in seen_indices:
                continue
            seen_indices.add(idx)

            source = sources[idx - 1]  # Convert 1-indexed to 0-indexed

            # Extract the surrounding context for the claim
            claim = _extract_claim_context(text, idx)

            citations.append(Citation(
                index=idx,
                source_url=_source_field(source, "url"),
                source_title=_source_field(source, "title"),
                source_domain=_source_field(source, "domain"),
                claim=claim,
            ))
        except (ValueError, IndexError):
            continue

    unique_invalid = sorted(set(invalid_indices))
    if unique_invalid:
        logger.warning(
            "Answer cited %d marker(s) with no matching source (valid range 1-%d): %s "
            "— possible fabricated citation",
            len(invalid_indices),
            len(sources),
            unique_invalid,
        )

    logger.info(
        "Extracted %d unique citations from answer (%d invalid marker(s) dropped)",
        len(citations),
        len(invalid_indices),
    )
    return citations, unique_invalid


def extract_citations(
    text: str,
    sources: list,
) -> list[Citation]:
    """
    Extract citation markers from text and map them to source URLs.

    Thin wrapper over :func:`extract_citations_detailed` for callers that only
    need the resolved citations and not the fabrication signal.
    """
    citations, _invalid = extract_citations_detailed(text, sources)
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

    return claim[:MAX_CLAIM_LENGTH]


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
