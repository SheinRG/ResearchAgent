"""
Reasoning trace — what the pipeline considered, and what it did with it.

The pipeline already decides all of this and then throws the paperwork away: the
reranker scores every passage, the top ones are packed into the prompt, and the
answer ends up citing a subset of those. The user only ever sees the last group.
This assembles the other two so the UI can show the shortlist alongside the
hires.

Each source lands in one of three states, which is the whole point of the view:

    cited       the answer referenced it with a [n] marker
    sent        it reached the model's prompt, which chose not to use it
    considered  it was ranked but did not make the prompt's budget

Purely derived — no extra model calls, no extra network, nothing recomputed.
"""

from __future__ import annotations

from typing import Any, Optional

# How many sources the trace describes. Sorted by score, so the tail this drops
# is the least relevant material; the counts above the list still reflect
# everything, so nothing is silently misreported.
MAX_TRACE_SOURCES = 25

# Enough of a passage to recognise it, not enough to bloat every stored turn.
PREVIEW_CHARS = 180


def _preview(text: str) -> str:
    text = " ".join((text or "").split())
    if len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS].rstrip() + "…"


def build_trace(
    *,
    sub_queries: list[str],
    all_chunks: list[dict],
    ranked_chunks: list[dict],
    cited_sources: list[dict],
    citations: list[Any],
    reranker_model: str = "",
) -> dict:
    """
    Assemble the trace payload for one research turn.

    Args:
        sub_queries: The questions triage decided to search.
        all_chunks: Every chunk that went into reranking (pre-selection).
        ranked_chunks: What the reranker kept, best first, with scores.
        cited_sources: The canonical numbered list handed to the model —
            index i here is marker [i+1] in the answer.
        citations: Resolved Citation objects from the finished answer.
        reranker_model: Which model did the ranking, for display.

    Returns:
        A JSON-serializable dict. Empty ``sources`` is valid and meaningful:
        it means retrieval found nothing to rank.
    """
    cited_urls: dict[str, int] = {}
    for citation in citations or []:
        url = getattr(citation, "source_url", None)
        index = getattr(citation, "index", None)
        if url and index and url not in cited_urls:
            cited_urls[url] = index

    # Sources that reached the prompt, in citation order.
    sent_urls = {s.get("url") for s in (cited_sources or []) if s.get("url")}

    # Collapse chunks to one row per source: a source with four strong passages
    # is one entry scored by its best, not four rows competing with each other.
    by_url: dict[str, dict] = {}
    for chunk in ranked_chunks or []:
        url = chunk.get("source_url") or ""
        if not url:
            continue
        score = float(chunk.get("score", 0.0) or 0.0)
        entry = by_url.get(url)
        if entry is None:
            by_url[url] = {
                "url": url,
                "title": chunk.get("source_title") or "",
                "domain": chunk.get("source_domain") or "",
                "score": score,
                "chunks": 1,
                "preview": _preview(chunk.get("text", "")),
            }
        else:
            entry["chunks"] += 1
            if score > entry["score"]:
                entry["score"] = score
                entry["preview"] = _preview(chunk.get("text", ""))

    sources: list[dict] = []
    for entry in by_url.values():
        url = entry["url"]
        citation_index = cited_urls.get(url)
        if citation_index:
            status = "cited"
        elif url in sent_urls:
            status = "sent"
        else:
            status = "considered"
        sources.append({**entry, "status": status, "citation_index": citation_index})

    # Best first. Cited sources win ties so the useful rows sit at the top.
    sources.sort(key=lambda s: (s["score"], s["status"] == "cited"), reverse=True)

    return {
        "sub_queries": list(sub_queries or []),
        "reranker_model": reranker_model,
        "counts": {
            "chunks_ranked": len(all_chunks or []),
            "chunks_kept": len(ranked_chunks or []),
            "sources_considered": len(by_url),
            "sources_sent": len(sent_urls),
            "sources_cited": len(cited_urls),
        },
        "sources": [
            {
                "url": s["url"],
                "title": s["title"],
                "domain": s["domain"],
                "score": round(s["score"], 4),
                "chunks": s["chunks"],
                "status": s["status"],
                "citation_index": s["citation_index"],
                "preview": s["preview"],
            }
            for s in sources[:MAX_TRACE_SOURCES]
        ],
    }


def is_empty(trace: Optional[dict]) -> bool:
    """Whether a trace has nothing worth showing (so the UI can hide the tab)."""
    if not trace:
        return True
    return not (trace.get("sources") or trace.get("sub_queries"))
