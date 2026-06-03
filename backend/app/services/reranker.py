"""
FlashRank re-ranker service for CPU-only relevance ranking.
Uses the nano model for fast, lightweight re-ranking of text chunks.
"""

import asyncio
import logging
from typing import Optional

from flashrank import Ranker, RerankRequest

from app.models.schemas import RankedChunk

logger = logging.getLogger(__name__)

# Singleton ranker instance
_ranker: Optional[Ranker] = None


def _get_ranker() -> Ranker:
    """Get or initialize the FlashRank ranker (singleton)."""
    global _ranker
    if _ranker is None:
        logger.info("Initializing FlashRank ranker (nano model)...")
        _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")
        logger.info("FlashRank ranker initialized successfully")
    return _ranker


async def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 10,
) -> list[RankedChunk]:
    """
    Re-rank text chunks by relevance to the query using FlashRank.

    Args:
        query: The search query to rank against.
        chunks: List of dicts with 'text', 'source_url', 'source_title', 'source_domain'.
        top_k: Number of top chunks to return.

    Returns:
        List of RankedChunk objects sorted by relevance score.
    """
    if not chunks:
        return []

    try:
        loop = asyncio.get_event_loop()
        ranked = await loop.run_in_executor(
            None,
            _sync_rerank,
            query,
            chunks,
            top_k,
        )
        return ranked
    except Exception as e:
        logger.error("Re-ranking failed: %s", e)
        # Fallback: return chunks as-is with default scores
        return [
            RankedChunk(
                text=c["text"],
                score=0.5,
                source_url=c["source_url"],
                source_title=c["source_title"],
                source_domain=c["source_domain"],
            )
            for c in chunks[:top_k]
        ]


def _sync_rerank(
    query: str,
    chunks: list[dict],
    top_k: int,
) -> list[RankedChunk]:
    """Synchronous re-ranking (runs in executor)."""
    ranker = _get_ranker()

    # Prepare passages for FlashRank
    passages = []
    for i, chunk in enumerate(chunks):
        passages.append({
            "id": i,
            "text": chunk["text"],
            "meta": {
                "source_url": chunk["source_url"],
                "source_title": chunk["source_title"],
                "source_domain": chunk["source_domain"],
            },
        })

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    ranked_chunks = []
    for result in results[:top_k]:
        meta = result.get("meta", {})
        if not meta:
            # Try to find original chunk by id
            orig_id = result.get("id", 0)
            if isinstance(orig_id, int) and orig_id < len(chunks):
                meta = {
                    "source_url": chunks[orig_id]["source_url"],
                    "source_title": chunks[orig_id]["source_title"],
                    "source_domain": chunks[orig_id]["source_domain"],
                }

        ranked_chunks.append(RankedChunk(
            text=result.get("text", ""),
            score=float(result.get("score", 0.0)),
            source_url=meta.get("source_url", ""),
            source_title=meta.get("source_title", ""),
            source_domain=meta.get("source_domain", ""),
        ))

    logger.info(
        "Re-ranked %d chunks → top %d (scores: %.3f - %.3f)",
        len(chunks),
        len(ranked_chunks),
        ranked_chunks[-1].score if ranked_chunks else 0,
        ranked_chunks[0].score if ranked_chunks else 0,
    )

    return ranked_chunks
