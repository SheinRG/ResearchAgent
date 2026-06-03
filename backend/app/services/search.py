"""
DuckDuckGo web search service.
Uses the duckduckgo-search library (no API key required).
Runs synchronous search in a thread executor for async compatibility.
"""

import asyncio
import logging
from typing import Optional
from urllib.parse import urlparse

from duckduckgo_search import DDGS

from app.models.schemas import SearchResult
from app.config import get_settings

logger = logging.getLogger(__name__)


async def search_web(
    query: str,
    max_results: Optional[int] = None,
) -> list[SearchResult]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects with url, title, domain, favicon, snippet.
    """
    settings = get_settings()
    if max_results is None:
        max_results = settings.search_results_per_query

    try:
        # Run synchronous DuckDuckGo search in thread executor
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(
            None,
            _sync_search,
            query,
            max_results,
        )

        results = []
        seen_domains = set()

        for r in raw_results:
            url = r.get("href", r.get("link", ""))
            title = r.get("title", "")
            snippet = r.get("body", r.get("snippet", ""))

            if not url or not title:
                continue

            domain = urlparse(url).netloc.replace("www.", "")

            # Skip duplicate domains for diversity
            if domain in seen_domains:
                continue
            seen_domains.add(domain)

            favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"

            results.append(SearchResult(
                url=url,
                title=title,
                domain=domain,
                favicon=favicon,
                snippet=snippet[:300] if snippet else "",
            ))

        logger.info("DuckDuckGo search for '%s': %d results", query, len(results))
        return results

    except Exception as e:
        logger.error("DuckDuckGo search failed for '%s': %s", query, e)
        return []


def _sync_search(query: str, max_results: int) -> list[dict]:
    """Synchronous DuckDuckGo search (runs in executor)."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=max_results,
                safesearch="moderate",
            ))
            return results
    except Exception as e:
        logger.error("DuckDuckGo sync search error: %s", e)
        return []
