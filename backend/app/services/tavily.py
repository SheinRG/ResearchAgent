"""
Tavily search-and-read service.

Tavily returns web results AND cleaned page content in a single API call, which
replaces the separate "Serper search → Trafilatura scrape" steps. One round-trip
per sub-query, with reliable content extraction for JS-heavy pages the raw
scraper returns empty for.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.config import get_settings
from app.services.retry import with_retries
from app.services.usage import record_search

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"

# Cap per-page content so a single huge page can't blow up chunking/reranking
# cost. The reranker only keeps the top-k chunks anyway.
_MAX_CONTENT_CHARS = 12000

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """Get or create a persistent HTTP client for reusing connections."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=get_settings().tavily_timeout)
    return _client


async def close_tavily() -> None:
    """Close the shared Tavily client (call on shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Tavily client closed")


async def tavily_search(
    query: str,
    max_results: Optional[int] = None,
    search_depth: Optional[str] = None,
) -> list[dict]:
    """
    Search the web with Tavily and return results that already include content.

    Transient failures (timeouts, 429, 5xx) are retried with exponential backoff
    under a wall-clock deadline. Best-effort after that: an exhausted budget
    returns an empty list so a single bad sub-query never breaks the research
    run — but it now takes a sustained outage rather than one dropped packet.

    Returns:
        List of dicts shaped as
        {url, title, domain, favicon, snippet, content}. `content` is the cleaned
        full-page text (falls back to the snippet); `snippet` is a short excerpt.
    """
    settings = get_settings()
    if max_results is None:
        max_results = settings.search_results_per_query
    if search_depth is None:
        search_depth = settings.tavily_search_depth

    headers = {
        "Authorization": f"Bearer {settings.tavily_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        # Full-page text only when explicitly enabled — it roughly triples latency.
        "include_raw_content": settings.tavily_include_raw_content,
        "include_answer": False,
        "include_images": False,
    }

    async def _request() -> dict:
        response = await _get_client().post(TAVILY_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    try:
        data = await with_retries(
            _request,
            label=f"Tavily search for '{query[:50]}'",
            attempts=settings.search_max_retries + 1,
            base_delay=settings.search_retry_base_delay,
            deadline=settings.search_retry_deadline,
            # Tavily bills per search, and a timeout does not mean the search
            # did not happen on their side. Each attempt is counted at the
            # configured depth so the guard sees an outage's real cost.
            on_attempt=lambda: record_search(search_depth),
        )

        results: list[dict] = []
        for r in data.get("results", []):
            url = r.get("url", "")
            title = r.get("title", "")
            if not url or not title:
                continue

            domain = urlparse(url).netloc.replace("www.", "")
            snippet = r.get("content", "") or ""
            # Prefer the full cleaned page text; fall back to the short excerpt.
            content = (r.get("raw_content") or snippet or "")[:_MAX_CONTENT_CHARS]

            results.append({
                "url": url,
                "title": title,
                "domain": domain,
                "favicon": f"https://www.google.com/s2/favicons?domain={domain}&sz=32",
                "snippet": snippet[:300],
                "content": content,
            })

        logger.info("Tavily search for '%s': %d results", query, len(results))
        return results

    except httpx.TimeoutException:
        logger.error("Tavily search timed out for '%s'", query)
        return []
    except httpx.HTTPStatusError as e:
        logger.error("Tavily HTTP error for '%s': %s", query, e.response.status_code)
        return []
    except Exception as e:
        logger.error("Tavily search failed for '%s': %s", query, e)
        return []
