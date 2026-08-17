"""
Serper.dev web search service.
Uses the Serper Google Search API for fast, reliable search results.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.models.schemas import SearchResult
from app.config import get_settings
from app.services.retry import with_retries
from app.services.usage import record_search

logger = logging.getLogger(__name__)

SERPER_ENDPOINT = "https://google.serper.dev/search"
SERPER_IMAGES_ENDPOINT = "https://google.serper.dev/images"

_client: Optional[httpx.AsyncClient] = None

def _get_client() -> httpx.AsyncClient:
    """Get or create a persistent HTTP client for reusing connections."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30)
    return _client


async def search_web(
    query: str,
    max_results: Optional[int] = None,
) -> list[SearchResult]:
    """
    Search the web using the Serper.dev Google Search API.

    Transient failures (timeouts, 429, 5xx) are retried with exponential
    backoff. Still best-effort at the end of that: an exhausted retry budget
    yields an empty list, because one dead sub-query should degrade a research
    run rather than fail it. What changed is that it is no longer *one* blip
    away — and every attempt is counted as billed.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects with url, title, domain, favicon, snippet.
    """
    settings = get_settings()
    if max_results is None:
        max_results = settings.search_results_per_query

    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": max_results,
    }

    async def _request() -> dict:
        response = await _get_client().post(
            SERPER_ENDPOINT,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    try:
        data = await with_retries(
            _request,
            label=f"Serper search for '{query[:50]}'",
            attempts=settings.search_max_retries + 1,
            base_delay=settings.search_retry_base_delay,
            deadline=settings.search_retry_deadline,
            # Serper bills the lookup whether or not the response reaches us, so
            # a retry is a second credit. Counting per attempt keeps the budget
            # guard honest about what an outage actually costs.
            on_attempt=lambda: record_search("basic"),
        )

        raw_results = data.get("organic", [])
        results = []
        domain_counts: dict[str, int] = {}

        for r in raw_results:
            url = r.get("link", "")
            title = r.get("title", "")
            snippet = r.get("snippet", "")

            if not url or not title:
                continue

            domain = urlparse(url).netloc.replace("www.", "")

            # Cap results per domain for diversity, but keep up to 2 so an
            # authoritative site isn't reduced to a single page.
            if domain_counts.get(domain, 0) >= 2:
                continue
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"

            results.append(SearchResult(
                url=url,
                title=title,
                domain=domain,
                favicon=favicon,
                snippet=snippet[:300] if snippet else "",
            ))

        logger.info("Serper search for '%s': %d results", query, len(results))
        return results

    except httpx.TimeoutException:
        logger.error("Serper search timed out for '%s'", query)
        return []
    except httpx.HTTPStatusError as e:
        logger.error("Serper HTTP error for '%s': %s", query, e.response.status_code)
        return []
    except Exception as e:
        logger.error("Serper search failed for '%s': %s", query, e)
        return []


async def search_images(query: str, max_results: int = 10) -> list[dict]:
    """
    Search for images using the Serper.dev Google Images API.

    Best-effort and resilient: any timeout, HTTP error, or unexpected failure
    returns an empty list so an image lookup can never break a research run.

    Deliberately *not* retried, unlike ``search_web``. Images are decorative —
    they populate a tab the user may never open — and a retry spends another
    billable lookup. Failing fast here leaves that credit for an answer.

    Args:
        query: The search query string.
        max_results: Maximum number of image results to return.

    Returns:
        List of dicts shaped as
        {url, thumbnail, title, source, domain}. Empty on any failure.
    """
    settings = get_settings()

    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": max_results,
    }

    try:
        client = _get_client()
        response = await client.post(
            SERPER_IMAGES_ENDPOINT,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        raw_images = data.get("images", [])
        images: list[dict] = []

        for item in raw_images:
            image_url = item.get("imageUrl", "")
            if not image_url:
                continue

            # The originating page the image was found on.
            source = item.get("link", "")
            domain = item.get("domain", "")
            if not domain and source:
                domain = urlparse(source).netloc.replace("www.", "")

            images.append({
                "url": image_url,
                "thumbnail": item.get("thumbnailUrl", "") or image_url,
                "title": item.get("title", ""),
                "source": source,
                "domain": domain,
            })

        logger.info("Serper image search for '%s': %d results", query, len(images))
        return images

    except httpx.TimeoutException:
        logger.error("Serper image search timed out for '%s'", query)
        return []
    except httpx.HTTPStatusError as e:
        logger.error("Serper image HTTP error for '%s': %s", query, e.response.status_code)
        return []
    except Exception as e:
        logger.error("Serper image search failed for '%s': %s", query, e)
        return []
