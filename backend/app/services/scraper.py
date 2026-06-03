"""
Web scraping service using Trafilatura for content extraction.
Fetches pages with httpx and extracts clean text using Trafilatura.
Supports parallel scraping via asyncio.gather.
"""

import asyncio
import logging
from typing import Optional

import httpx
import trafilatura

logger = logging.getLogger(__name__)

# Request headers to mimic a browser
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


async def scrape_url(url: str, timeout: int = 15) -> Optional[str]:
    """
    Fetch a URL and extract its main text content using Trafilatura.

    Args:
        url: The URL to scrape.
        timeout: Request timeout in seconds.

    Returns:
        Extracted text content, or None if extraction fails.
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=_HEADERS,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

        # Run Trafilatura extraction in thread executor (it's CPU-bound)
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            _extract_text,
            html,
            url,
        )

        if text and len(text.strip()) > 50:
            logger.info("Scraped %s: %d chars", url, len(text))
            return text.strip()
        else:
            logger.warning("Scraped %s but got minimal content", url)
            return None

    except httpx.TimeoutException:
        logger.warning("Scrape timeout for %s", url)
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("Scrape HTTP error for %s: %s", url, e.response.status_code)
        return None
    except Exception as e:
        logger.warning("Scrape failed for %s: %s", url, e)
        return None


def _extract_text(html: str, url: str) -> Optional[str]:
    """Extract text from HTML using Trafilatura (synchronous)."""
    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )
        return text
    except Exception as e:
        logger.warning("Trafilatura extraction error: %s", e)
        return None


async def scrape_urls(urls: list[str], timeout: int = 15) -> dict[str, Optional[str]]:
    """
    Scrape multiple URLs in parallel.

    Args:
        urls: List of URLs to scrape.
        timeout: Per-request timeout in seconds.

    Returns:
        Dict mapping URL → extracted text (or None on failure).
    """
    tasks = [scrape_url(url, timeout) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scraped = {}
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logger.warning("Scrape exception for %s: %s", url, result)
            scraped[url] = None
        else:
            scraped[url] = result

    successful = sum(1 for v in scraped.values() if v is not None)
    logger.info("Scraped %d/%d URLs successfully", successful, len(urls))

    return scraped
