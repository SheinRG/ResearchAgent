"""
Researcher Node — Search → Scrape → Chunk → Rerank pipeline.
One researcher node runs per sub-query, executing in parallel via LangGraph Send API.
"""

import asyncio
import logging
from app.services.search import search_web
from app.services.scraper import scrape_urls
from app.services.reranker import rerank_chunks
from app.services.cache import cache_get, cache_set
from app.utils.chunker import chunk_text
from app.agents.state import ResearchState
from app.config import get_settings

logger = logging.getLogger(__name__)


async def _search_single_query(sub_query: str, max_results: int) -> tuple[str, list[dict]]:
    """Search a single sub-query, using cache when available.

    Returns:
        Tuple of (sub_query, results_list).
    """
    cached = await cache_get("search", sub_query)
    if cached:
        logger.info("Cache hit for search: %s", sub_query[:50])
        return sub_query, cached

    results_objs = await search_web(sub_query, max_results=max_results)
    results = [r.model_dump() for r in results_objs]
    await cache_set("search", sub_query, results)
    return sub_query, results


async def researcher_node(state: ResearchState) -> dict:
    """
    Researcher node: runs the full search → scrape → chunk → rerank pipeline
    for all sub-queries. Called once with all sub-queries, processes them in parallel.

    Args:
        state: Current research state with sub_queries.

    Returns:
        Updated state with search_results, scraped_content.
    """
    sub_queries = state.get("sub_queries", [])
    sse_callback = state.get("sse_callback")
    settings = get_settings()

    if not sub_queries:
        logger.warning("Researcher: no sub-queries to process")
        return {"search_results": [], "scraped_content": []}

    logger.info("Researcher: processing %d sub-queries", len(sub_queries))

    # Phase: Searching
    if sse_callback:
        await sse_callback("phase", {
            "phase": "searching",
            "message": f"Searching {len(sub_queries)} sub-questions across the web..."
        })

    # --- Step 1: Search all sub-queries IN PARALLEL ---
    search_tasks = [
        _search_single_query(sq, settings.search_results_per_query)
        for sq in sub_queries
    ]
    search_results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)

    all_search_results = []
    all_urls_to_scrape = []
    url_to_result = {}

    for result in search_results_per_query:
        if isinstance(result, Exception):
            logger.warning("Search task failed: %s", result)
            continue
        _sub_query, results = result
        for r in results:
            if r["url"] not in url_to_result:
                url_to_result[r["url"]] = r
                all_search_results.append(r)
                if len(all_urls_to_scrape) < settings.scrape_top_n * len(sub_queries):
                    all_urls_to_scrape.append(r["url"])

    # Send sources event to frontend
    if sse_callback and all_search_results:
        await sse_callback("sources", {
            "sources": all_search_results[:15]  # Send top 15 sources
        })

    logger.info("Researcher: found %d unique sources, scraping %d",
                len(all_search_results), len(all_urls_to_scrape))

    # Phase: Reading
    if sse_callback:
        await sse_callback("phase", {
            "phase": "reading",
            "message": f"Reading and analyzing {len(all_urls_to_scrape)} sources..."
        })

    # --- Step 2: Scrape URLs in parallel ---
    scraped_content = []
    urls_to_scrape = all_urls_to_scrape[:settings.scrape_top_n * len(sub_queries)]

    # Check cache for scraped content
    uncached_urls = []
    for url in urls_to_scrape:
        cached = await cache_get("scrape", url)
        if cached:
            scraped_content.append(cached)
        else:
            uncached_urls.append(url)

    # Scrape uncached URLs
    if uncached_urls:
        scraped = await scrape_urls(uncached_urls)
        for url, text in scraped.items():
            if text:
                result_info = url_to_result.get(url, {})
                content_entry = {
                    "url": url,
                    "text": text,
                    "title": result_info.get("title", ""),
                    "domain": result_info.get("domain", ""),
                }
                scraped_content.append(content_entry)
                await cache_set("scrape", url, content_entry)

    # --- Step 3: Chunk all scraped content IN PARALLEL ---
    all_chunks = []
    loop = asyncio.get_running_loop()

    async def _process_content(content):
        # Run CPU-bound chunking in a thread pool to avoid blocking the event loop
        chunks = await loop.run_in_executor(
            None, chunk_text, content["text"], settings.chunk_size, settings.chunk_overlap
        )
        return [
            {
                "text": chunk_text_str,
                "source_url": content["url"],
                "source_title": content.get("title", ""),
                "source_domain": content.get("domain", ""),
            }
            for chunk_text_str in chunks
        ]

    if scraped_content:
        chunk_tasks = [_process_content(c) for c in scraped_content]
        chunk_results = await asyncio.gather(*chunk_tasks)
        for res in chunk_results:
            all_chunks.extend(res)

    logger.info("Researcher: generated %d chunks from %d scraped pages",
                len(all_chunks), len(scraped_content))

    return {
        "search_results": all_search_results,
        "scraped_content": all_chunks,
    }


async def rerank_node(state: ResearchState) -> dict:
    """
    Re-rank all accumulated chunks by relevance to the original query.

    Args:
        state: Current state with scraped_content chunks.

    Returns:
        Updated state with ranked_chunks and all_sources.
    """
    query = state["query"]
    chunks = state.get("scraped_content", [])
    search_results = state.get("search_results", [])
    settings = get_settings()

    if not chunks:
        logger.warning("Reranker: no chunks to rank")
        return {"ranked_chunks": [], "all_sources": search_results}

    logger.info("Reranker: ranking %d chunks for query: %s", len(chunks), query[:80])

    ranked = await rerank_chunks(query, chunks, top_k=settings.rerank_top_k)

    ranked_dicts = [r.model_dump() for r in ranked]

    logger.info("Reranker: top chunk score=%.3f, bottom=%.3f",
                ranked_dicts[0]["score"] if ranked_dicts else 0,
                ranked_dicts[-1]["score"] if ranked_dicts else 0)

    return {
        "ranked_chunks": ranked_dicts,
        "all_sources": search_results,
    }
