"""
LangGraph state definition for the research agent.
Uses TypedDict with Annotated reducers for accumulating results across parallel nodes.
"""

from typing import TypedDict, Annotated, Any
from operator import add


def _replace(existing: Any, new: Any) -> Any:
    """Reducer that replaces the old value with the new one."""
    return new


def _merge_lists(existing: list, new: list) -> list:
    """Reducer that merges two lists, deduplicating by URL for search results."""
    if not existing:
        return new
    if not new:
        return existing

    # If items have 'url' attribute, deduplicate
    merged = list(existing)
    existing_urls = set()
    for item in existing:
        if isinstance(item, dict) and 'url' in item:
            existing_urls.add(item['url'])
        elif hasattr(item, 'url'):
            existing_urls.add(item.url)

    for item in new:
        if isinstance(item, dict) and 'url' in item:
            if item['url'] not in existing_urls:
                merged.append(item)
                existing_urls.add(item['url'])
        elif hasattr(item, 'url'):
            if item.url not in existing_urls:
                merged.append(item)
                existing_urls.add(item.url)
        else:
            merged.append(item)

    return merged


class ResearchState(TypedDict, total=False):
    """
    Shared state object for the LangGraph research agent.

    Fields marked with Annotated use reducers to handle concurrent updates
    from parallel researcher nodes.
    """
    # --- Input ---
    query: str                                          # Original user question
    max_iterations: int                                 # Max reflection loops allowed

    # --- Planner Output ---
    sub_queries: list[str]                              # Decomposed sub-questions

    # --- Researcher Output (accumulated across parallel nodes) ---
    search_results: Annotated[list[dict], _merge_lists] # All search results (deduplicated)
    scraped_content: Annotated[list[dict], add]         # All scraped & chunked content
    ranked_chunks: list[dict]                           # Re-ranked top chunks

    # --- Synthesizer Output ---
    draft_answer: str                                   # Current synthesized answer
    citations: list[dict]                               # Extracted citations
    all_sources: list[dict]                             # Flattened unique sources for UI

    # --- Reflector Output ---
    reflection: dict                                    # Reflection analysis
    confidence: float                                   # Confidence score (0-1)

    # --- Control ---
    iteration: int                                      # Current loop count
    phase: str                                          # Current phase for UI
    follow_up_suggestions: list[str]                    # Generated follow-up questions
    error: str                                          # Error message if any

    # --- SSE Callback ---
    sse_callback: Any                                   # Async callback for SSE events
