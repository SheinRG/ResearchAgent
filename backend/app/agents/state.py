"""
LangGraph state definition for the research agent.
Uses TypedDict with Annotated reducers for accumulating results across parallel nodes.
"""

from typing import TypedDict, Annotated, Any
from operator import add


def _replace(existing: Any, new: Any) -> Any:
    """Reducer that replaces the old value with the new one."""
    return new


def format_history(
    history: list[dict] | None,
    max_turns: int = 4,
    max_answer_chars: int = 600,
) -> str:
    """
    Render prior conversation turns into a compact transcript for prompts.

    Keeps the most recent ``max_turns`` exchanges and truncates each answer so
    follow-up prompts stay grounded in context without blowing the token budget.
    Returns an empty string when there is no usable history.
    """
    if not history:
        return ""

    lines: list[str] = []
    for turn in history[-max_turns:]:
        question = (turn.get("query") or "").strip()
        answer = (turn.get("answer") or "").strip()
        if len(answer) > max_answer_chars:
            answer = answer[:max_answer_chars].rstrip() + "…"
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")

    return "\n".join(lines)


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
    history: list[dict]                                 # Prior {query, answer} turns for follow-up context
    user_name: str                                      # What the user wants the agent to call them ("" = none)

    # --- Router / Planner Output ---
    mode: str                                           # "chat" (direct reply) or "research" (full pipeline)
    # When documents are attached, whether to also run web search to augment them.
    # False = answer from the uploaded doc(s) only; True = also hit the web.
    needs_web: bool

    # --- Planner Output ---
    sub_queries: list[str]                              # Decomposed sub-questions
    answer_format: dict                                 # Reasoned presentation format {type, reasoning, columns?}

    # --- Uploaded Document Input ---
    documents: list[dict]                               # [{name, text}] from the request; passed through unchanged
    document_chunks: list[dict]                         # Chunked doc text produced by researcher (single writer — no reducer needed)

    # --- Researcher Output (accumulated across parallel nodes) ---
    search_results: Annotated[list[dict], _merge_lists] # All search results (deduplicated)
    scraped_content: Annotated[list[dict], add]         # All scraped & chunked content
    ranked_chunks: list[dict]                           # Re-ranked top chunks
    images: list[dict]                                  # Image results for the original query (Images tab)

    # --- Synthesizer Output ---
    draft_answer: str                                   # Current synthesized answer
    citations: list[dict]                               # Extracted citations
    all_sources: list[dict]                             # Flattened unique sources for UI

    # --- Reflector Output ---
    reflection: dict                                    # Reflection analysis
    confidence: float                                   # Confidence score (0-1)

    # --- Control ---
    iteration: int                                      # Current reflection loop count
    phase: str                                          # Current agent phase (used by UI)
    follow_up_suggestions: list[str]                    # Generated follow-up questions
    error: str                                          # Error message if any

    # --- SSE Callback ---
    sse_callback: Any                                   # Async callback for SSE events
