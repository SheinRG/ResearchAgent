"""
Triage Node — Intent routing AND query planning in ONE LLM call.

Decides whether a message needs a normal conversational reply ("chat") or the
full web-research pipeline ("research"); and when it's research, decomposes the
question into searchable sub-queries and picks the best answer format — all in a
single structured response. Merging routing + planning removes a sequential LLM
round-trip from every research query.
"""

import logging
from datetime import date

from app.services.llm import get_llm_client
from app.agents.state import ResearchState, format_history

logger = logging.getLogger(__name__)

TRIAGE_SYSTEM = """You triage a user's message for an AI assistant named goon, and — when the message needs web research — you also plan that research, all in one JSON response.

STEP 1 — pick the mode:
- "chat": normal conversation or a simple request the assistant can answer well from its own knowledge — greetings and small talk (hi, hello, thanks, how are you), questions about the assistant itself (who are you, what can you do, your name), opinions, jokes, encouragement, casual advice, writing / rephrasing / translating / summarizing text the user gives you, basic math, and widely-known facts that do not depend on recent or live information.
- "research": anything that genuinely benefits from current information or web sources — news and recent events, live or real-world data (prices, stats, scores, weather), comparisons of specific real products/tools/services ("best X", "X vs Y"), specific people / companies / papers / places, how-tos where citations add real value, or any question where giving an outdated or made-up answer would matter.

Decide from the LATEST user message, using the prior conversation only for context. Lean toward "chat" for greetings and clearly casual or self-contained messages. When genuinely unsure, choose "research".

STEP 2 — ONLY when mode is "research", plan the research:
- Generate 2-4 focused sub-queries that together cover the full scope of the question. Each must be specific and searchable, target a different aspect, and prefer authoritative angles (official data, primary sources, expert analysis). Do NOT repeat the original question verbatim.
- For time-sensitive topics, anchor recency to the current year ({current_year}) — e.g. "... {current_year}" or "latest {current_year}".
- Follow-up handling: the question may rely on earlier conversation. Resolve references like "it", "they", "that", "the company" to the ACTUAL named entities. Every sub-query MUST be self-contained — it is sent to a web search engine with NO memory of the conversation, so spell out the real names, places, and topics instead of pronouns.

  Also decide the single best format for the final answer by reasoning about what the user is trying to accomplish and what is most useful and scannable — NOT by keyword matching:
  - "table" -> the answer is a set of MULTIPLE NAMED THINGS the user is choosing between or comparing — resources, tools, products, courses, services, sheets, websites, libraries, options, or entities — where each has attributes a chooser would weigh (price, topics, difficulty, ratings, pros/cons, specs, who it's best for). This is the RIGHT default for "best X", "top X", "recommended X", "which X should I use", and "X vs Y". Each thing becomes a row; pick 3-5 columns that matter for THIS query. Example: "best DSA sheets online" -> ["Sheet", "Topics covered", "No. of problems", "Cost", "Best for"].
  - "steps" -> a process, how-to, setup, or ordered ranking the user follows in sequence.
  - "list" -> discrete points, tips, reasons, takeaways, or facts that are NOT comparable named entities with shared columns. If the items are named options that could be compared, prefer "table".
  - "prose" -> a single fact, definition, explanation, cause/effect, or open-ended discussion.
  Decision rule: if you can imagine the answer as rows-and-columns where each row is one named option, choose "table". Fall back to "list" only when there is nothing to compare, and "prose" for a single-topic explanation.

For "chat" mode, set "sub_queries" to [] and "answer_format" to {{"type": "prose", "reasoning": "", "columns": []}}.

Respond ONLY with valid JSON in this exact format:
{{"mode": "chat|research", "sub_queries": ["sub-query 1", "sub-query 2"], "answer_format": {{"type": "table|list|steps|prose", "reasoning": "one short clause", "columns": ["Col A", "Col B"]}}}}
Note: "columns" is REQUIRED only when type is "table" (2-6 short header strings tailored to the query); use [] otherwise."""

TRIAGE_PROMPT = """{conversation}Latest user message: {query}

Respond with JSON only:"""

_ALLOWED_FORMATS = {"table", "list", "steps", "prose"}
_DEFAULT_FORMAT = {"type": "prose", "reasoning": "", "columns": []}


def _clean_format(raw: object) -> dict:
    """Validate and normalize the planner's answer_format object."""
    if not isinstance(raw, dict):
        return dict(_DEFAULT_FORMAT)

    fmt_type = raw.get("type")
    if fmt_type not in _ALLOWED_FORMATS:
        fmt_type = "prose"

    fmt_reasoning = raw.get("reasoning")
    if not isinstance(fmt_reasoning, str):
        fmt_reasoning = ""

    fmt_columns = raw.get("columns")
    if not isinstance(fmt_columns, list):
        fmt_columns = []
    else:
        fmt_columns = [str(c) for c in fmt_columns if str(c).strip()]

    # Columns are only meaningful for tables.
    if fmt_type != "table":
        fmt_columns = []

    return {"type": fmt_type, "reasoning": fmt_reasoning, "columns": fmt_columns}


def _clean_sub_queries(raw: object, query: str) -> list[str]:
    """Deduplicate, trim, and bound the sub-queries to 2-4 distinct entries."""
    if not isinstance(raw, list):
        return [query]

    seen: set[str] = set()
    clean: list[str] = []
    for q in raw:
        q_clean = str(q).strip()
        if q_clean and q_clean.lower() not in seen:
            seen.add(q_clean.lower())
            clean.append(q_clean)

    clean = clean[:4]
    if len(clean) < 2 and query.lower() not in seen:
        clean.append(query)
    return clean or [query]


async def router_node(state: ResearchState) -> dict:
    """Classify the message and, for research, plan sub-queries + format — one call."""
    query = state["query"]
    history = state.get("history", [])
    sse_callback = state.get("sse_callback")

    conversation = ""
    history_text = format_history(history)
    if history_text:
        conversation = (
            "This may be a FOLLOW-UP in an ongoing research conversation. Use the "
            "conversation below to resolve references, then make every sub-query "
            "self-contained.\n\n"
            f"Conversation so far:\n{history_text}\n\n"
        )

    mode = "research"
    sub_queries: list[str] = []
    answer_format = dict(_DEFAULT_FORMAT)

    try:
        llm = get_llm_client()
        result = await llm.generate_structured(
            prompt=TRIAGE_PROMPT.format(conversation=conversation, query=query),
            system=TRIAGE_SYSTEM.format(current_year=date.today().year),
            temperature=0.2,
        )

        candidate = str(result.get("mode", "")).strip().lower()
        if candidate in ("chat", "research"):
            mode = candidate

        if mode == "research":
            sub_queries = _clean_sub_queries(result.get("sub_queries"), query)
            answer_format = _clean_format(result.get("answer_format"))
    except Exception as e:
        # Fail safe: on any error, fall back to researching the original query.
        logger.warning("Triage failed, defaulting to research: %s", e)
        sub_queries = [query]

    # When the user uploaded documents, force the full research pipeline so the
    # cited synthesizer path runs — even for queries triage would call "chat".
    # Sub-queries are still generated above so web results can supplement the docs.
    if state.get("documents") and mode == "chat":
        logger.info("Triage: overriding mode chat→research because documents are present")
        mode = "research"
        if not sub_queries:
            sub_queries = _clean_sub_queries(None, query)

    logger.info(
        "Triage: mode=%s, %d sub-queries, format=%s for: %s",
        mode, len(sub_queries), answer_format.get("type"), query[:80],
    )

    # Surface planning progress + sub-queries to the UI for research runs only;
    # chat runs stream their reply straight from the conversational node.
    if mode == "research" and sse_callback:
        await sse_callback("phase", {
            "phase": "planning",
            "message": "Breaking down your question...",
        })
        await sse_callback("sub_queries", {"queries": sub_queries})

    return {
        "mode": mode,
        "sub_queries": sub_queries,
        "answer_format": answer_format,
    }
