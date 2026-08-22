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

# Rewritten 2026-08-22 for openai/gpt-oss-20b, and cut from ~1,360 tokens to
# roughly half that. Two reasons, and the second is the one that bites:
#
# 1. The old chat lane said chat was right for "widely-known facts that do not
#    depend on recent or live information". gpt-oss-20b read that literally and
#    sent "what is retrieval augmented generation", "what causes the northern
#    lights" and "how do I reduce a docker image size" to chat -- 6 of the 20
#    eval queries. That is the wrong lane for this product: an answer with no
#    sources is the one thing goon is not for. Chat is now only greetings, small
#    talk, questions about the assistant, and text the user supplied.
#
# 2. Groq's free tier allows 8,000 tokens per MINUTE. At 1,360 tokens of system
#    prompt plus the message, that was about four triage calls a minute for the
#    whole service -- and the eval harness, running four queries at once, spent
#    most of its wall clock in 429 backoff rather than in the model. Every token
#    cut here is throughput for every request.
TRIAGE_SYSTEM = """You triage a message for a research assistant named goon and, when it needs the web, plan that research — one JSON response.

STEP 1 — mode:
- "chat": ONLY greetings and small talk, questions about the assistant itself (who are you, what can you do), thanks, jokes, and operations on text the user has already given you (rephrase, translate, summarize, fix). Basic arithmetic counts too.
- "research": EVERYTHING else, including questions you could answer from your own knowledge. Definitions, explanations, how-tos, history, comparisons, news, live data, people, companies, products. If the user is asking to LEARN something rather than chat, it is research — goon's whole point is that answers carry sources, so answering from memory is a failure, not a shortcut.
When unsure, choose "research".

STEP 2 — research only, plan it:
- 2-4 focused sub-queries covering the question's full scope. Each specific, searchable, targeting a different aspect. Never repeat the question verbatim.
- Anchor time-sensitive topics to {current_year} (e.g. "... {current_year}", "latest {current_year}").
- Follow-ups: resolve "it", "they", "that" to the real named entities. Each sub-query goes to a search engine with NO memory of this conversation, so it must stand alone.

Also pick the answer format by asking what is most useful to read:
- "table" — MULTIPLE NAMED THINGS being compared or chosen between (tools, products, services, courses, libraries), each with attributes worth weighing. The right default for "best X", "top X", "X vs Y", "which X should I use". Each thing is a row; pick 3-5 columns that matter here.
- "steps" — a process, setup, how-to, or ordered sequence the user follows.
- "list" — discrete points, tips, reasons or takeaways that are NOT comparable named entities.
- "prose" — one fact, definition, explanation, or open-ended discussion.
Rule of thumb: if the answer looks like rows-and-columns with one named option per row, use "table". Use "list" when there is nothing to compare, "prose" for a single topic.

For "chat": sub_queries [] and answer_format {{"type": "prose", "reasoning": "", "columns": []}}.

STEP 3 — time_sensitive (both modes): true if a correct answer changes over time (news, prices, scores, "latest", versions, rankings); false for definitions, history, established facts, stable how-tos. Unsure -> true. This only sets cache lifetime; it never changes what gets researched.

STEP 4 — needs_web (only when documents are attached): false when the question is about or answerable from the attached document(s) — summarize, explain, extract, Q&A. That is the DEFAULT with a document present. true ONLY when answering needs information beyond the document. With no documents attached this is ignored.

Respond ONLY with valid JSON:
{{"mode": "chat|research", "sub_queries": ["sub-query 1", "sub-query 2"], "answer_format": {{"type": "table|list|steps|prose", "reasoning": "one short clause", "columns": ["Col A", "Col B"]}}, "time_sensitive": true|false, "needs_web": true|false}}
"columns" is REQUIRED only for type "table" (2-6 short headers); use [] otherwise."""

TRIAGE_PROMPT = """{conversation}{documents_note}Latest user message: {query}

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

    # Determine whether documents are attached so we can inject the right context
    # into the triage prompt and apply the correct needs_web default.
    has_documents = bool(state.get("documents"))

    conversation = ""
    history_text = format_history(history)
    if history_text:
        conversation = (
            "This may be a FOLLOW-UP in an ongoing research conversation. Use the "
            "conversation below to resolve references, then make every sub-query "
            "self-contained.\n\n"
            f"Conversation so far:\n{history_text}\n\n"
        )

    # Inject a note about attached documents so the LLM can judge needs_web.
    documents_note = (
        "The user has ATTACHED one or more documents to this message.\n\n"
        if has_documents else ""
    )

    mode = "research"
    sub_queries: list[str] = []
    answer_format = dict(_DEFAULT_FORMAT)
    # Whether the answer goes stale. Only controls cache lifetime, never what
    # gets researched. Defaults to True so a triage failure can never cause a
    # live-data question to be served from an old cache entry.
    time_sensitive = True
    # Default: when docs are present, skip web unless triage explicitly enables it;
    # when no docs, web always runs (needs_web=True is irrelevant but harmless).
    needs_web: bool = not has_documents

    try:
        llm = get_llm_client()
        result = await llm.generate_structured(
            prompt=TRIAGE_PROMPT.format(
                conversation=conversation,
                documents_note=documents_note,
                query=query,
            ),
            system=TRIAGE_SYSTEM.format(current_year=date.today().year),
            temperature=0.2,
            stage="triage",
        )

        candidate = str(result.get("mode", "")).strip().lower()
        if candidate in ("chat", "research"):
            mode = candidate

        if mode == "research":
            sub_queries = _clean_sub_queries(result.get("sub_queries"), query)
            answer_format = _clean_format(result.get("answer_format"))

        # Read it for both modes. A missing key means the model omitted it, in
        # which case the safe reading is "assume it goes stale".
        raw_time_sensitive = result.get("time_sensitive")
        if isinstance(raw_time_sensitive, bool):
            time_sensitive = raw_time_sensitive

        # Only read needs_web from triage when documents are actually attached;
        # default False (doc-only) so the LLM must opt in to web augmentation.
        if has_documents:
            needs_web = bool(result.get("needs_web", False))

    except Exception as e:
        # Fail safe: on any error, fall back to researching the original query.
        # When docs are present, stay doc-only (needs_web=False) so we don't
        # accidentally web-search a doc-referential question.
        logger.warning("Triage failed, defaulting to research: %s", e)
        sub_queries = [query]
        needs_web = not has_documents  # keep safe default

    # When the user uploaded documents, force the full research pipeline so the
    # cited synthesizer path runs — even for queries triage would call "chat".
    # Sub-queries are still generated above so web results can supplement when needed.
    if has_documents and mode == "chat":
        logger.info("Triage: overriding mode chat→research because documents are present")
        mode = "research"
        if not sub_queries:
            sub_queries = _clean_sub_queries(None, query)

    logger.info(
        "Triage: mode=%s, %d sub-queries, format=%s, time_sensitive=%s%s for: %s",
        mode, len(sub_queries), answer_format.get("type"), time_sensitive,
        f", needs_web={needs_web}" if has_documents else "",
        query[:80],
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
        "time_sensitive": time_sensitive,
        "needs_web": needs_web,
    }
