"""
Synthesizer Node — Cited Markdown Answer Generation.
Receives top-ranked chunks, builds a canonical numbered source list, and
generates a comprehensive answer with [1], [2] markers that map exactly to
that list — streaming tokens via the (stronger) synthesis model.
"""

import logging
from datetime import date

from app.services.llm import get_llm_client
from app.utils.citations import build_cited_context, extract_citations
from app.agents.state import ResearchState, format_history
from app.config import get_settings

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM = """You are an expert research analyst. You write accurate, comprehensive, well-structured answers grounded strictly in the numbered sources provided.

CITATION RULES (critical):
- Support every factual claim with a citation marker like [1], [2] that refers to the numbered sources given to you.
- Place the citation immediately after the claim it supports. Combine markers when several sources agree: [1][3].
- Use ONLY the source numbers that appear in the provided sources. Never invent a source number.
- If the sources do not contain enough information to answer part of the question, say so explicitly instead of guessing. Do NOT fabricate facts, numbers, or sources.

WRITING RULES:
- Open with a 2-3 sentence direct answer to the question, then expand with detail.
- Use `##` section headings when the answer spans multiple themes; use bullet lists for enumerable points.
- Lead with specifics: concrete figures, dates, names, and findings drawn from the sources.
- When sources disagree, surface the disagreement and attribute each view.
- Be thorough but do not pad. Prefer information density over filler.
- Write in clean Markdown. Do not include a "Sources" or "References" list at the end — the UI renders citations from the [n] markers."""

SYNTHESIZER_PROMPT = """Today's date is {today}. Answer the question using ONLY the numbered sources below.
{conversation_context}
**Question:** {query}

**Sources:**
{context}

Write a thorough, well-cited Markdown answer now. Every factual claim must carry a [n] citation that matches a source number above."""

SYNTHESIZER_FOLLOWUP_GUIDANCE = (
    "\nThis is a follow-up question in an ongoing conversation. Interpret it in "
    "light of the exchange below and resolve references like \"it\" or \"they\" to "
    "the entities already discussed. Answer the NEW question directly — build on "
    "the prior context instead of repeating it.\n\n"
    "**Conversation so far:**\n{history}\n"
)


async def synthesizer_node(state: ResearchState) -> dict:
    """
    Synthesizer node: generates a cited markdown answer by streaming tokens.

    Args:
        state: Current research state with ranked_chunks.

    Returns:
        Updated state with draft_answer, citations, all_sources, phase.
    """
    query = state["query"]
    ranked_chunks = state.get("ranked_chunks", [])
    search_results = state.get("search_results", [])
    history = state.get("history", [])
    sse_callback = state.get("sse_callback")
    settings = get_settings()

    logger.info("Synthesizer: generating answer from %d chunks", len(ranked_chunks))

    # --- Build the canonical numbered source list + matching context ---
    cited_sources, context = build_cited_context(
        ranked_chunks,
        search_results,
        max_sources=settings.max_cited_sources,
        max_chunks=settings.rerank_top_k,
    )

    # Fall back to raw search results if re-ranking produced nothing usable.
    if not cited_sources and search_results:
        cited_sources = search_results[: settings.max_cited_sources]

    # No usable sources at all — return a clear message rather than asking the
    # model to answer from nothing (which invites hallucination).
    if not cited_sources:
        logger.warning("Synthesizer: no sources available, returning fallback message")
        message = (
            "I couldn't find reliable sources to answer this question. "
            "This can happen with very new, niche, or ambiguous topics — "
            "try rephrasing the question or making it more specific."
        )
        if sse_callback:
            await sse_callback("phase", {"phase": "writing", "message": "Synthesizing your answer..."})
            await sse_callback("token", {"token": message})
        return {
            "draft_answer": message,
            "citations": [],
            "all_sources": [],
            "phase": "writing",
        }

    # Phase: Writing
    if sse_callback:
        await sse_callback("phase", {
            "phase": "writing",
            "message": "Synthesizing your answer...",
        })
        # Authoritative source list — index i here == [i] in the answer.
        # `replace` tells the UI to swap its provisional list for this one.
        if cited_sources:
            await sse_callback("sources", {"sources": cited_sources, "replace": True})

    # Carry recent conversation into the prompt so a follow-up answer reads as a
    # coherent continuation, while staying grounded strictly in the sources.
    conversation_context = ""
    history_text = format_history(history, max_answer_chars=800)
    if history_text:
        conversation_context = SYNTHESIZER_FOLLOWUP_GUIDANCE.format(history=history_text)

    prompt = SYNTHESIZER_PROMPT.format(
        today=date.today().isoformat(),
        conversation_context=conversation_context,
        query=query,
        context=context,
    )

    try:
        llm = get_llm_client()
        full_answer = ""

        async for token in llm.generate_stream(
            prompt=prompt,
            system=SYNTHESIZER_SYSTEM,
            temperature=0.4,
            model=settings.groq_synth_model,
            max_tokens=settings.synth_max_tokens,
        ):
            full_answer += token
            if sse_callback:
                await sse_callback("token", {"token": token})

        # Citations resolve against the SAME canonical list shown to the model.
        citations = extract_citations(full_answer, cited_sources)
        citation_dicts = [c.model_dump() for c in citations]

        logger.info(
            "Synthesizer: generated %d char answer, %d sources, %d citations",
            len(full_answer), len(cited_sources), len(citation_dicts),
        )

        return {
            "draft_answer": full_answer,
            "citations": citation_dicts,
            "all_sources": cited_sources,
            "phase": "writing",
        }

    except Exception as e:
        logger.error("Synthesizer failed: %s", e)
        error_answer = (
            "I encountered an error while generating the answer. "
            "Please try again or rephrase your question."
        )
        return {
            "draft_answer": error_answer,
            "citations": [],
            "all_sources": cited_sources,
            "phase": "writing",
            "error": f"Synthesizer error: {str(e)}",
        }
