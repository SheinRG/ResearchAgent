"""
Synthesizer Node — Cited Markdown Answer Generation.
Receives top-ranked chunks and generates a comprehensive answer
with [1], [2] citation markers, streaming tokens via Ollama.
"""

import logging
from app.services.llm import get_llm_client
from app.utils.citations import extract_citations, format_chunks_for_prompt
from app.agents.state import ResearchState
from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM = """You are an expert research synthesizer. Your job is to write comprehensive, well-structured answers based on the provided source material.

CRITICAL RULES:
1. ALWAYS cite your sources using [1], [2], [3] markers that correspond to the source numbers provided
2. Write in clear, professional markdown with proper headings, bullet points, and paragraphs
3. Start with a brief overview paragraph, then dive into details
4. Be comprehensive but concise — cover all aspects of the question
5. If sources conflict, acknowledge the disagreement and present both views
6. NEVER make claims without citing a source
7. Use ## for section headings when the answer covers multiple topics
8. Include specific data, numbers, and facts from the sources
9. End with a brief conclusion or summary

Format example:
Recent research has shown significant progress in this area [1]. According to multiple studies, the key finding is... [2][3].

## Key Developments
- Point one with supporting evidence [1]
- Point two with data [2]"""

SYNTHESIZER_PROMPT = """Based on the following sources, write a comprehensive and well-cited answer to this question:

**Question:** {query}

**Sources:**
{context}

Write a thorough answer with [1], [2], etc. citation markers. Use markdown formatting."""


async def synthesizer_node(state: ResearchState) -> dict:
    """
    Synthesizer node: generates a cited markdown answer by streaming tokens.

    Args:
        state: Current research state with ranked_chunks.

    Returns:
        Updated state with draft_answer, citations, phase.
    """
    query = state["query"]
    ranked_chunks = state.get("ranked_chunks", [])
    search_results = state.get("search_results", [])
    sse_callback = state.get("sse_callback")

    logger.info("Synthesizer: generating answer from %d chunks", len(ranked_chunks))

    # Phase: Writing
    if sse_callback:
        await sse_callback("phase", {
            "phase": "writing",
            "message": "Synthesizing your answer..."
        })

    # Build context from ranked chunks
    # Create RankedChunk-like objects for the formatter
    class ChunkProxy:
        def __init__(self, d):
            self.text = d.get("text", "")
            self.source_url = d.get("source_url", "")
            self.source_title = d.get("source_title", "")
            self.source_domain = d.get("source_domain", "")

    chunk_proxies = [ChunkProxy(c) for c in ranked_chunks]
    context = format_chunks_for_prompt(chunk_proxies, max_chunks=10)

    prompt = SYNTHESIZER_PROMPT.format(query=query, context=context)

    try:
        llm = get_llm_client()
        full_answer = ""

        # Stream tokens
        async for token in llm.generate_stream(
            prompt=prompt,
            system=SYNTHESIZER_SYSTEM,
            temperature=0.7,
        ):
            full_answer += token
            if sse_callback:
                await sse_callback("token", {"token": token})

        # Extract citations from the generated answer
        source_objects = [
            SearchResult(**r) if isinstance(r, dict) else r
            for r in search_results
        ]
        citations = extract_citations(full_answer, source_objects)
        citation_dicts = [c.model_dump() for c in citations]

        logger.info("Synthesizer: generated %d char answer with %d citations",
                    len(full_answer), len(citation_dicts))

        return {
            "draft_answer": full_answer,
            "citations": citation_dicts,
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
            "phase": "writing",
            "error": f"Synthesizer error: {str(e)}",
        }
