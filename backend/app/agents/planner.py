"""
Planner Node — Query Decomposition.
Takes the user's question and generates 2-4 focused sub-queries
using Groq's structured JSON output.
"""

import logging
from datetime import date

from app.services.llm import get_llm_client
from app.agents.state import ResearchState, format_history

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are a research planning assistant. Your job is to break down complex questions into focused sub-queries that can be individually searched on the web.

Rules:
- Generate 2-4 sub-queries that together cover the full scope of the original question
- Each sub-query should be specific and searchable (good for web search)
- For time-sensitive topics, anchor recency to the current year ({current_year}) — e.g. "... {current_year}" or "latest {current_year}"
- Each sub-query should target a different aspect of the question
- Prefer authoritative angles (official data, primary sources, expert analysis)
- Do NOT repeat the original question verbatim as a sub-query

Follow-up handling (when a conversation is provided):
- The new question may be a follow-up that relies on the earlier conversation. Resolve references like "it", "they", "that", "this", or "the company" to the ACTUAL named entities from the conversation.
- Every sub-query MUST be self-contained: it is sent to a web search engine that has NO memory of the conversation, so spell out the real names, places, and topics instead of pronouns.

Respond ONLY with valid JSON in this exact format:
{{"sub_queries": ["sub-query 1", "sub-query 2", "sub-query 3"]}}"""

PLANNER_PROMPT_TEMPLATE = """{conversation_context}Break down this research question into 2-4 focused, searchable sub-queries:

Question: {query}

{refinement_context}

Respond with JSON only:"""


async def planner_node(state: ResearchState) -> dict:
    """
    Planner node: decomposes the user query into sub-queries.

    Args:
        state: Current research state.

    Returns:
        Updated state fields: sub_queries, phase.
    """
    query = state["query"]
    iteration = state.get("iteration", 0)
    history = state.get("history", [])
    sse_callback = state.get("sse_callback")

    logger.info("Planner: decomposing query (iteration %d): %s", iteration, query[:100])

    # Send phase update
    if sse_callback:
        await sse_callback("phase", {
            "phase": "planning",
            "message": "Breaking down your question..." if iteration == 0 else "Refining search strategy..."
        })

    # Build refinement context from previous reflection
    refinement_context = ""
    reflection = state.get("reflection")
    if reflection and iteration > 0:
        gaps = reflection.get("gaps", [])
        if gaps:
            refinement_context = (
                f"Previous search found gaps in these areas:\n"
                f"{chr(10).join(f'- {g}' for g in gaps)}\n"
                f"Focus the new sub-queries on filling these specific gaps."
            )

    # Build conversation context for follow-up questions. Only the planner needs
    # the full transcript to resolve references into self-contained sub-queries.
    conversation_context = ""
    history_text = format_history(history)
    if history_text:
        conversation_context = (
            "This is a FOLLOW-UP question in an ongoing research conversation. "
            "Use the conversation below to resolve any references to real entities, "
            "then make every sub-query self-contained.\n\n"
            f"Conversation so far:\n{history_text}\n\n"
        )

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        conversation_context=conversation_context,
        query=query,
        refinement_context=refinement_context,
    )
    system = PLANNER_SYSTEM.format(current_year=date.today().year)

    try:
        llm = get_llm_client()
        result = await llm.generate_structured(
            prompt=prompt,
            system=system,
            temperature=0.3,
        )

        sub_queries = result.get("sub_queries", [])

        # Validate and clean
        if not sub_queries or not isinstance(sub_queries, list):
            logger.warning("Planner returned invalid sub_queries, using fallback")
            sub_queries = [query]

        # Ensure 2-4 queries and deduplicate to prevent redundant pipeline execution
        seen = set()
        clean_queries = []
        for q in sub_queries:
            q_clean = q.strip()
            if q_clean and q_clean.lower() not in seen:
                seen.add(q_clean.lower())
                clean_queries.append(q_clean)
        
        sub_queries = clean_queries[:4]
        
        # Add original query as fallback if we don't have enough distinct sub-queries
        if len(sub_queries) < 2 and query.lower() not in seen:
            sub_queries.append(query)

        logger.info("Planner generated %d sub-queries: %s", len(sub_queries), sub_queries)

        # Send sub-queries event
        if sse_callback:
            await sse_callback("sub_queries", {"queries": sub_queries})

        return {
            "sub_queries": sub_queries,
            "phase": "planning",
        }

    except Exception as e:
        logger.error("Planner failed: %s", e)
        # Fallback: use the original query
        fallback = [query]
        if sse_callback:
            await sse_callback("sub_queries", {"queries": fallback})
        return {
            "sub_queries": fallback,
            "phase": "planning",
            "error": f"Planner error: {str(e)}",
        }
