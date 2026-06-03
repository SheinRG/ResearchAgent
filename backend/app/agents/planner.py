"""
Planner Node — Query Decomposition.
Takes the user's question and generates 2-4 focused sub-queries
using Ollama's structured JSON output.
"""

import logging
from app.services.llm import get_llm_client
from app.agents.state import ResearchState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are a research planning assistant. Your job is to break down complex questions into focused sub-queries that can be individually searched on the web.

Rules:
- Generate 2-4 sub-queries that together cover the full scope of the original question
- Each sub-query should be specific and searchable (good for web search)
- Include temporal context if relevant (e.g., "2024-2025")
- Each sub-query should target a different aspect of the question
- Do NOT repeat the original question as a sub-query

Respond ONLY with valid JSON in this exact format:
{"sub_queries": ["sub-query 1", "sub-query 2", "sub-query 3"]}"""

PLANNER_PROMPT_TEMPLATE = """Break down this research question into 2-4 focused, searchable sub-queries:

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

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        query=query,
        refinement_context=refinement_context,
    )

    try:
        llm = get_llm_client()
        result = await llm.generate_structured(
            prompt=prompt,
            system=PLANNER_SYSTEM,
            temperature=0.3,
        )

        sub_queries = result.get("sub_queries", [])

        # Validate and clean
        if not sub_queries or not isinstance(sub_queries, list):
            logger.warning("Planner returned invalid sub_queries, using fallback")
            sub_queries = [query]

        # Ensure 2-4 queries
        sub_queries = [q.strip() for q in sub_queries if q.strip()][:4]
        if len(sub_queries) < 2:
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
