"""
Reflector Node — Gap Analysis, Confidence Scoring, Loop Decision.
Evaluates the draft answer for completeness, identifies gaps,
and decides whether to loop back to the planner or finalize.
"""

import logging
from app.services.llm import get_llm_client
from app.agents.state import ResearchState

logger = logging.getLogger(__name__)

REFLECTOR_SYSTEM = """You are a research quality evaluator. Your job is to assess whether a research answer fully addresses the original question.

Evaluate based on:
1. Coverage: Does the answer address ALL aspects of the question?
2. Evidence: Are claims properly supported by cited sources?
3. Accuracy: Are there any contradictions or unsupported claims?
4. Depth: Is the answer sufficiently detailed for the complexity of the question?

Respond ONLY with valid JSON in this exact format:
{
    "confidence": 0.85,
    "gaps": ["gap description 1", "gap description 2"],
    "should_continue": false,
    "refined_queries": ["refined query 1"],
    "reasoning": "Brief explanation of your assessment"
}

Rules for confidence scoring:
- 0.9-1.0: Excellent coverage, all aspects addressed with strong evidence
- 0.7-0.89: Good coverage, minor gaps that don't significantly affect quality
- 0.5-0.69: Moderate coverage, notable gaps that should be addressed
- Below 0.5: Poor coverage, significant aspects missing

Set should_continue=true ONLY if confidence < 0.7 AND there are specific, actionable gaps."""

REFLECTOR_PROMPT = """Evaluate this research answer for completeness and quality:

**Original Question:** {query}

**Sub-questions that were researched:**
{sub_queries}

**Draft Answer:**
{answer}

**Number of sources used:** {num_sources}

Evaluate the answer and respond with JSON only:"""

FOLLOWUP_SYSTEM = """You are a research assistant. Generate 3-4 natural follow-up questions that a curious reader might ask after reading the answer to the original question. The follow-ups should:
1. Explore related but different angles
2. Go deeper into specific mentioned topics
3. Be specific and searchable
4. NOT repeat what was already answered

Respond ONLY with valid JSON:
{"suggestions": ["question 1", "question 2", "question 3"]}"""

FOLLOWUP_PROMPT = """Original question: {query}

Answer summary: {answer_summary}

Generate 3-4 natural follow-up questions. Respond with JSON only:"""


async def reflector_node(state: ResearchState) -> dict:
    """
    Reflector node: evaluates answer quality, decides to loop or finalize.

    Args:
        state: Current research state with draft_answer.

    Returns:
        Updated state with reflection, confidence, follow_up_suggestions.
    """
    query = state["query"]
    draft_answer = state.get("draft_answer", "")
    sub_queries = state.get("sub_queries", [])
    search_results = state.get("search_results", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)
    sse_callback = state.get("sse_callback")

    logger.info("Reflector: evaluating answer (iteration %d/%d)", iteration, max_iterations)

    # Phase: Reflecting
    if sse_callback:
        await sse_callback("phase", {
            "phase": "reflecting",
            "message": "Checking for gaps and evaluating quality..."
        })

    try:
        llm = get_llm_client()

        # --- Evaluate the answer ---
        prompt = REFLECTOR_PROMPT.format(
            query=query,
            sub_queries="\n".join(f"- {q}" for q in sub_queries),
            answer=draft_answer[:3000],  # Limit length for evaluation
            num_sources=len(search_results),
        )

        result = await llm.generate_structured(
            prompt=prompt,
            system=REFLECTOR_SYSTEM,
            temperature=0.2,
        )

        confidence = float(result.get("confidence", 0.7))
        gaps = result.get("gaps", [])
        should_continue = result.get("should_continue", False)
        refined_queries = result.get("refined_queries", [])
        reasoning = result.get("reasoning", "")

        # Force stop if we've reached max iterations
        if iteration >= max_iterations:
            should_continue = False
            logger.info("Reflector: max iterations reached, finalizing")

        # Don't loop for minor gaps
        if confidence >= 0.7:
            should_continue = False

        reflection = {
            "confidence": confidence,
            "gaps": gaps if isinstance(gaps, list) else [],
            "should_continue": should_continue,
            "refined_queries": refined_queries if isinstance(refined_queries, list) else [],
            "reasoning": reasoning,
        }

        logger.info(
            "Reflector: confidence=%.2f, gaps=%d, continue=%s",
            confidence, len(gaps), should_continue,
        )

        # --- Generate follow-up suggestions ---
        follow_ups = []
        if not should_continue:
            follow_ups = await _generate_follow_ups(llm, query, draft_answer)
            if sse_callback and follow_ups:
                await sse_callback("follow_up", {"suggestions": follow_ups})

        return {
            "reflection": reflection,
            "confidence": confidence,
            "iteration": iteration + 1,
            "follow_up_suggestions": follow_ups,
            "phase": "reflecting",
        }

    except Exception as e:
        logger.error("Reflector failed: %s", e)
        return {
            "reflection": {
                "confidence": 0.6,
                "gaps": [],
                "should_continue": False,
                "refined_queries": [],
                "reasoning": f"Reflection failed: {str(e)}",
            },
            "confidence": 0.6,
            "iteration": iteration + 1,
            "follow_up_suggestions": [],
            "phase": "reflecting",
            "error": f"Reflector error: {str(e)}",
        }


async def _generate_follow_ups(llm, query: str, answer: str) -> list[str]:
    """Generate follow-up question suggestions."""
    try:
        # Use first 500 chars as summary
        answer_summary = answer[:500] + "..." if len(answer) > 500 else answer

        prompt = FOLLOWUP_PROMPT.format(
            query=query,
            answer_summary=answer_summary,
        )

        result = await llm.generate_structured(
            prompt=prompt,
            system=FOLLOWUP_SYSTEM,
            temperature=0.5,
        )

        suggestions = result.get("suggestions", [])
        if isinstance(suggestions, list):
            return [s.strip() for s in suggestions if s.strip()][:4]
        return []

    except Exception as e:
        logger.warning("Follow-up generation failed: %s", e)
        return []
