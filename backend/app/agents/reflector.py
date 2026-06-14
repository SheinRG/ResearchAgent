"""
Reflector Node — Gap Analysis, Confidence Scoring, Loop Decision.
Evaluates the draft answer for completeness, identifies gaps,
and decides whether to loop back to the planner or finalize.

Generates follow-up suggestions in the SAME LLM call to avoid
an extra round-trip to Groq.
"""

import logging
from app.services.llm import get_llm_client
from app.agents.state import ResearchState

logger = logging.getLogger(__name__)

# Combined system prompt: reflection + follow-up generation in one call
REFLECTOR_SYSTEM = """You are a research quality evaluator. Your job is to assess whether a research answer fully addresses the original question AND suggest follow-up questions.

Evaluate based on:
1. Coverage: Does the answer address ALL aspects of the question?
2. Evidence: Are claims properly supported by cited sources?
3. Accuracy: Are there any contradictions or unsupported claims?
4. Depth: Is the answer sufficiently detailed for the complexity of the question?

Also generate 3 natural follow-up questions that a curious reader might ask after reading the answer. The follow-ups should:
- Explore related but different angles
- Go deeper into specific mentioned topics
- Be specific and searchable
- NOT repeat what was already answered

Respond ONLY with valid JSON in this exact format:
{
    "confidence": 0.85,
    "gaps": ["gap description 1", "gap description 2"],
    "should_continue": false,
    "refined_queries": ["refined query 1"],
    "reasoning": "Brief explanation of your assessment",
    "follow_up_suggestions": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
}

Rules for confidence scoring:
- 0.9-1.0: Excellent coverage, all aspects addressed with strong evidence
- 0.7-0.89: Good coverage, minor gaps that don't significantly affect quality
- 0.5-0.69: Moderate coverage, notable gaps that should be addressed
- Below 0.5: Poor coverage, significant aspects missing

Set should_continue=true ONLY if confidence < 0.7 AND there are specific, actionable gaps."""

REFLECTOR_PROMPT = """Evaluate this research answer for completeness and quality, then suggest follow-up questions:

**Original Question:** {query}

**Sub-questions that were researched:**
{sub_queries}

**Draft Answer:**
{answer}

**Number of sources used:** {num_sources}
{prior_questions}
Evaluate the answer AND generate 3 follow-up questions. Respond with JSON only:"""


async def reflector_node(state: ResearchState) -> dict:
    """
    Reflector node: evaluates answer quality, decides to loop or finalize,
    and generates follow-up suggestions — all in a single LLM call.

    Args:
        state: Current research state with draft_answer.

    Returns:
        Updated state with reflection, confidence, follow_up_suggestions.
    """
    query = state["query"]
    draft_answer = state.get("draft_answer", "")
    sub_queries = state.get("sub_queries", [])
    search_results = state.get("search_results", [])
    history = state.get("history", [])
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

        # List earlier questions so suggested follow-ups don't repeat them.
        prior_questions = ""
        asked = [(t.get("query") or "").strip() for t in history if t.get("query")]
        if asked:
            prior_questions = (
                "\n**Already asked in this conversation (do NOT suggest these or "
                "close paraphrases):**\n"
                + "\n".join(f"- {q}" for q in asked)
                + "\n"
            )

        # --- Single combined LLM call for reflection + follow-ups ---
        prompt = REFLECTOR_PROMPT.format(
            query=query,
            sub_queries="\n".join(f"- {q}" for q in sub_queries),
            answer=draft_answer[:3000],  # Limit length for evaluation
            num_sources=len(search_results),
            prior_questions=prior_questions,
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

        # Extract follow-up suggestions from the same response
        follow_ups = result.get("follow_up_suggestions", [])
        if isinstance(follow_ups, list):
            follow_ups = [s.strip() for s in follow_ups if isinstance(s, str) and s.strip()][:4]
        else:
            follow_ups = []

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
            "Reflector: confidence=%.2f, gaps=%d, continue=%s, follow_ups=%d",
            confidence, len(gaps), should_continue, len(follow_ups),
        )

        # Send follow-up suggestions to frontend
        if sse_callback and follow_ups and not should_continue:
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
