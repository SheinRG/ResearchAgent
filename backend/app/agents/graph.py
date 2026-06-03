"""
LangGraph State Graph — Wiring all agent nodes together.
Defines the research agent as a state machine:
Planner → Researcher → Reranker → Synthesizer → Reflector → (loop or end)
"""

import logging
from langgraph.graph import StateGraph, END

from app.agents.state import ResearchState
from app.agents.planner import planner_node
from app.agents.researcher import researcher_node, rerank_node
from app.agents.synthesizer import synthesizer_node
from app.agents.reflector import reflector_node

logger = logging.getLogger(__name__)


def should_continue(state: ResearchState) -> str:
    """
    Conditional edge: decides whether to loop back to planner or end.

    Returns:
        'planner' if gaps found and iterations remain, 'end' otherwise.
    """
    reflection = state.get("reflection", {})
    should_loop = reflection.get("should_continue", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)

    if should_loop and iteration < max_iterations:
        logger.info("Reflector decided to loop (iteration %d/%d)", iteration, max_iterations)
        return "planner"
    else:
        logger.info("Reflector decided to end (confidence=%.2f)", state.get("confidence", 0))
        return "end"


def build_research_graph() -> StateGraph:
    """
    Build and compile the LangGraph research agent.

    Graph flow:
        planner → researcher → reranker → synthesizer → reflector → (planner | end)

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("reranker", rerank_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("reflector", reflector_node)

    # Define edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "reranker")
    graph.add_edge("reranker", "synthesizer")
    graph.add_edge("synthesizer", "reflector")

    # Conditional edge from reflector
    graph.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "planner": "planner",
            "end": END,
        },
    )

    compiled = graph.compile()
    logger.info("Research graph compiled successfully")
    return compiled


# Singleton compiled graph
_graph = None


def get_research_graph():
    """Get the singleton compiled research graph."""
    global _graph
    if _graph is None:
        _graph = build_research_graph()
    return _graph
