"""
LangGraph State Graph — Wiring all agent nodes together.
Defines the research agent as a state machine:
Triage (route + plan) → Researcher → Reranker → Synthesizer → end

The triage node both routes (chat vs research) and plans the sub-queries in one
LLM call, and the synthesizer generates follow-ups concurrently with the answer
stream — so the pipeline is a single forward pass with no trailing reflector
round-trip and no separate planner call.
"""

import logging
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from app.agents.state import ResearchState
from app.agents.router import router_node
from app.agents.conversational import conversational_node
from app.agents.researcher import researcher_node, rerank_node
from app.agents.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


def route_mode(state: ResearchState) -> str:
    """Entry routing: a 'chat' message replies directly; anything else researches."""
    return "chat" if state.get("mode") == "chat" else "research"


def build_research_graph() -> StateGraph:
    """
    Build and compile the LangGraph research agent.

    Graph flow:
        router → (chat → conversational) | (research → researcher → reranker → synthesizer)

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("conversational", conversational_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("reranker", rerank_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Triage first: a casual/simple message gets a direct chat reply; a genuine
    # research question goes straight into the pipeline (sub-queries were already
    # planned inside the triage call, so there is no separate planner step).
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_mode,
        {"chat": "conversational", "research": "researcher"},
    )
    graph.add_edge("conversational", END)

    # Research pipeline: one forward pass, ending at the synthesizer (which also
    # emits follow-up suggestions before it returns).
    graph.add_edge("researcher", "reranker")
    graph.add_edge("reranker", "synthesizer")
    graph.add_edge("synthesizer", END)

    compiled = graph.compile()
    logger.info("Research graph compiled successfully")
    return compiled


# Singleton compiled graph
_graph: Optional[CompiledStateGraph] = None


def get_research_graph() -> CompiledStateGraph:
    """Get the singleton compiled research graph."""
    global _graph
    if _graph is None:
        _graph = build_research_graph()
    return _graph
