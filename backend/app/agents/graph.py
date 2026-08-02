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
from typing import Awaitable, Callable, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from app.agents.state import ResearchState
from app.agents.router import router_node
from app.agents.conversational import conversational_node
from app.agents.researcher import researcher_node, rerank_node
from app.agents.synthesizer import synthesizer_node
from app.services import tracing

logger = logging.getLogger(__name__)

NodeFn = Callable[[ResearchState], Awaitable[dict]]


# What each stage is worth recording. Deliberately counts and sizes rather than
# payloads: the full state carries every scraped chunk, which is megabytes of
# noise in a trace viewer. Answer/query text arrives via the LLM generations
# nested inside these spans, and is subject to langfuse_capture_content there.
_STAGE_OUTPUT: dict[str, Callable[[dict], dict]] = {
    "triage": lambda out: {
        "mode": out.get("mode"),
        "sub_queries": len(out.get("sub_queries") or []),
        "answer_format": (out.get("answer_format") or {}).get("type"),
        "needs_web": out.get("needs_web"),
    },
    "search": lambda out: {
        "sources": len(out.get("search_results") or []),
        "chunks": len(out.get("scraped_content") or []),
        "document_chunks": len(out.get("document_chunks") or []),
        "images": len(out.get("images") or []),
    },
    "rerank": lambda out: {
        "ranked_chunks": len(out.get("ranked_chunks") or []),
        "top_score": round((out.get("ranked_chunks") or [{}])[0].get("score", 0.0), 4),
    },
    "synthesize": lambda out: {
        "answer_chars": len(out.get("draft_answer") or ""),
        "citations": len(out.get("citations") or []),
        "invalid_citations": out.get("invalid_citations", 0),
        "confidence": out.get("confidence", 0.0),
        "follow_ups": len(out.get("follow_up_suggestions") or []),
    },
    "chat": lambda out: {"answer_chars": len(out.get("draft_answer") or "")},
}


def _traced(stage: str, node: NodeFn) -> NodeFn:
    """
    Wrap a graph node so its wall time and result shape land in a span.

    Done here rather than inside the nodes so the agent code stays free of
    observability plumbing, and so no future node can be added without a
    deliberate decision about how it is traced.
    """

    async def traced_node(state: ResearchState) -> dict:
        with tracing.span(stage) as observation:
            result = await node(state)
            try:
                observation.update(output=_STAGE_OUTPUT[stage](result))
            except Exception as e:  # a summariser must never break the pipeline
                logger.debug("Stage summary failed for %s (ignored): %s", stage, e)
            return result

    traced_node.__name__ = getattr(node, "__name__", stage)
    return traced_node


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

    # Add nodes. Each is wrapped so the stage shows up as its own span with its
    # own latency; the LLM calls inside nest underneath as generations.
    graph.add_node("router", _traced("triage", router_node))
    graph.add_node("conversational", _traced("chat", conversational_node))
    graph.add_node("researcher", _traced("search", researcher_node))
    graph.add_node("reranker", _traced("rerank", rerank_node))
    graph.add_node("synthesizer", _traced("synthesize", synthesizer_node))

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
