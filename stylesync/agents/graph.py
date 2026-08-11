"""LangGraph workflow — the multi-agent orchestration graph.

Architecture
════════════

  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │  Garment        │     │  RAG             │     │  Image          │
  │  Analyzer       │────▶│  Retriever       │────▶│  Generator      │
  │  (GPT-4o Vision)│     │  (Pinecone)      │     │  (SDXL+CN)      │
  └─────────────────┘     └─────────────────┘     └─────────────────┘

Each node is a pure function: AgentState → AgentState.
The graph executes sequentially because each stage depends on the previous.
Error handling routes to an error node that logs and returns gracefully.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from stylesync.agents.garment_analyzer import garment_analysis_node
from stylesync.agents.image_generator import image_generation_node
from stylesync.agents.rag_retriever import rag_retrieval_node
from stylesync.agents.state import AgentState

logger = logging.getLogger(__name__)


def error_handler_node(state: AgentState) -> AgentState:
    """Catch-all error node."""
    error = state.get("error", "Unknown error")
    logger.error("Pipeline error: %s", error)
    return {**state, "error": error}


def should_continue(state: AgentState) -> str:
    """Edge condition: route to error handler if an error is set."""
    if state.get("error"):
        return "error"
    return "continue"


def build_stylesync_graph() -> StateGraph:
    """Construct and compile the StyleSync multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_garment", garment_analysis_node)
    workflow.add_node("retrieve_style", rag_retrieval_node)
    workflow.add_node("generate_images", image_generation_node)
    workflow.add_node("handle_error", error_handler_node)

    workflow.set_entry_point("analyze_garment")

    workflow.add_conditional_edges("analyze_garment", should_continue, {"continue": "retrieve_style", "error": "handle_error"})
    workflow.add_conditional_edges("retrieve_style", should_continue, {"continue": "generate_images", "error": "handle_error"})
    workflow.add_edge("generate_images", END)
    workflow.add_edge("handle_error", END)

    return workflow.compile()


stylesync_graph = build_stylesync_graph()
