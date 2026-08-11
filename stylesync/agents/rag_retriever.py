"""Agent 2: RAG Retriever — fetches brand-specific style guidelines from Pinecone."""

from __future__ import annotations

import logging

from stylesync.agents.state import AgentState
from stylesync.rag.vector_store import ProductVectorStore

logger = logging.getLogger(__name__)

_store: ProductVectorStore | None = None


def _get_store() -> ProductVectorStore:
    global _store
    if _store is None:
        _store = ProductVectorStore()
    return _store


def rag_retrieval_node(state: AgentState) -> AgentState:
    """LangGraph node: query Pinecone for matching style guidelines."""
    brand = state.get("detected_brand", "unknown")
    category = state.get("detected_category", "t-shirt")
    description = state.get("garment_description", "")
    pattern = state.get("detected_pattern", "solid")

    query = f"{brand} {category} {pattern} {description}"
    logger.info("RAG query: %s", query[:200])

    store = _get_store()
    guideline = store.get_style_guideline(
        query_text=query,
        brand=brand if brand != "unknown" else None,
        category=category,
    )

    if state.get("has_logo") and not guideline.has_logo:
        guideline.has_logo = True
        guideline.logo_position = state.get("detected_logo_position")
        guideline.negative_prompt_hints.append("do not alter, distort, or remove the logo/graphic")

    guideline.pattern = state.get("detected_pattern", guideline.pattern)

    context_str = (
        f"Brand: {guideline.brand}\n"
        f"Pose: {guideline.pose}\n"
        f"Background: {guideline.background}\n"
        f"Lighting: {guideline.lighting}\n"
        f"Logo: {'yes at ' + (guideline.logo_position or 'unknown') if guideline.has_logo else 'no'}\n"
        f"Style notes: {guideline.style_notes}\n"
        f"Negative hints: {', '.join(guideline.negative_prompt_hints)}"
    )
    logger.info("RAG context:\n%s", context_str)

    return {**state, "style_guideline": guideline, "rag_context": context_str}
