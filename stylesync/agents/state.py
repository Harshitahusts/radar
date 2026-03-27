"""Shared state definition for the LangGraph multi-agent workflow.

The AgentState TypedDict flows through the graph — each agent node reads
from and writes to specific fields, enabling clean data handoff.
"""

from __future__ import annotations

from typing import Any, Optional

from PIL import Image
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from stylesync.rag.schema import StyleGuideline


class AgentState(TypedDict, total=False):
    """State shared across all agents in the LangGraph workflow.

    Flow: input → garment_analysis → rag_lookup → image_generation → output
    """

    # ── Input ────────────────────────────────────────────────────────────
    garment_image_path: str                    # Path to uploaded flat-lay image
    brand_hint: Optional[str]                  # Optional brand name from user
    category_hint: Optional[str]               # Optional category hint
    reference_model_path: Optional[str]        # Optional reference model photo
    num_images: int                            # How many variants to generate

    # ── Garment Analysis Agent output ────────────────────────────────────
    garment_description: str                   # LLM-generated description of garment
    detected_brand: Optional[str]              # Brand detected from image
    detected_category: str                     # e.g. "t-shirt"
    detected_pattern: str                      # e.g. "logo", "solid"
    detected_logo_position: Optional[str]      # e.g. "center-chest"
    has_logo: bool

    # ── RAG Agent output ─────────────────────────────────────────────────
    style_guideline: StyleGuideline            # Retrieved style rules
    rag_context: str                           # Raw context string for prompt

    # ── Preprocessing output ─────────────────────────────────────────────
    garment_clean: Any                         # PIL Image (bg removed)
    garment_normalized: Any                    # PIL Image (resized)
    pose_image: Any                            # PIL Image (OpenPose skeleton)
    inpaint_mask: Any                          # PIL Image (logo protection mask)

    # ── Generation Agent output ──────────────────────────────────────────
    generated_images: list[Any]                # List of PIL Images
    output_paths: list[str]                    # Saved file paths

    # ── Error handling ───────────────────────────────────────────────────
    error: Optional[str]
