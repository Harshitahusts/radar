"""Shared state definition for the LangGraph multi-agent workflow."""

from __future__ import annotations

from typing import Any, Optional

from PIL import Image
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from stylesync.rag.schema import StyleGuideline


class AgentState(TypedDict, total=False):
    """State shared across all agents in the LangGraph workflow."""

    garment_image_path: str
    brand_hint: Optional[str]
    category_hint: Optional[str]
    reference_model_path: Optional[str]
    num_images: int

    garment_description: str
    detected_brand: Optional[str]
    detected_category: str
    detected_pattern: str
    detected_logo_position: Optional[str]
    has_logo: bool

    style_guideline: StyleGuideline
    rag_context: str

    garment_clean: Any
    garment_normalized: Any
    pose_image: Any
    inpaint_mask: Any

    generated_images: list[Any]
    output_paths: list[str]

    error: Optional[str]
