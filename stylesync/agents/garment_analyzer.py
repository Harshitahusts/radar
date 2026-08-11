"""Agent 1: Garment Analysis — analyzes the uploaded flat-lay image.

Uses a vision-capable LLM (GPT-4o) to extract structured metadata from the
garment photo: brand, category, pattern, logo presence/position, colors, etc.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from stylesync.agents.state import AgentState
from stylesync.utils.config import settings

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """\
You are a fashion product analyst for an eCommerce platform.

Analyze this flat-lay garment image and extract the following attributes.
Return your answer as a structured JSON object with these exact keys:

{
  "brand": "<detected brand name or 'unknown'>",
  "category": "<garment type: t-shirt, hoodie, jacket, polo, etc.>",
  "pattern": "<solid, striped, graphic, logo, checkered, etc.>",
  "has_logo": <true/false>,
  "logo_position": "<center-chest, left-chest, back, sleeve, null>",
  "color_primary": "<dominant color>",
  "color_secondary": "<accent color or null>",
  "fabric_guess": "<cotton, polyester, denim, etc.>",
  "fit": "<slim, regular, oversized>",
  "description": "<1-2 sentence description of the garment for prompt engineering>"
}

Be precise about logo detection — if there is any visible text, graphic, or
brand marking on the garment, set has_logo to true and specify its position.
"""


def garment_analysis_node(state: AgentState) -> AgentState:
    """LangGraph node: analyze the garment image using GPT-4o vision."""
    image_path = state["garment_image_path"]
    logger.info("Analyzing garment image: %s", image_path)

    image_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_bytes).decode()

    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)

    message = HumanMessage(content=[
        {"type": "text", "text": ANALYSIS_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])

    response = llm.invoke([message])
    raw = response.content
    logger.info("Garment analysis raw response: %s", raw[:300])

    import json
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    analysis = json.loads(text)

    brand = state.get("brand_hint") or analysis.get("brand", "unknown")
    category = state.get("category_hint") or analysis.get("category", "t-shirt")

    return {
        **state,
        "garment_description": analysis.get("description", ""),
        "detected_brand": brand,
        "detected_category": category,
        "detected_pattern": analysis.get("pattern", "solid"),
        "detected_logo_position": analysis.get("logo_position"),
        "has_logo": analysis.get("has_logo", False),
    }
