"""RAG schema — Pydantic models for product attributes stored in Pinecone.

Vector DB Schema Design
═══════════════════════
Each vector in Pinecone represents one garment/product and carries structured
metadata alongside the embedding.  The embedding is generated from a combined
text representation of brand + category + style notes so that semantic search
returns contextually relevant style guidelines.

Pinecone Index Layout
─────────────────────
  Namespace : \"products\"
  Dimension : 384  (all-MiniLM-L6-v2)
  Metric    : cosine

  Metadata fields (filterable):
    brand           : str       — e.g. \"Nike\", \"Zara\"
    category        : str       — e.g. \"t-shirt\", \"hoodie\", \"jacket\"
    sub_category    : str       — e.g. \"crew-neck\", \"v-neck\", \"polo\"
    color_primary   : str       — dominant garment color
    color_secondary : str|None  — accent / secondary color
    pattern         : str       — \"solid\", \"striped\", \"graphic\", \"logo\"
    fabric          : str       — \"cotton\", \"polyester\", \"blend\"
    fit             : str       — \"slim\", \"regular\", \"oversized\"
    gender          : str       — \"mens\", \"womens\", \"unisex\"
    has_logo        : bool      — whether garment has a visible logo/print
    logo_position   : str|None  — \"center-chest\", \"left-chest\", \"back\", etc.
    style_notes     : str       — free-text brand style guidelines
    model_pose      : str       — recommended pose: \"front\", \"3/4\", \"side\"
    background      : str       — recommended bg: \"white\", \"lifestyle\", \"gradient\"
    lighting        : str       — \"studio-soft\", \"natural\", \"dramatic\"
    sku             : str       — product SKU for traceability
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GarmentMetadata(BaseModel):
    """Metadata stored alongside each product vector in Pinecone."""

    brand: str = Field(..., description="Brand name, e.g. 'Nike'")
    category: str = Field(..., description="Garment category, e.g. 't-shirt'")
    sub_category: str = Field("crew-neck", description="Sub-category, e.g. 'v-neck'")
    color_primary: str = Field(..., description="Dominant garment color")
    color_secondary: Optional[str] = Field(None, description="Accent color if any")
    pattern: str = Field("solid", description="Pattern type: solid, striped, graphic, logo")
    fabric: str = Field("cotton", description="Fabric composition")
    fit: str = Field("regular", description="Fit type: slim, regular, oversized")
    gender: str = Field("unisex", description="Target gender")
    has_logo: bool = Field(False, description="Whether garment has a visible logo/print")
    logo_position: Optional[str] = Field(None, description="Position of logo on garment")
    style_notes: str = Field("", description="Brand-specific styling guidelines")
    model_pose: str = Field("front", description="Recommended model pose")
    background: str = Field("white", description="Recommended background style")
    lighting: str = Field("studio-soft", description="Recommended lighting setup")
    sku: str = Field(..., description="Product SKU")

    def to_embedding_text(self) -> str:
        """Combine key fields into a single string for embedding generation."""
        parts = [
            f"Brand: {self.brand}",
            f"Category: {self.category} {self.sub_category}",
            f"Color: {self.color_primary}",
            f"Pattern: {self.pattern}",
            f"Fit: {self.fit}",
            f"Fabric: {self.fabric}",
        ]
        if self.has_logo and self.logo_position:
            parts.append(f"Logo at {self.logo_position}")
        if self.style_notes:
            parts.append(f"Style: {self.style_notes}")
        return ". ".join(parts)


class StyleGuideline(BaseModel):
    """Structured style guideline returned by RAG to downstream agents."""

    brand: str
    pose: str
    background: str
    lighting: str
    style_notes: str
    has_logo: bool
    logo_position: Optional[str]
    pattern: str
    fit: str
    negative_prompt_hints: list[str] = Field(
        default_factory=list,
        description="Things to avoid in generation, e.g. 'do not alter logo'",
    )
