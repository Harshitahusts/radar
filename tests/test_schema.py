"""Tests for the RAG schema models."""

from stylesync.rag.schema import GarmentMetadata, StyleGuideline


def test_garment_metadata_to_embedding_text():
    meta = GarmentMetadata(
        brand="Nike", category="t-shirt", sub_category="crew-neck",
        color_primary="black", pattern="logo", fabric="cotton", fit="regular",
        gender="mens", has_logo=True, logo_position="center-chest",
        style_notes="Swoosh logo must remain sharp", sku="SKU-001",
    )
    text = meta.to_embedding_text()
    assert "Nike" in text
    assert "t-shirt" in text
    assert "logo" in text
    assert "center-chest" in text
    assert "Swoosh" in text


def test_garment_metadata_no_logo():
    meta = GarmentMetadata(brand="Zara", category="t-shirt", color_primary="white", pattern="solid", sku="SKU-002")
    text = meta.to_embedding_text()
    assert "Zara" in text
    assert "center-chest" not in text


def test_style_guideline_defaults():
    sg = StyleGuideline(
        brand="Test", pose="front", background="white", lighting="studio-soft",
        style_notes="", has_logo=False, logo_position=None, pattern="solid", fit="regular",
    )
    assert sg.negative_prompt_hints == []
    assert sg.has_logo is False


def test_style_guideline_with_logo_hints():
    sg = StyleGuideline(
        brand="Supreme", pose="3/4", background="lifestyle", lighting="dramatic",
        style_notes="Box logo pixel-perfect", has_logo=True, logo_position="center-chest",
        pattern="graphic", fit="oversized", negative_prompt_hints=["do not alter logo"],
    )
    assert len(sg.negative_prompt_hints) == 1
    assert "logo" in sg.negative_prompt_hints[0]
