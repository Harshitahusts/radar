"""Tests for the TryOnPipeline prompt builder (no GPU required)."""

from stylesync.imaging.tryon_pipeline import TryOnPipeline
from stylesync.rag.schema import StyleGuideline


def test_build_prompt_basic():
    guideline = StyleGuideline(
        brand="Nike",
        pose="front",
        background="white",
        lighting="studio-soft",
        style_notes="Swoosh must be sharp",
        has_logo=True,
        logo_position="center-chest",
        pattern="logo",
        fit="regular",
        negative_prompt_hints=["do not alter the logo"],
    )
    pos, neg = TryOnPipeline.build_prompt(guideline)
    assert "professional ecommerce" in pos
    assert "facing the camera" in pos
    assert "logo/graphic perfectly preserved" in pos
    assert "Swoosh must be sharp" in pos
    assert "do not alter the logo" in neg
    assert "blurry" in neg


def test_build_prompt_no_logo():
    guideline = StyleGuideline(
        brand="Zara",
        pose="3/4",
        background="gradient",
        lighting="natural",
        style_notes="",
        has_logo=False,
        logo_position=None,
        pattern="solid",
        fit="slim",
    )
    pos, neg = TryOnPipeline.build_prompt(guideline)
    assert "three-quarter angle" in pos
    assert "gradient" in pos
    assert "logo" not in pos
    assert "natural window lighting" in pos


def test_build_prompt_lifestyle():
    guideline = StyleGuideline(
        brand="Supreme",
        pose="3/4",
        background="lifestyle",
        lighting="dramatic",
        style_notes="Streetwear aesthetic",
        has_logo=True,
        logo_position="center-chest",
        pattern="graphic",
        fit="oversized",
        negative_prompt_hints=["no warping", "no color shift"],
    )
    pos, neg = TryOnPipeline.build_prompt(guideline)
    assert "urban lifestyle" in pos
    assert "dramatic" in pos
    assert "no warping" in neg
    assert "no color shift" in neg
