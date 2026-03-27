"""Agent 3: Image Generation — preprocesses garment and runs virtual try-on.

Orchestrates the full imaging pipeline:
  1. Garment preprocessing (bg removal, normalization, logo extraction)
  2. Pose estimation (from reference image or default skeleton)
  3. ControlNet + inpainting generation
  4. Output saving
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from stylesync.agents.state import AgentState
from stylesync.imaging.garment_processor import GarmentProcessor
from stylesync.imaging.pose_estimator import extract_pose, generate_default_pose
from stylesync.imaging.tryon_pipeline import TryOnPipeline
from stylesync.utils.config import settings

logger = logging.getLogger(__name__)

# Module-level singletons (lazy)
_processor: GarmentProcessor | None = None
_pipeline: TryOnPipeline | None = None


def _get_processor() -> GarmentProcessor:
    global _processor
    if _processor is None:
        _processor = GarmentProcessor()
    return _processor


def _get_pipeline() -> TryOnPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TryOnPipeline(
            sdxl_model_id=settings.sdxl_model_id,
            controlnet_model_id=settings.controlnet_model_id,
        )
    return _pipeline


def image_generation_node(state: AgentState) -> AgentState:
    """LangGraph node: preprocess garment, estimate pose, generate images."""
    image_path = state["garment_image_path"]
    guideline = state["style_guideline"]
    num_images = state.get("num_images", 1)

    logger.info("Starting image generation pipeline")

    # ── Step 1: Garment preprocessing ────────────────────────────────────
    processor = _get_processor()
    logo_position = state.get("detected_logo_position") if state.get("has_logo") else None
    preprocess_result = processor.preprocess(image_path, logo_position=logo_position)

    garment_clean = preprocess_result["garment_clean"]
    garment_normalized = preprocess_result["garment_normalized"]
    inpaint_mask = preprocess_result["inpaint_mask"]

    # ── Step 2: Pose estimation ──────────────────────────────────────────
    ref_model_path = state.get("reference_model_path")
    if ref_model_path and Path(ref_model_path).exists():
        logger.info("Extracting pose from reference model: %s", ref_model_path)
        ref_image = Image.open(ref_model_path).convert("RGB")
        pose_image = extract_pose(ref_image)
    else:
        logger.info("Using default pose: %s", guideline.pose)
        pose_image = generate_default_pose(pose_type=guideline.pose)

    # ── Step 3: Generate images ──────────────────────────────────────────
    pipeline = _get_pipeline()
    generated = pipeline.generate(
        garment_image=garment_normalized,
        pose_image=pose_image,
        guideline=guideline,
        inpaint_mask=inpaint_mask,
        num_images=num_images,
        steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
    )

    # ── Step 4: Save outputs ─────────────────────────────────────────────
    settings.ensure_dirs()
    sku = guideline.brand.lower().replace(" ", "-")
    output_paths = pipeline.save_results(
        generated, settings.output_dir, prefix=f"stylesync_{sku}"
    )

    return {
        **state,
        "garment_clean": garment_clean,
        "garment_normalized": garment_normalized,
        "pose_image": pose_image,
        "inpaint_mask": inpaint_mask,
        "generated_images": generated,
        "output_paths": [str(p) for p in output_paths],
    }
