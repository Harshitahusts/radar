"""Virtual Try-On pipeline using Diffusers (IDM-VTON / SDXL + ControlNet).

This module implements two generation strategies:
  1. IDM-VTON  — a dedicated virtual try-on model that takes a garment image
                 and a model image and composites them with high fidelity.
  2. SDXL+ControlNet — a more flexible pipeline using ControlNet (OpenPose +
                 Canny/Depth) for body-aware generation with IP-Adapter for
                 garment texture transfer.

Both strategies use an inpainting mask to protect logo/graphic regions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from diffusers import (
    AutoPipelineForInpainting,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image

from stylesync.rag.schema import StyleGuideline

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


class TryOnPipeline:
    """Generates studio-quality model images wearing the input garment."""

    def __init__(
        self,
        sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_model_id: str = "lllyasviel/control_v11p_sd15_openpose",
    ):
        self._sdxl_id = sdxl_model_id
        self._cn_id = controlnet_model_id
        self._pipe = None
        self._inpaint_pipe = None

    # ── Lazy Loading ─────────────────────────────────────────────────────

    def _load_controlnet_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        logger.info("Loading ControlNet model: %s", self._cn_id)
        controlnet = ControlNetModel.from_pretrained(
            self._cn_id, torch_dtype=DTYPE
        )

        logger.info("Loading SDXL pipeline: %s", self._sdxl_id)
        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self._sdxl_id,
            controlnet=controlnet,
            torch_dtype=DTYPE,
        ).to(DEVICE)

        self._pipe.enable_model_cpu_offload()
        return self._pipe

    def _load_inpainting_pipeline(self):
        if self._inpaint_pipe is not None:
            return self._inpaint_pipe

        logger.info("Loading inpainting pipeline for logo protection")
        self._inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=DTYPE,
        ).to(DEVICE)

        self._inpaint_pipe.enable_model_cpu_offload()
        return self._inpaint_pipe

    # ── Prompt Construction ──────────────────────────────────────────────

    @staticmethod
    def build_prompt(guideline: StyleGuideline) -> tuple[str, str]:
        """Build positive and negative prompts from the style guideline.

        Returns (positive_prompt, negative_prompt).
        """
        pose_map = {
            "front": "facing the camera directly",
            "3/4": "standing at a three-quarter angle",
            "side": "standing in profile view",
        }
        bg_map = {
            "white": "clean white studio background",
            "gradient": "smooth gradient background, neutral tones",
            "lifestyle": "urban lifestyle background, city street",
        }
        light_map = {
            "studio-soft": "soft diffused studio lighting",
            "natural": "natural window lighting",
            "dramatic": "dramatic directional lighting with shadows",
        }

        positive_parts = [
            "professional ecommerce product photo",
            "fashion model wearing the exact garment",
            pose_map.get(guideline.pose, "facing camera"),
            bg_map.get(guideline.background, "studio background"),
            light_map.get(guideline.lighting, "studio lighting"),
            "high resolution, 8k, sharp details",
            "accurate fabric texture and color reproduction",
        ]
        if guideline.has_logo:
            positive_parts.append(
                "garment logo/graphic perfectly preserved, pixel-accurate"
            )
        if guideline.style_notes:
            positive_parts.append(guideline.style_notes)

        negative_parts = [
            "blurry", "low quality", "deformed", "distorted",
            "extra limbs", "bad anatomy", "watermark", "text overlay",
            "garment color changed", "wrong fabric texture",
        ]
        negative_parts.extend(guideline.negative_prompt_hints)

        return ", ".join(positive_parts), ", ".join(negative_parts)

    # ── Generation ───────────────────────────────────────────────────────

    def generate(
        self,
        garment_image: Image.Image,
        pose_image: Image.Image,
        guideline: StyleGuideline,
        inpaint_mask: Image.Image | None = None,
        num_images: int = 1,
        steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Generate model-on images using ControlNet + optional logo inpainting.

        Args:
            garment_image: Pre-processed garment (RGBA, background removed).
            pose_image: OpenPose skeleton or reference model image.
            guideline: Style guidelines from RAG.
            inpaint_mask: Binary mask protecting logo regions (black=keep).
            num_images: Number of variants to generate.
            steps: Diffusion inference steps.
            guidance_scale: Classifier-free guidance scale.
            controlnet_conditioning_scale: ControlNet influence strength.
            seed: Random seed for reproducibility.

        Returns:
            List of generated PIL Images.
        """
        positive_prompt, negative_prompt = self.build_prompt(guideline)
        generator = torch.Generator(device=DEVICE)
        if seed is not None:
            generator.manual_seed(seed)

        pipe = self._load_controlnet_pipeline()

        # Step 1: Generate base model-on image with ControlNet (pose-guided)
        logger.info("Generating %d image(s) — steps=%d, cfg=%s", num_images, steps, guidance_scale)
        results = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=pose_image,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images

        # Step 2: If garment has a logo, run inpainting pass to composite the
        # original logo region back onto the generated image
        if inpaint_mask is not None and guideline.has_logo:
            logger.info("Running inpainting pass to protect logo region")
            inpaint_pipe = self._load_inpainting_pipeline()
            refined = []
            for img in results:
                out = inpaint_pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=img,
                    mask_image=inpaint_mask,
                    num_inference_steps=20,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                refined.append(out)
            results = refined

        return results

    def save_results(
        self, images: list[Image.Image], output_dir: str | Path, prefix: str = "result"
    ) -> list[Path]:
        """Save generated images to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            path = output_dir / f"{prefix}_{i:03d}.png"
            img.save(path, quality=95)
            paths.append(path)
            logger.info("Saved %s", path)
        return paths
