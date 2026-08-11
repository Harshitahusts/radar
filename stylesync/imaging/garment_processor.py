"""Garment image pre-processing: background removal, segmentation, masking."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove

logger = logging.getLogger(__name__)


class GarmentProcessor:
    """Pre-processes a flat-lay garment photo for downstream try-on generation."""

    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        """Remove the background from a flat-lay garment image using rembg (U2-Net)."""
        result = remove(image)
        logger.info("Background removed — output size %s", result.size)
        return result

    @staticmethod
    def create_garment_mask(image_no_bg: Image.Image) -> Image.Image:
        """Create a binary mask from the background-removed garment image."""
        alpha = image_no_bg.split()[-1]
        mask = alpha.point(lambda p: 255 if p > 20 else 0)
        return mask.convert("L")

    @staticmethod
    def extract_logo_region(garment_image: Image.Image, garment_mask: Image.Image, logo_position: str = "center-chest") -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Extract the logo/graphic region based on position hint."""
        w, h = garment_image.size
        position_rois = {
            "center-chest": (0.25, 0.15, 0.75, 0.55),
            "left-chest": (0.1, 0.15, 0.45, 0.45),
            "back": (0.2, 0.2, 0.8, 0.7),
            "full-front": (0.1, 0.1, 0.9, 0.85),
        }
        roi = position_rois.get(logo_position, position_rois["center-chest"])
        x1, y1 = int(roi[0] * w), int(roi[1] * h)
        x2, y2 = int(roi[2] * w), int(roi[3] * h)
        cropped = garment_image.crop((x1, y1, x2, y2))
        return cropped, (x1, y1, x2, y2)

    @staticmethod
    def create_inpainting_mask(target_size: tuple[int, int], garment_bbox: tuple[int, int, int, int], logo_bbox: tuple[int, int, int, int] | None = None) -> Image.Image:
        """Create an inpainting mask that protects the logo region."""
        mask = Image.new("L", target_size, 255)
        mask_array = np.array(mask)
        if logo_bbox:
            x1, y1, x2, y2 = logo_bbox
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(target_size[0], x2 + pad)
            y2 = min(target_size[1], y2 + pad)
            mask_array[y1:y2, x1:x2] = 0
        return Image.fromarray(mask_array)

    @staticmethod
    def normalize_garment(image: Image.Image, target_size: tuple[int, int] = (768, 1024)) -> Image.Image:
        """Resize and center-pad the garment image to a standard resolution."""
        image.thumbnail(target_size, Image.LANCZOS)
        canvas = Image.new("RGBA", target_size, (255, 255, 255, 0))
        offset_x = (target_size[0] - image.width) // 2
        offset_y = (target_size[1] - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
        return canvas

    def preprocess(self, image_path: str | Path, logo_position: str | None = None) -> dict:
        """Full preprocessing pipeline for a flat-lay garment image."""
        image = Image.open(image_path).convert("RGBA")
        logger.info("Loaded garment image %s (%s)", image_path, image.size)
        clean = self.remove_background(image)
        mask = self.create_garment_mask(clean)
        normalized = self.normalize_garment(clean)
        result = {"garment_clean": clean, "garment_mask": mask, "garment_normalized": normalized, "logo_crop": None, "logo_bbox": None, "inpaint_mask": None}
        if logo_position:
            logo_crop, logo_bbox = self.extract_logo_region(normalized, mask, logo_position)
            inpaint_mask = self.create_inpainting_mask(normalized.size, (0, 0, *normalized.size), logo_bbox)
            result.update(logo_crop=logo_crop, logo_bbox=logo_bbox, inpaint_mask=inpaint_mask)
        return result
