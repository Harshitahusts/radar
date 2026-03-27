"""Pose estimation utilities for ControlNet conditioning.

Uses controlnet_aux (OpenPose) to generate skeleton images from reference
model photos, or generates default pose skeletons for common eCommerce poses.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Lazy import — controlnet_aux can be slow to load
_openpose_detector = None


def get_openpose_detector():
    global _openpose_detector
    if _openpose_detector is None:
        from controlnet_aux import OpenposeDetector

        _openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        logger.info("OpenPose detector loaded")
    return _openpose_detector


def extract_pose(reference_image: Image.Image) -> Image.Image:
    """Extract OpenPose skeleton from a reference model photo."""
    detector = get_openpose_detector()
    pose_image = detector(reference_image)
    return pose_image


def generate_default_pose(
    pose_type: str = "front",
    size: tuple[int, int] = (768, 1024),
) -> Image.Image:
    """Generate a simple default pose skeleton for common eCommerce shots.

    This is a fallback when no reference model image is provided.
    The skeleton is drawn programmatically as a basic stick figure.
    """
    img = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    w, h = size
    cx = w // 2

    # Color coding matches OpenPose convention
    joint_color = (255, 0, 0)
    limb_color = (0, 255, 0)
    width = 4

    if pose_type == "front":
        # Head
        head_y = int(h * 0.08)
        neck_y = int(h * 0.15)
        draw.ellipse([cx - 25, head_y - 25, cx + 25, head_y + 25], fill=joint_color)
        # Neck
        draw.line([(cx, head_y + 25), (cx, neck_y)], fill=limb_color, width=width)
        # Shoulders
        ls_x, rs_x = cx - int(w * 0.18), cx + int(w * 0.18)
        draw.line([(ls_x, neck_y), (rs_x, neck_y)], fill=limb_color, width=width)
        # Torso
        hip_y = int(h * 0.45)
        draw.line([(cx, neck_y), (cx, hip_y)], fill=limb_color, width=width)
        # Arms (relaxed at sides)
        elbow_y = int(h * 0.30)
        wrist_y = int(h * 0.42)
        for sx in (ls_x, rs_x):
            draw.line([(sx, neck_y), (sx, elbow_y)], fill=limb_color, width=width)
            draw.line([(sx, elbow_y), (sx, wrist_y)], fill=limb_color, width=width)
        # Legs
        lh_x, rh_x = cx - int(w * 0.08), cx + int(w * 0.08)
        knee_y = int(h * 0.65)
        ankle_y = int(h * 0.88)
        for hx in (lh_x, rh_x):
            draw.line([(hx, hip_y), (hx, knee_y)], fill=limb_color, width=width)
            draw.line([(hx, knee_y), (hx, ankle_y)], fill=limb_color, width=width)

    elif pose_type == "3/4":
        # Slight offset for three-quarter angle
        offset = int(w * 0.05)
        head_y = int(h * 0.08)
        neck_y = int(h * 0.15)
        draw.ellipse(
            [cx + offset - 25, head_y - 25, cx + offset + 25, head_y + 25],
            fill=joint_color,
        )
        draw.line(
            [(cx + offset, head_y + 25), (cx + offset, neck_y)],
            fill=limb_color, width=width,
        )
        ls_x = cx + offset - int(w * 0.15)
        rs_x = cx + offset + int(w * 0.20)
        draw.line([(ls_x, neck_y), (rs_x, neck_y)], fill=limb_color, width=width)
        hip_y = int(h * 0.45)
        draw.line([(cx + offset, neck_y), (cx + offset, hip_y)], fill=limb_color, width=width)
        elbow_y = int(h * 0.30)
        wrist_y = int(h * 0.42)
        for sx in (ls_x, rs_x):
            draw.line([(sx, neck_y), (sx, elbow_y)], fill=limb_color, width=width)
            draw.line([(sx, elbow_y), (sx, wrist_y)], fill=limb_color, width=width)
        lh_x = cx + offset - int(w * 0.06)
        rh_x = cx + offset + int(w * 0.10)
        knee_y = int(h * 0.65)
        ankle_y = int(h * 0.88)
        for hx in (lh_x, rh_x):
            draw.line([(hx, hip_y), (hx, knee_y)], fill=limb_color, width=width)
            draw.line([(hx, knee_y), (hx, ankle_y)], fill=limb_color, width=width)

    return img
