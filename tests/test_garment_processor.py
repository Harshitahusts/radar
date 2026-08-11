"""Tests for garment image preprocessing."""

import numpy as np
from PIL import Image

from stylesync.imaging.garment_processor import GarmentProcessor


def _make_test_image(size=(200, 300), color=(255, 0, 0)):
    """Create a simple test RGBA image."""
    img = Image.new("RGBA", size, (*color, 255))
    return img


def test_create_garment_mask():
    # Create an image with alpha channel (simulating bg removal)
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    # Set top-left quadrant to transparent
    pixels = img.load()
    for x in range(50):
        for y in range(50):
            pixels[x, y] = (0, 0, 0, 0)

    mask = GarmentProcessor.create_garment_mask(img)
    assert mask.mode == "L"
    arr = np.array(mask)
    # Top-left should be black (transparent)
    assert arr[0, 0] == 0
    # Bottom-right should be white (opaque)
    assert arr[99, 99] == 255


def test_normalize_garment():
    img = _make_test_image(size=(400, 600))
    normalized = GarmentProcessor.normalize_garment(img, target_size=(768, 1024))
    assert normalized.size == (768, 1024)


def test_extract_logo_region():
    img = _make_test_image(size=(768, 1024))
    mask = Image.new("L", (768, 1024), 255)

    crop, bbox = GarmentProcessor.extract_logo_region(img, mask, "center-chest")
    x1, y1, x2, y2 = bbox
    assert x2 > x1
    assert y2 > y1
    assert crop.size == (x2 - x1, y2 - y1)


def test_create_inpainting_mask_with_logo():
    target_size = (768, 1024)
    garment_bbox = (0, 0, 768, 1024)
    logo_bbox = (200, 150, 550, 560)

    mask = GarmentProcessor.create_inpainting_mask(target_size, garment_bbox, logo_bbox)
    arr = np.array(mask)

    # Logo region should be black (protected)
    assert arr[200, 300] == 0
    # Outside logo should be white (regenerate)
    assert arr[0, 0] == 255


def test_create_inpainting_mask_no_logo():
    target_size = (768, 1024)
    mask = GarmentProcessor.create_inpainting_mask(target_size, (0, 0, 768, 1024))
    arr = np.array(mask)
    # Everything should be white (regenerate all)
    assert arr.min() == 255
