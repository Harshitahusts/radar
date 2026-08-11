"""Tests for garment image preprocessing."""

import numpy as np
from PIL import Image

from stylesync.imaging.garment_processor import GarmentProcessor


def _make_test_image(size=(200, 300), color=(255, 0, 0)):
    return Image.new("RGBA", size, (*color, 255))


def test_create_garment_mask():
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    pixels = img.load()
    for x in range(50):
        for y in range(50):
            pixels[x, y] = (0, 0, 0, 0)
    mask = GarmentProcessor.create_garment_mask(img)
    assert mask.mode == "L"
    arr = np.array(mask)
    assert arr[0, 0] == 0
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
    mask = GarmentProcessor.create_inpainting_mask(target_size, (0, 0, 768, 1024), (200, 150, 550, 560))
    arr = np.array(mask)
    assert arr[200, 300] == 0
    assert arr[0, 0] == 255


def test_create_inpainting_mask_no_logo():
    target_size = (768, 1024)
    mask = GarmentProcessor.create_inpainting_mask(target_size, (0, 0, 768, 1024))
    arr = np.array(mask)
    assert arr.min() == 255
