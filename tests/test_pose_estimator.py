"""Tests for pose estimation utilities."""

from PIL import Image

from stylesync.imaging.pose_estimator import generate_default_pose


def test_generate_default_pose_front():
    pose = generate_default_pose("front", size=(768, 1024))
    assert pose.size == (768, 1024)
    assert pose.mode == "RGB"
    # Pose skeleton should have non-zero pixels (not fully black)
    pixels = list(pose.getdata())
    non_black = [p for p in pixels if p != (0, 0, 0)]
    assert len(non_black) > 0


def test_generate_default_pose_three_quarter():
    pose = generate_default_pose("3/4", size=(768, 1024))
    assert pose.size == (768, 1024)
    pixels = list(pose.getdata())
    non_black = [p for p in pixels if p != (0, 0, 0)]
    assert len(non_black) > 0


def test_generate_default_pose_unknown_falls_back():
    """Unknown pose type should still produce an image (empty skeleton)."""
    pose = generate_default_pose("overhead", size=(512, 512))
    assert pose.size == (512, 512)
