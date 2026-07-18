"""Unit tests for the K-means + classifier color analysis pipeline."""
import numpy as np

from app.core.color_analyzer import analyze_part_colors


def _solid_bgr(color_bgr, h=200, w=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def test_returns_none_for_empty_mask():
    img = _solid_bgr((50, 50, 200))  # mostly red BGR
    mask = np.zeros((200, 200), dtype=bool)
    assert analyze_part_colors(img, mask) is None


def test_red_image_dominant_red():
    img = _solid_bgr((40, 40, 200))  # BGR: red ~ (200,40,40)
    mask = np.ones((200, 200), dtype=bool)
    res = analyze_part_colors(img, mask, k=3)
    assert res is not None
    assert res.dominant.name == "Red"
    assert res.dominant.percentage > 50.0


def test_blue_image_dominant_blue():
    img = _solid_bgr((200, 100, 30))  # BGR: blue ~ (30,100,200)
    mask = np.ones((200, 200), dtype=bool)
    res = analyze_part_colors(img, mask, k=3)
    assert res is not None
    assert res.dominant.name == "Blue"


def test_distribution_is_sorted():
    img = _solid_bgr((40, 40, 200))
    mask = np.ones((200, 200), dtype=bool)
    res = analyze_part_colors(img, mask, k=3)
    pcts = [c.percentage for c in res.distribution]
    assert pcts == sorted(pcts, reverse=True)
