"""Unit tests for the HSV -> named color classifier."""
from app.utils.colors import classify_hsv


def test_achromatic():
    assert classify_hsv(0, 0, 0) == "Black"
    assert classify_hsv(0, 10, 20) == "Black"
    assert classify_hsv(60, 30, 40) == "Black"
    assert classify_hsv(0, 5, 230) == "White"
    assert classify_hsv(60, 5, 120) == "Gray"


def test_red_extremes():
    assert classify_hsv(0, 200, 200) == "Red"
    assert classify_hsv(175, 200, 200) == "Red"


def test_primary_hues():
    assert classify_hsv(15, 200, 200) == "Orange"
    assert classify_hsv(28, 200, 200) == "Yellow"
    assert classify_hsv(60, 200, 200) == "Green"
    assert classify_hsv(110, 200, 200) == "Blue"
    assert classify_hsv(140, 200, 200) == "Purple"
    assert classify_hsv(165, 200, 200) == "Pink"


def test_brown():
    # Dark-warm, low V -> Brown
    assert classify_hsv(15, 120, 60) == "Brown"
    # Moderate saturation, warm hue -> Brown (skin, dirt, leather)
    assert classify_hsv(10, 60, 150) == "Brown"
    assert classify_hsv(4, 80, 210) == "Brown"


def test_skin_not_red():
    # Skin tones should never be "Red".
    # Fair skin: H ~4-15, weak S, high V.
    assert classify_hsv(4, 80, 210) == "Brown"
    assert classify_hsv(10, 50, 190) == "Brown"
    assert classify_hsv(15, 40, 180) == "Brown"
    # Medium-dark skin
    assert classify_hsv(8, 70, 120) == "Brown"
    # Vivid red clothing / lipstick still returns Red.
    assert classify_hsv(2, 180, 190) == "Red"
    assert classify_hsv(10, 200, 150) == "Orange"


def test_wrap_around():
    assert classify_hsv(178, 200, 200) == "Red"
    assert classify_hsv(0, 200, 200) == "Red"
