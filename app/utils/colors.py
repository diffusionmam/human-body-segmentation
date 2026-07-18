"""Named color classification from HSV values.

The 11 categories required by the assignment are mapped from HSV using
hue ranges and saturation/value gates.  Returns are deterministic and
unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# OpenCV HSV ranges: H in [0, 179], S in [0, 255], V in [0, 255].
HUE_MAX = 179


@dataclass(frozen=True)
class ColorEntry:
    name: str
    rgb: Tuple[int, int, int]


# Reference swatch per named color (used for legends/UI, not for classification).
NAMED_COLORS: dict[str, Tuple[int, int, int]] = {
    "Red":    (220,  20,  60),
    "Orange": (255, 140,   0),
    "Yellow": (255, 215,   0),
    "Green":  ( 34, 139,  34),
    "Blue":   ( 30, 100, 200),
    "Purple": (128,   0, 128),
    "Pink":   (255, 105, 180),
    "Brown":  (139,  90,  43),
    "Black":  (  0,   0,   0),
    "White":  (245, 245, 245),
    "Gray":   (128, 128, 128),
}


def _hue_in(h: int, lo: int, hi: int) -> bool:
    """Handle wrap-around at 180 by passing either end in lo/hi."""
    if lo <= hi:
        return lo <= h <= hi
    return h >= lo or h <= hi


def classify_hsv(h: float, s: float, v: float) -> str:
    """Map an HSV point (H in [0,179], S/V in [0,255]) to a named color."""
    h = max(0, min(HUE_MAX, int(round(h))))
    s = max(0, min(255, int(round(s))))
    v = max(0, min(255, int(round(v))))

    # Achromatic checks first — dark / bright / desaturated.
    if v < 50:
        return "Black"
    if v > 215 and s < 20:
        return "White"
    if s < 30 and 50 <= v <= 215:
        return "Gray"

    # Warm hues (0 .. 25) default to Brown *unless* they are vivid enough
    # to be called Red or Orange.  This prevents skin, hair, leather, and
    # neutral browns from leaking into Red/Orange.
    if _hue_in(h, 0, 25):
        if s >= 160 and v >= 100:
            if _hue_in(h, 0, 5):
                return "Red"
            return "Orange"
        return "Brown"

    # Cool / mixed hues: still require a minimum saturation.
    if s < 50:
        return "Gray"

    if _hue_in(h, 170, 179):
        return "Red"
    if _hue_in(h, 26, 40):
        return "Yellow"
    if _hue_in(h, 41, 85):
        return "Green"
    if _hue_in(h, 86, 125):
        return "Blue"
    if _hue_in(h, 126, 155):
        return "Purple"
    if _hue_in(h, 156, 169):
        return "Pink"

    return "Gray"


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV (OpenCV convention) to RGB 0-255 ints."""
    import colorsys

    h_norm = (h / HUE_MAX) * 2.0 * 3.141592653589793
    s_norm = s / 255.0
    v_norm = v / 255.0
    r, g, b = colorsys.hsv_to_rgb(h_norm / (2 * 3.141592653589793), s_norm, v_norm)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return b, g, r
