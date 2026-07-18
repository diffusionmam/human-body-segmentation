"""Geometry helpers for body part region computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2

    def as_dict(self) -> dict:
        return {
            "x_min": int(self.x_min),
            "y_min": int(self.y_min),
            "x_max": int(self.x_max),
            "y_max": int(self.y_max),
        }

    def clamp(self, w: int, h: int) -> "BBox":
        return BBox(
            x_min=max(0, min(self.x_min, w - 1)),
            y_min=max(0, min(self.y_min, h - 1)),
            x_max=max(0, min(self.x_max, w - 1)),
            y_max=max(0, min(self.y_max, h - 1)),
        )


def bbox_from_points(
    points: Sequence[Tuple[float, float]],
    w: int,
    h: int,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> Optional[BBox]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    bw = max(1, x_max - x_min)
    bh = max(1, y_max - y_min)
    x_min -= pad_x * bw
    x_max += pad_x * bw
    y_min -= pad_y * bh
    y_max += pad_y * bh
    return BBox(
        x_min=int(round(x_min)),
        y_min=int(round(y_min)),
        x_max=int(round(x_max)),
        y_max=int(round(y_max)),
    ).clamp(w, h)


def bbox_from_mask(mask: np.ndarray, pad: int = 2) -> Optional[BBox]:
    """Return the tight bounding box of a boolean mask."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    h, w = mask.shape[:2]
    return BBox(
        x_min=max(0, x_min - pad),
        y_min=max(0, y_min - pad),
        x_max=min(w - 1, x_max + pad),
        y_max=min(h - 1, y_max + pad),
    )


def split_bbox_horizontally(bbox: BBox, side: str) -> BBox:
    """Return a left or right half of a bbox (used for L/R separation)."""
    x_min, y_min, x_max, y_max = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
    mid = (x_min + x_max) // 2
    if side == "left":
        return BBox(x_min=x_min, y_min=y_min, x_max=mid, y_max=y_max)
    return BBox(x_min=mid, y_min=y_min, x_max=x_max, y_max=y_max)


def split_bbox_vertically(bbox: BBox, portion: float) -> BBox:
    """Return the upper (portion<0.5) or lower (portion>=0.5) slice of a bbox."""
    x_min, y_min, x_max, y_max = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
    h_total = max(1, y_max - y_min)
    cut = y_min + int(h_total * portion)
    if portion < 0.5:
        return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=cut)
    return BBox(x_min=x_min, y_min=cut, x_max=x_max, y_max=y_max)


def line_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
