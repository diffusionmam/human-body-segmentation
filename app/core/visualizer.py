"""Annotated-frame rendering: bounding boxes, labels, color swatches."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.body_parts import PARTS, Landmark, PartDef
from app.core.color_analyzer import ColorAnalysis
from app.utils.colors import NAMED_COLORS
from app.utils.geometry import BBox


# Stable per-part color so the same body part is always drawn the same hue.
PART_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "head":      (255, 200, 100),
    "face":      (255, 220, 150),
    "hair":      (200, 100, 200),
    "neck":      (255, 180, 200),
    "torso":     ( 50, 200, 255),
    "left_arm":  ( 90, 220,  90),
    "right_arm": ( 90, 180,  90),
    "left_hand": (180, 255,  90),
    "right_hand":(140, 200,  90),
    "left_leg":  (255,  90,  90),
    "right_leg": (200,  80,  80),
    "left_foot": (255, 130,  60),
    "right_foot":(220, 110,  50),
}


@dataclass
class PartRender:
    part: PartDef
    bbox: BBox
    confidence: float
    color: ColorAnalysis


def _palette_color(part_name: str) -> Tuple[int, int, int]:
    if part_name in PART_PALETTE:
        return PART_PALETTE[part_name]
    random.seed(part_name)
    return (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.5,
    thickness: int = 1,
    pad: int = 4,
) -> Tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - pad), (x + tw + pad * 2, y + baseline), bg, -1)
    cv2.putText(img, text, (x + pad, y), font, scale, fg, thickness, cv2.LINE_AA)
    return tw + pad * 2, th + pad * 2


def draw_legend(img: np.ndarray, parts: Sequence[PartRender]) -> None:
    """Top-left legend listing detected parts + dominant colors."""
    h, w = img.shape[:2]
    x0, y0 = 12, 12
    line_h = 22
    box = 14
    if not parts:
        cv2.rectangle(img, (x0 - 6, y0 - 6), (w - 12, y0 + line_h + 6), (0, 0, 0), -1)
        cv2.putText(img, "no person detected", (x0, y0 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return

    title = f"{len(parts)} parts detected"
    rows = [title]
    for pr in parts:
        sw = pr.color.dominant.rgb  # already RGB ints
        sw_bgr = (int(sw[2]), int(sw[1]), int(sw[0]))
        rows.append((pr.part.name, pr.color.dominant.name, sw_bgr))

    bg_h = (len(rows)) * line_h + 14
    cv2.rectangle(img, (x0 - 6, y0 - 6), (x0 + 240, y0 + bg_h), (0, 0, 0), -1)

    cv2.putText(img, title, (x0, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    y = y0 + line_h + 14
    if len(rows) > 1:
        for name, cname, sw_bgr in rows[1:]:
            cv2.rectangle(img, (x0, y), (x0 + box, y + box), sw_bgr, -1)
            cv2.rectangle(img, (x0, y), (x0 + box, y + box), (255, 255, 255), 1)
            text = f"{name}: {cname}  {int(pr_for(parts, name).confidence * 100)}%"
            cv2.putText(img, text, (x0 + box + 8, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += line_h


def pr_for(parts: Sequence[PartRender], name: str) -> PartRender:
    for p in parts:
        if p.part.name == name:
            return p
    raise KeyError(name)


def draw_pose_skeleton(
    img: np.ndarray,
    landmarks: Optional[Dict[str, Landmark]],
    seg_mask: Optional[np.ndarray] = None,
) -> None:
    """Draw Pose landmarks and edges for debugging (lightweight)."""
    if landmarks is None:
        return
    h, w = img.shape[:2]

    edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    for a, b in edges:
        la = landmarks.get(a)
        lb = landmarks.get(b)
        if la is None or lb is None:
            continue
        if max(la.visibility, la.presence) < 0.3 or max(lb.visibility, lb.presence) < 0.3:
            continue
        xa, ya = int(la.x * w), int(la.y * h)
        xb, yb = int(lb.x * w), int(lb.y * h)
        cv2.line(img, (xa, ya), (xb, yb), (200, 200, 200), 1, cv2.LINE_AA)
    for name, lm in landmarks.items():
        if max(lm.visibility, lm.presence) < 0.3:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_part_overlay(
    img: np.ndarray,
    part: PartDef,
    bbox: BBox,
    color: ColorAnalysis,
    confidence: float,
    label: bool = True,
) -> None:
    pc = _palette_color(part.name)
    cv2.rectangle(img, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), pc, 2)
    if label:
        sw = color.dominant.rgb
        sw_bgr = (int(sw[2]), int(sw[1]), int(sw[0]))
        sw_x = bbox.x_min
        sw_y = max(0, bbox.y_min - 28)
        cv2.rectangle(img, (sw_x, sw_y), (sw_x + 24, sw_y + 18), sw_bgr, -1)
        cv2.rectangle(img, (sw_x, sw_y), (sw_x + 24, sw_y + 18), (255, 255, 255), 1)
        text = f"{part.name} {color.dominant.name} {int(confidence * 100)}%"
        _draw_text_with_bg(img, text, (sw_x + 30, sw_y + 14), scale=0.5, thickness=1)


def render_annotations(
    img: np.ndarray,
    parts: Sequence[PartRender],
    landmarks: Optional[Dict[str, Landmark]] = None,
) -> np.ndarray:
    """Annotate the image with bounding boxes, swatches, and a legend."""
    out = img.copy()
    if landmarks is not None:
        draw_pose_skeleton(out, landmarks)
    for pr in parts:
        draw_part_overlay(out, pr.part, pr.bbox, pr.color, pr.confidence)
    draw_legend(out, parts)
    return out


def overlay_segmentation_mask(img: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Lightly tint the image by the segmentation mask classes (debug aid)."""
    colors_bgr = np.array([
        (  0,   0,   0),   # background - transparent
        (200, 100, 200),   # hair
        (180, 180, 255),   # body-skin
        (255, 220, 180),   # face-skin
        (180, 255, 180),   # clothes
        (255, 255, 100),   # others
    ], dtype=np.uint8)
    overlay = np.zeros_like(img)
    for cls in range(1, colors_bgr.shape[0]):
        mask = (seg_mask == cls)
        if mask.any():
            overlay[mask] = colors_bgr[cls]
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
