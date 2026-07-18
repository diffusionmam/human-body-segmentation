"""Top-level orchestrator: pose + segmentation + color analysis + drawing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.core.body_parts import (
    Landmark,
    PARTS,
    compute_part_bbox,
    part_confidence,
    part_mask,
)
from app.core.color_analyzer import ColorAnalysis, analyze_part_colors
from app.core.pose import PoseDetector
from app.core.segmenter import BodyPartSegmenter
from app.core.visualizer import PartRender, render_annotations
from app.utils.geometry import BBox


@dataclass
class PartResult:
    part_name: str
    bbox: BBox
    confidence: float
    color: ColorAnalysis


@dataclass
class FrameAnalysis:
    timestamp_ms: int
    person_detected: bool
    parts: List[PartResult] = field(default_factory=list)
    landmarks: Optional[Dict[str, Landmark]] = None
    seg_mask: Optional[np.ndarray] = None


class HumanBodyAnalyzer:
    """Combines pose landmarks, segmentation mask, and color analysis."""

    def __init__(self):
        self.pose = PoseDetector()
        self.segmenter = BodyPartSegmenter()

    def close(self) -> None:
        self.pose.close()
        self.segmenter.close()

    # ---------------------------------------------------------------
    def analyze(self, bgr_image: np.ndarray, timestamp_ms: Optional[int] = None) -> FrameAnalysis:
        h, w = bgr_image.shape[:2]
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        landmarks = self.pose.detect(bgr_image)
        seg_mask = self.segmenter.segment(bgr_image)

        if landmarks is None:
            return FrameAnalysis(
                timestamp_ms=timestamp_ms,
                person_detected=False,
                landmarks=None,
                seg_mask=seg_mask,
            )

        results: List[PartResult] = []
        for part in PARTS:
            bbox = compute_part_bbox(part, landmarks, w, h)
            if bbox is None or bbox.area <= 0:
                continue
            mask = part_mask(part, seg_mask, bbox)
            if not mask.any():
                # Fall back to all pixels in the bbox if no segmentation class
                # matched (e.g. the multiclass model classified them as
                # something unexpected).  Keeps detection robust.
                x0, y0, x1, y1 = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
                if x1 > x0 and y1 > y0:
                    fallback = np.zeros((h, w), dtype=bool)
                    fallback[y0:y1, x0:x1] = True
                    mask = fallback
                else:
                    continue
            color = analyze_part_colors(bgr_image, mask)
            if color is None:
                continue
            conf = part_confidence(part, landmarks)
            results.append(PartResult(
                part_name=part.name,
                bbox=bbox,
                confidence=conf,
                color=color,
            ))

        return FrameAnalysis(
            timestamp_ms=timestamp_ms,
            person_detected=True,
            parts=results,
            landmarks=landmarks,
            seg_mask=seg_mask,
        )

    # ---------------------------------------------------------------
    def annotate(self, bgr_image: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        renders: List[PartRender] = [
            PartRender(
                part=part,  # type: ignore[arg-type]
                bbox=p.bbox,
                confidence=p.confidence,
                color=p.color,
            )
            for p, part in zip(analysis.parts, PARTS)
            if p.part_name == part.name
        ]
        return render_annotations(bgr_image, renders, analysis.landmarks)
