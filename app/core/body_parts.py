"""Body part definitions and the 12-part mapping used throughout the app.

We use two complementary signals:

* MediaPipe Image Segmenter (`selfie_multiclass`):
    0 = background, 1 = hair, 2 = body-skin, 3 = face-skin,
    4 = clothes,      5 = others (accessories)
* MediaPipe Pose Landmarker (33 landmarks):

  Indices used here (see
  https://developers.google.com/mediapipe/solutions/vision/pose_landmarker):
    0  = nose
    1,2,3,4   = right eye inner/outer + left eye (mirror)
    5,6       = shoulders
    7,8       = elbows
    9,10      = wrists
    11,12     = hips
    13,14     = knees
    15,16     = ankles
    17,18     = heels
    19,20     = foot index
    21,22     = hand pinky (used for hand bbox extension)

The 12 requested parts are derived from these.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.utils.geometry import BBox, bbox_from_points, split_bbox_horizontally, split_bbox_vertically


# Selfie Multiclass class indices.
SEG_BG = 0
SEG_HAIR = 1
SEG_BODY_SKIN = 2
SEG_FACE_SKIN = 3
SEG_CLOTHES = 4
SEG_OTHERS = 5


# Pose landmark indices (kept here for clarity).
LM = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


def landmark_to_xy(lm) -> Tuple[float, float]:
    return float(lm.x), float(lm.y)


@dataclass
class PartDef:
    name: str
    # Class indices from selfie_multiclass that belong to this part (preferred).
    seg_classes: Tuple[int, ...]
    # Landmark names used to compute the region (bbox / polygon).
    region_landmarks: Tuple[str, ...] = ()
    # Sub-region of the constructed bbox. e.g. ("vertical", "lower") for hand.
    sub_region: Optional[Tuple[str, str]] = None
    # Optional side split: ("horizontal", "left"|"right").
    side: Optional[str] = None
    # Padding around the bbox (fraction of width/height).
    pad: float = 0.05


PARTS: List[PartDef] = [
    PartDef(
        name="face",
        seg_classes=(SEG_FACE_SKIN,),
        region_landmarks=("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                          "mouth_left", "mouth_right"),
        pad=0.10,
    ),
    PartDef(
        name="hair",
        seg_classes=(SEG_HAIR,),
        region_landmarks=("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
        pad=0.25,
    ),
    PartDef(
        name="neck",
        # body-skin / face-skin in the region between head and shoulders
        seg_classes=(SEG_BODY_SKIN, SEG_FACE_SKIN, SEG_CLOTHES),
        region_landmarks=("nose", "left_shoulder", "right_shoulder"),
        pad=0.0,
    ),
    PartDef(
        name="torso",
        seg_classes=(SEG_CLOTHES, SEG_BODY_SKIN),
        region_landmarks=("left_shoulder", "right_shoulder",
                          "left_hip", "right_hip"),
        pad=0.05,
    ),
    PartDef(
        name="left_arm",
        seg_classes=(SEG_BODY_SKIN, SEG_CLOTHES),
        region_landmarks=("left_shoulder", "left_elbow", "left_wrist"),
        side="left",
        pad=0.05,
    ),
    PartDef(
        name="right_arm",
        seg_classes=(SEG_BODY_SKIN, SEG_CLOTHES),
        region_landmarks=("right_shoulder", "right_elbow", "right_wrist"),
        side="right",
        pad=0.05,
    ),
    PartDef(
        name="left_hand",
        seg_classes=(SEG_BODY_SKIN,),
        region_landmarks=("left_wrist", "left_pinky", "left_index", "left_thumb"),
        side="left",
        pad=0.20,
    ),
    PartDef(
        name="right_hand",
        seg_classes=(SEG_BODY_SKIN,),
        region_landmarks=("right_wrist", "right_pinky", "right_index", "right_thumb"),
        side="right",
        pad=0.20,
    ),
    PartDef(
        name="left_leg",
        seg_classes=(SEG_CLOTHES, SEG_BODY_SKIN),
        region_landmarks=("left_hip", "left_knee", "left_ankle"),
        side="left",
        pad=0.05,
    ),
    PartDef(
        name="right_leg",
        seg_classes=(SEG_CLOTHES, SEG_BODY_SKIN),
        region_landmarks=("right_hip", "right_knee", "right_ankle"),
        side="right",
        pad=0.05,
    ),
    PartDef(
        name="left_foot",
        seg_classes=(SEG_CLOTHES, SEG_BODY_SKIN),
        region_landmarks=("left_ankle", "left_heel", "left_foot_index"),
        side="left",
        pad=0.10,
    ),
    PartDef(
        name="right_foot",
        seg_classes=(SEG_CLOTHES, SEG_BODY_SKIN),
        region_landmarks=("right_ankle", "right_heel", "right_foot_index"),
        side="right",
        pad=0.10,
    ),
]


def _visible(landmarks: Dict[str, Landmark], names: Sequence[str], thresh: float = 0.5) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for n in names:
        lm = landmarks.get(n)
        if lm is None:
            continue
        if lm.visibility >= thresh or lm.presence >= thresh:
            pts.append(landmark_to_xy(lm))
    return pts


def compute_part_bbox(
    part: PartDef,
    landmarks: Dict[str, Landmark],
    image_w: int,
    image_h: int,
) -> Optional[BBox]:
    """Compute the (x_min, y_min, x_max, y_max) region for a body part.

    Coordinates are in image pixels (origin = top-left).
    """
    pts = _visible(landmarks, part.region_landmarks)
    if not pts:
        return None

    # Convert normalised landmarks to pixel coordinates.
    pixel_pts: List[Tuple[float, float]] = [(x * image_w, y * image_h) for x, y in pts]
    bbox = bbox_from_points(pixel_pts, image_w, image_h, pad_x=part.pad, pad_y=part.pad)
    if bbox is None:
        return None

    if part.side == "left":
        bbox = split_bbox_horizontally(bbox, "left")
    elif part.side == "right":
        bbox = split_bbox_horizontally(bbox, "right")

    if part.sub_region is not None:
        kind, where = part.sub_region
        if kind == "vertical":
            portion = 1 / 3 if where == "lower" else 1 / 2
            bbox = split_bbox_vertically(bbox, portion)
    return bbox


def part_mask(part: PartDef, seg_mask: np.ndarray, bbox: BBox) -> np.ndarray:
    """Return the boolean mask for `part` inside `bbox`.

    Combines the bbox region with whichever segmentation classes belong to it.
    The output has the same shape as the full image; only pixels inside bbox
    and matching one of the part's classes are True.
    """
    h, w = seg_mask.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
    if x1 <= x0 or y1 <= y0:
        return mask
    region = seg_mask[y0:y1, x0:x1]
    if region.ndim == 3:
        region = region[:, :, 0]
    combined = np.isin(region, part.seg_classes)
    mask[y0:y1, x0:x1] = combined
    return mask


def part_confidence(part: PartDef, landmarks: Dict[str, Landmark]) -> float:
    """Mean visibility of the part's landmarks (0..1)."""
    vals: List[float] = []
    for n in part.region_landmarks:
        lm = landmarks.get(n)
        if lm is None:
            continue
        vals.append(max(lm.visibility, lm.presence))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))
