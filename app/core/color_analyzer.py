"""Color analysis pipeline: extract dominant colors for a body part mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.utils.colors import classify_hsv, hsv_to_rgb


@dataclass
class ColorStat:
    name: str
    rgb: Tuple[int, int, int]
    hsv: Tuple[int, int, int]
    percentage: float


@dataclass
class ColorAnalysis:
    dominant: ColorStat
    distribution: List[ColorStat]  # sorted by percentage desc
    pixel_count: int


def analyze_part_colors(
    bgr_image: np.ndarray,
    mask: np.ndarray,
    k: int = 5,
    max_samples: int = 5000,
) -> Optional[ColorAnalysis]:
    """Run K-means in HSV on the part pixels and classify each cluster.

    Returns None if the mask has no pixels.
    """
    if mask is None or not mask.any():
        return None

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    pixels_hsv = hsv[mask]
    pixel_count = int(pixels_hsv.shape[0])
    if pixel_count == 0:
        return None

    # Subsample for speed when part is very large.
    if pixel_count > max_samples:
        idx = np.random.choice(pixel_count, size=max_samples, replace=False)
        sample = pixels_hsv[idx]
    else:
        sample = pixels_hsv

    sample_f32 = np.float32(sample)

    # k-means in HSV; OpenCV requires k <= n_samples.
    k_eff = max(1, min(k, sample_f32.shape[0]))
    if k_eff == 1:
        # Single-cluster shortcut.
        labels = np.zeros(sample_f32.shape[0], dtype=np.int32)
        centers = np.array([sample_f32.mean(axis=0)], dtype=np.float32)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        attempts = 3
        _, labels, centers = cv2.kmeans(
            sample_f32,
            k_eff,
            None,
            criteria,
            attempts,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.flatten()

    # Map cluster centers back to all part pixels to get real percentages.
    # We quantise each part pixel to its nearest center using a coarse
    # histogram-like assignment to keep cost low.
    center_int = centers.astype(np.int32)  # k_eff x 3
    # Build per-cluster counts on the full mask.
    cluster_counts = np.zeros(k_eff, dtype=np.int64)
    # Vectorised nearest-center assignment is feasible because k_eff is small.
    diffs = pixels_hsv.astype(np.int32)[:, None, :] - center_int[None, :, :]
    dists = np.sum(diffs * diffs, axis=2)
    nearest = np.argmin(dists, axis=1)
    counts = np.bincount(nearest, minlength=k_eff)
    cluster_counts = counts.astype(np.int64)

    total = max(1, int(cluster_counts.sum()))
    cluster_stats: List[ColorStat] = []
    for i in range(k_eff):
        h, s, v = (int(round(c)) for c in centers[i])
        h = max(0, min(179, h))
        s = max(0, min(255, s))
        v = max(0, min(255, v))
        name = classify_hsv(h, s, v)
        rgb = hsv_to_rgb(h, s, v)
        pct = float(cluster_counts[i]) / float(total) * 100.0
        cluster_stats.append(ColorStat(name=name, rgb=rgb, hsv=(h, s, v), percentage=pct))

    cluster_stats.sort(key=lambda c: c.percentage, reverse=True)
    return ColorAnalysis(
        dominant=cluster_stats[0],
        distribution=cluster_stats,
        pixel_count=pixel_count,
    )
