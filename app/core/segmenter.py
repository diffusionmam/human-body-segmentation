"""MediaPipe Tasks Image Segmenter wrapper (selfie_multiclass)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import settings


class BodyPartSegmenter:
    """Wraps MediaPipe's selfie_multiclass 6-class segmenter.

    Output is an H x W int8 array of class indices:
        0=background, 1=hair, 2=body-skin,
        3=face-skin,  4=clothes, 5=others.
    """

    NUM_CLASSES = 6

    def __init__(self):
        import mediapipe as mp

        if not settings.segmenter_model_path.exists():
            raise FileNotFoundError(
                f"Segmenter model not found at {settings.segmenter_model_path}. "
                f"Run `bash scripts/download_models.sh` first."
            )

        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(settings.segmenter_model_path)),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
        )
        self._segmenter = ImageSegmenter.create_from_options(options)
        self._mp = mp

    def close(self) -> None:
        try:
            self._segmenter.close()
        except Exception:
            pass

    def segment(self, bgr_image: np.ndarray) -> np.ndarray:
        """Return H x W uint8 mask of class indices (0..5)."""
        import cv2
        import mediapipe as mp

        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        mask = result.category_mask.numpy_view()  # H x W uint8
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.uint8)
