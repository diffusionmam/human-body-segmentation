"""MediaPipe Tasks Pose Landmarker wrapper."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.core.body_parts import LM, Landmark


class PoseDetector:
    """Thin wrapper around mediapipe.tasks.vision.PoseLandmarker.

    Detects up to `num_poses` (default 1) and converts the first person's
    landmarks into a dict indexed by semantic name.
    """

    def __init__(self, num_poses: int = 1):
        import mediapipe as mp

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = str(settings.pose_model_path)
        if not settings.pose_model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. "
                f"Run `bash scripts/download_models.sh` first."
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=settings.pose_min_detection_confidence,
            min_pose_presence_confidence=settings.pose_min_detection_confidence,
            min_tracking_confidence=settings.pose_min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def detect(self, bgr_image: np.ndarray) -> Optional[Dict[str, Landmark]]:
        """Return landmarks for the first detected person, or None."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2_to_srgb(bgr_image))
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        out: Dict[str, Landmark] = {}
        for name, idx in LM.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                out[name] = Landmark(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(lm.visibility),
                    presence=float(getattr(lm, "presence", lm.visibility)),
                )
        return out


def cv2_to_srgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to SRGB (MediaPipe expects this layout)."""
    import cv2

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
