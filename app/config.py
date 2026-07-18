"""Application configuration loaded from environment variables / .env."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "*",
        ]
    )

    # --- UI ---
    ui_host: str = "0.0.0.0"
    ui_port: int = 8501
    api_base_url: str = "http://localhost:8000"
    ws_base_url: str = "ws://localhost:8000"

    # --- Model ---
    inference_size: int = 384
    pose_model_complexity: int = 1
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5

    # --- Storage ---
    data_dir: Path = Path("./data/jobs")
    models_dir: Path = Path("./models")

    @property
    def pose_model_path(self) -> Path:
        return self.models_dir / "mediapipe" / "pose_landmarker.task"

    @property
    def segmenter_model_path(self) -> Path:
        return self.models_dir / "mediapipe" / "selfie_multiclass.tflite"


settings = Settings()
