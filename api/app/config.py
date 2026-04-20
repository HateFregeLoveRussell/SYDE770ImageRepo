from __future__ import annotations

import pathlib
from functools import lru_cache

from pydantic_settings import BaseSettings

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    models_dir: pathlib.Path = _REPO_ROOT / "models"
    default_model: str = "yolov8s_v3_260"
    default_conf: float = 0.40
    default_iou: float = 0.7
    img_size: int = 640
    batch_size: int = 16
    host: str = "0.0.0.0"
    port: int = 6045

    # Group info
    group_name: str = "group5"
    group_members: list[str] = ["Liam Markus Prieditis", "Samyar Goordazi", "Zaid Mubeen", "Mohamed Ali Mourtada"]

    # Monitoring
    evidently_window_size: int = 200

    model_config = {"env_prefix": "API_"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
