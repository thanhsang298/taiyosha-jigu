import logging
from dataclasses import dataclass
from typing import Tuple
from pydantic_settings import BaseSettings


class TaiyoConfig(BaseSettings):
    class API:
        PROJECT_NAME: str = "Taiyo Jigu Detector"
        API_V1_STR: str = "/api/v1"
        PORT: int = 8000
        IMG_HEIGHT: int = 3648
        IMG_WIDTH: int = 5472

    class Log:
        PATH: str = "app.log"

    class LogType:
        DEBUG: int = logging.DEBUG
        INFO: int = logging.INFO
        WARNING: int = logging.WARNING
        ERROR: int = logging.ERROR
        CRITICAL: int = logging.CRITICAL

    class Detector:
        DEVICE: str = "cuda"
        IMG_SIZE: int = 1280
        BATCH_SIZE: int = 1
        CONF: float  = 0.45
        CROP_BOX: tuple = (5, 1250, 5467, 2400)
        MODEL_PATH: str = "./weights/best.pt"
        DEBUG: bool = True
        VIS_PATH: str = "./visualization/debug_img.jpg"
