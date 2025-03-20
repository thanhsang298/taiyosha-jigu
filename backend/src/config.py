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
        IMG_SIZE: int = 1920
        BATCH_SIZE: int = 1
        CONF: float  = 0.25
        CROP_BOX: tuple = (300, 1250, 5167, 2400)
        MODEL_PATH: str = "./weights/detector/best.pt"
        DEBUG: bool = True
        VIS_PATH: str = "./visualization/detector-img.jpg"
        
    class Classifier:
        DEVICE: str = "cuda"
        IMG_SIZE: int = 640
        BATCH_SIZE: int = 1
        CONF: float  = 0.25
        MODEL_PATH: str = "./weights/cls/best.pt"
        DEBUG: bool = True  
        VIS_PATH: str = "./visualization/classifier-img.jpg"