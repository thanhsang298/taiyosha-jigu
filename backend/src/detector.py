from src.core import Detector, Classifier
from src.config import TaiyoConfig
from src.utils import logger
from src.utils.common import visualize_objects


class TaiyoJiguDetector:
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.detector = Detector(config)

    def run(self, image):
        results = []
        try:
            results = self.detector.run(image)
        except Exception as e:
            logger.error(f"[Detector] Error due to: {e}")
        return results