from src.core import Detector
from src.config import TaiyoConfig
from src.utils.logger import Logger
from src.utils.common import visualize_objects

class TaiyoJiguDetector:
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.logger = Logger(config)
        self.detector = Detector(config)

    def run(self, image):
        results = []
        results = self.detector.run(image)
        return results