from src.core import Detector, Classifier
from src.config import TaiyoConfig
from src.utils.logger import Logger
from src.utils.common import visualize_objects

class TaiyoJiguDetector:
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.logger = Logger(config)
        self.classifier = Classifier(config)
        self.detector = Detector(config)

    def run(self, image):
        results = []
        is_obj = self.classifier.run(image)
        if is_obj:
            print("This image contains object")
            results = self.detector.run(image)
        return results, is_obj