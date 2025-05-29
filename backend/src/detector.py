from src.core import Detector, Classifier
from src.config import TaiyoConfig
from src.utils import logger
from src.utils.common import visualize_objects


class TaiyoJiguDetector:
    """
    Combines classification and detection pipeline.
    If classification is enabled and object detected, runs detector.
    Always returns a tuple: (results: list, is_obj: bool).
    """
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.classifier = Classifier(config)
        self.detector = Detector(config)

    def run(self, image) -> tuple[list, bool]:
        results = []
        is_obj = True  # default: assume object present

        # Classification stage
        if getattr(self.config.Classifier, 'ENABLE', False):
            try:
                is_obj = self.classifier.run(image)
                logger.info(f"[TaiyoJiguDetector] Classifier result: {is_obj}")
            except Exception as e:
                logger.error(f"[TaiyoJiguDetector] Classification error: {e}")
                # If classification fails, skip detection and return no results
                return results, False

        # Detection stage (only if classified as object or classification disabled)
        if is_obj:
            try:
                results = self.detector.run(image)
                logger.info(f"[TaiyoJiguDetector] Detection returned {len(results)} objects")
            except Exception as e:
                logger.error(f"[TaiyoJiguDetector] Detection error: {e}")

        if results and getattr(self.config, 'VISUALIZE', False):
            _ = visualize_objects(image, results)

        return results, is_obj
