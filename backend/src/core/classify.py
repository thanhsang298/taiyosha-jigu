import numpy as np
import cv2
from ultralytics import YOLO

from src.config import TaiyoConfig
from src.utils.common import crop_center
from src.utils import logger



class Classifier:
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.imgW = config.API.IMG_WIDTH
        self.imgH = config.API.IMG_HEIGHT
        self.img_size = config.Classifier.IMG_SIZE
        self.model_path = config.Classifier.MODEL_PATH
        self.device = config.Classifier.DEVICE
        self.conf = config.Classifier.CONF
        self.debug = config.Classifier.DEBUG
        self.vis_path = config.Classifier.VIS_PATH
        self.model = self._load_model() 
    
    def _load_model(self):
        try:
            logger.info(f"Loading Classifier model from {self.model_path} on device '{self.device}'...")
            model = YOLO(self.model_path)
            model.to(self.device)
            logger.info(f"Loaded Classifier successfully.")
            return model
            
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}. Please check the path.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            raise

    def _cropbox(self, image: np.ndarray) -> np.ndarray:
        """
        Private method to crop a specified region from a NumPy array image.
        :param image: NumPy array representing an image.
        :param crop_box: Tuple (x1, y1, x2, y2) specifying the crop region.
        :return: Cropped NumPy array image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        
        x1, y1, x2, y2 = self.crop_box
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image


    def run(self, image):
        cropped_img  = crop_center(image, self.imgH, self.imgH, new_height=self.img_size, new_width=self.img_size)
        results = self.model(cropped_img, verbose=False)
        
        for result in results:
            predicted_class_index = result.probs.top1
            
        if self.debug:
            cv2.imwrite(self.vis_path, cropped_img)
        return int(predicted_class_index)
