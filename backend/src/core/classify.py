import numpy as np
import cv2
from src.config import TaiyoConfig
from ultralytics import YOLO
from src.utils.common import visualize_objects


class Classifier:
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.img_size = config.Classifier.IMG_SIZE
        self.model_path = config.Classifier.MODEL_PATH
        self.crop_box = config.Classifier.CROP_BOX
        self.device = config.Classifier.DEVICE
        self.conf = config.Classifier.CONF
        self.debug = config.Classifier.DEBUG
        self.vis_path = config.Classifier.VIS_PATH
        self.model = self._load_model() 
    
    def _load_model(self):
        try:
            print(f"------------------------------------> Loading Classifier model from {self.model_path} on device '{self.device}'...")
            model = YOLO(self.model_path)
            print(f"------------------------------------> Loaded Classifier successfully.")
            return model
            
        except FileNotFoundError:
            print(f"Model file not found at {self.model_path}. Please check the path.")
            raise
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
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
        cropped_img  = self._cropbox(image)
        results = self.model(cropped_img, imgsz=(self.img_size,self.img_size), conf=self.conf)
        
        for result in results:
            predicted_class_index = result.probs.top1
            predicted_class_name = result.names[predicted_class_index]
            print(f"Predicted class: {predicted_class_name}")
            
        if self.debug:
            cv2.imwrite(self.vis_path, cropped_img)
        return int(predicted_class_index)
