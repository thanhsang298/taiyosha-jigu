import numpy as np
from src.config import TaiyoConfig
from ultralytics import YOLO
from src.utils.common import visualize_objects
from src.utils import logger


class Detector:
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.img_size = config.Detector.IMG_SIZE
        self.model_path = config.Detector.MODEL_PATH
        self.device = config.Detector.DEVICE
        self.crop_box = config.Detector.CROP_BOX
        self.conf = config.Detector.CONF
        self.debug = config.Detector.DEBUG
        self.vis_path = config.Detector.VIS_PATH
        self.model = self._load_model() 
    
    def _load_model(self):
        try:
            logger.info(f"Loading detector model from {self.model_path}...")
            model = YOLO(self.model_path)
            model.to(self.device)
            logger.info(f"Loaded detector successfully on device '{model.device.type}'.")
            return model
            
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}. Please check the path.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            raise

    def _preprocess(self, image):
        pass
    
    def _cropbox(self, image: np.ndarray) -> np.ndarray:
        """
        Private method to crop a specified region from a NumPy array image.
        :param image: NumPy array representing an image.
        :param crop_box: Tuple (x1, y1, x2, y2) specifying the crop region.
        :return: Cropped NumPy array image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("[Preprocess] Input image must be a numpy array")
        
        x1, y1, x2, y2 = self.crop_box
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image


    def run(self, image):
        detections = []
        try:
            cropped_img  = self._cropbox(image)
            results = self.model.predict(cropped_img, imgsz=(self.img_size,self.img_size), conf=self.conf, stream=True, verbose=False)

            for result in results:
                for det in result.boxes.data:
                    x1, y1, x2, y2 = det[:4].tolist()
                    
                    # Add crop_box offset to transform coordinates back to original image space
                    orig_x1 = x1 + self.crop_box[0]  
                    orig_y1 = y1 + self.crop_box[1] 
                    orig_x2 = x2 + self.crop_box[0]  
                    orig_y2 = y2 + self.crop_box[1] 
                    
                    detection = {
                        'bbox': [orig_x1, orig_y1, orig_x2, orig_y2],
                        'conf': float(det[4]),
                        'class_name': result.names[int(det[5])]
                    }
                    detections.append(detection)
            if self.debug:
                visualize_objects(image=image, obj=detections, output_path=self.vis_path)
            return detections
        except Exception as e:
            logger.error(f"[Detect] Error due to: {e}")
        return detections
