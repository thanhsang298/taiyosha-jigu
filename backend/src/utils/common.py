from typing import List, Dict
import numpy as np
import cv2

def visualize_objects(image: np.array, obj: List[Dict], output_path: str) -> None:
    for item in obj:
        bbox = item['bbox']
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Extract confidence and class name
        conf = item['conf']
        class_name = item['class_name']

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Create the text label
        label = f'{class_name}: {conf:.2f}'

        # Put the text label on the image
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)

