import io
import time
import cv2
import numpy as np
from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from src.config import TaiyoConfig
from src.detector import TaiyoJiguDetector
from src.utils.logger import logger
from PIL import Image


api_router = APIRouter()
detector = TaiyoJiguDetector(config=TaiyoConfig)

@api_router.post("/detect")
async def detect(
        file: UploadFile = File(...),
    ):
    
    results = {}
    
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                content={"error": "Failed to decode image."},
                status_code=400
            )

        # Perform inference
        t1 = time.time()

        results = detector.run(image=img)
        
        logger.info(f'Inference API time: {time.time() - t1:.3f}s')

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"[API] Error due to: {e}")
        return JSONResponse(
            content={"error": "Internal server error."},
            status_code=500
        )

