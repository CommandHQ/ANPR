from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from mangum import Mangum
from app.models.request import ImageRequest
from app.models.response import OCRResponse
from app.services.image_processor import ImageProcessor
from app.utils.image_utils import decode_base64_image, fetch_image_from_url
from app.utils.logger import logger
import cv2
import numpy as np
import gc

app = FastAPI(
    title="Indian License Plate OCR API",
    description="API for detecting and recognizing Indian license plates using YOLOv8 and PaddleOCR.",
    version="1.0.0"
)

# Initialize the image processor
image_processor = ImageProcessor()

@app.post("/ocr", response_model=OCRResponse)
async def process_image(request: ImageRequest):
    """
    Process an image to detect and recognize Indian license plates.
    Accepts either a base64-encoded image or an image URL.
    """
    try:
        if request.image_base64 and request.image_url:
            raise HTTPException(status_code=400, detail="Provide either image_base64 or image_url, not both")
        elif not (request.image_base64 or request.image_url):
            raise HTTPException(status_code=400, detail="Either image_base64 or image_url must be provided")

        # Load the image
        if request.image_base64:
            logger.info("Processing base64 image")
            image = decode_base64_image(request.image_base64)
        else:
            logger.info(f"Fetching image from URL: {request.image_url}")
            image = fetch_image_from_url(request.image_url)

        # Process the image
        logger.info("Starting image processing")
        result = image_processor.process_image(image)
        logger.info(f"Image processing completed: {result}")

        # Clean up image from memory
        del image
        gc.collect()

        return OCRResponse(**result)

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        gc.collect()
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        gc.collect()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ocr/upload", response_model=OCRResponse)
async def process_image_upload(file: UploadFile = File(...)):
    """
    Process an uploaded image file to detect and recognize Indian license plates.
    """
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode uploaded image")

        # Clean up contents and array immediately
        del contents, image_array
        gc.collect()

        # Process the image
        logger.info("Starting image processing")
        result = image_processor.process_image(image)
        logger.info(f"Image processing completed: {result}")

        # Clean up image from memory
        del image
        gc.collect()

        return OCRResponse(**result)

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        gc.collect()
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        gc.collect()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "ANPR OCR API"}

# Create the Lambda handler using Mangum (for AWS Lambda compatibility)
handler = Mangum(app, lifespan="off")