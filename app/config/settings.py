# app/config/settings.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file, if present
load_dotenv()

class Settings:
    # Absolute path to the directory containing this file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of settings.py
    
    # Use the custom trained model
    MODEL_PATH = os.path.join(BASE_DIR, "../../last.pt") 
    # Path to the model file (can be overridden using the MODEL_PATH environment variable)

    # Confidence threshold for object detection (default: 0.5)
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

    # Minimum OCR confidence to consider a result valid (default: 0.3)
    MIN_OCR_CONFIDENCE = float(os.getenv("MIN_OCR_CONFIDENCE", 0.3))

    # Memory optimization settings
    YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", 320))  # Smaller image size for YOLO inference
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1024))  # Max input image size


# Instantiate the settings object
settings = Settings()
