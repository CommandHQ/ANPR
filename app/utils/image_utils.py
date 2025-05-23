import cv2
import numpy as np
import base64
import requests
from typing import Optional

def decode_base64_image(image_base64: str) -> Optional[np.ndarray]:
    try:
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")

def fetch_image_from_url(image_url: str) -> Optional[np.ndarray]:
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to fetch or decode image from URL")
        return image
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch image from URL: {str(e)}")