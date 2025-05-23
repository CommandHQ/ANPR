import cv2
import numpy as np
import base64
import requests
from typing import Optional
import gc
from app.config.settings import settings

def decode_base64_image(image_base64: str) -> Optional[np.ndarray]:
    try:
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Immediately resize if too large to save memory
        image = _resize_if_needed(image)
        
        # Clean up
        del image_data, image_array
        gc.collect()
        
        return image
    except Exception as e:
        gc.collect()
        raise ValueError(f"Invalid base64 image: {str(e)}")

def fetch_image_from_url(image_url: str) -> Optional[np.ndarray]:
    try:
        # Use streaming with limited content length
        response = requests.get(
            image_url, 
            stream=True, 
            timeout=10,
            headers={'Range': 'bytes=0-10485760'}  # Limit to 10MB
        )
        response.raise_for_status()
        
        # Read content in chunks to avoid loading huge images
        content = b''
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 10485760:  # 10MB limit
                raise ValueError("Image too large (>10MB)")
        
        image_array = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to fetch or decode image from URL")
        
        # Immediately resize if too large to save memory
        image = _resize_if_needed(image)
        
        # Clean up
        del content, image_array
        gc.collect()
        
        return image
    except requests.RequestException as e:
        gc.collect()
        raise ValueError(f"Failed to fetch image from URL: {str(e)}")

def _resize_if_needed(image: np.ndarray) -> np.ndarray:
    """Resize image if it exceeds maximum dimensions to save memory"""
    height, width = image.shape[:2]
    max_size = settings.MAX_IMAGE_SIZE
    
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image