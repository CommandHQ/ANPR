from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, List, Dict
from app.services.ocr_service import OCRService
from app.config.settings import settings
import gc

class ImageProcessor:
    def __init__(self):
        # Load model with memory optimization
        self.model = YOLO(settings.MODEL_PATH)
        # Configure model for memory efficiency
        self.model.overrides['verbose'] = False
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.ocr_service = OCRService()

    def _resize_image_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it's too large to save memory"""
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

    def detect_license_plates(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        # Resize image to save memory
        image = self._resize_image_if_needed(image)
        
        # Use smaller inference size for memory efficiency
        results = self.model(
            image, 
            conf=self.confidence_threshold,
            imgsz=settings.YOLO_IMGSZ,
            device='cpu',  # Ensure CPU usage
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    detections.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})
        
        # Clear GPU cache if available and force garbage collection
        gc.collect()
        
        return detections, image

    def preprocess_license_plate(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        intermediate_steps = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        intermediate_steps.append(('Binarized', binary.copy()))
        return binary, intermediate_steps

    def process_image(self, image: np.ndarray, min_confidence: float = settings.MIN_OCR_CONFIDENCE) -> Dict:
        try:
            detections, original_image = self.detect_license_plates(image)
            if not detections:
                return {
                    'success': True,
                    'detections_count': 0,
                    'message': 'No license plates detected',
                    'results': []
                }

            results = []
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                det_confidence = detection['confidence']
                x1, y1, x2, y2 = bbox
                height, width = original_image.shape[:2]
                padding = 5  # Reduced padding to save memory
                x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding)

                cropped_plate = original_image[y1:y2, x1:x2]
                if cropped_plate.size == 0:
                    continue

                preprocessed, _ = self.preprocess_license_plate(cropped_plate)
                ocr_result = self.ocr_service.extract_text(preprocessed)
                results.append({
                    'detection_id': i,
                    'bbox': bbox,
                    'detection_confidence': det_confidence,
                    'text': ocr_result['text'],
                    'confidence': ocr_result['confidence']
                })

            # Force garbage collection after processing
            gc.collect()

            return {
                'success': True,
                'detections_count': len(detections),
                'results': results
            }
        except Exception as e:
            # Force cleanup on error
            gc.collect()
            raise RuntimeError(f"Image processing failed: {str(e)}")