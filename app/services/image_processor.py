from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, List, Dict
from app.services.ocr_service import OCRService
from app.config.settings import settings

class ImageProcessor:
    def __init__(self):
        self.model = YOLO(settings.MODEL_PATH)
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.ocr_service = OCRService()

    def detect_license_plates(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        results = self.model(image, conf=self.confidence_threshold)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    detections.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})
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
                padding = 10
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

            return {
                'success': True,
                'detections_count': len(detections),
                'results': results
            }
        except Exception as e:
            raise RuntimeError(f"Image processing failed: {str(e)}")