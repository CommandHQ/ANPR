import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Dict
from app.services.ocr_service import OCRService
from app.config.settings import settings

class ImageProcessor:
    def __init__(self):
        self.session = ort.InferenceSession(settings.MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.ocr_service = OCRService()
        self.input_size = (640, 640)  # Adjust based on training

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.input_size)
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def postprocess_output(self, output, original_shape) -> List[Dict]:
        detections = []
        boxes = output[0]
        height, width = original_shape[:2]

        for box in boxes:
            x1, y1, x2, y2, score, class_id = box[:6]
            if score > self.confidence_threshold:
                x1 = int(x1 / self.input_size[0] * width)
                y1 = int(y1 / self.input_size[1] * height)
                x2 = int(x2 / self.input_size[0] * width)
                y2 = int(y2 / self.input_size[1] * height)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score)
                })
        return detections

    def detect_license_plates(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        input_tensor = self.preprocess_input(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess_output(outputs, image.shape)
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
