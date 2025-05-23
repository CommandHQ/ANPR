from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import Dict

class OCRService:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            det_db_box_thresh=0.3,
            det_db_score_mode="slow",
            det_db_unclip_ratio=2.0,
            rec_char_type="en",
            drop_score=0.3
        )

    def extract_text(self, image: np.ndarray) -> Dict[str, float]:
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = self.ocr.ocr(image, cls=True)
            lines = []
            if result and isinstance(result, list) and len(result) > 0:
                for line in result:
                    for subline in line:
                        text = subline[1][0].strip()
                        confidence = subline[1][1]
                        if 'ind' not in text.lower():
                            lines.append((text, confidence))

            if lines:
                combined_text = ''.join(line[0] for line in lines)
                combined_confidence = sum(line[1] for line in lines) / len(lines)
                return {'text': combined_text, 'confidence': combined_confidence}
            return {'text': '', 'confidence': 0.0}
        except Exception as e:
            raise RuntimeError(f"PaddleOCR error: {str(e)}")