from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import Dict
import gc

class OCRService:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            use_gpu=False,
            show_log=False,
            det_db_box_thresh=0.5,
            det_db_score_mode="fast",
            det_db_unclip_ratio=1.5,
            rec_char_type="en",
            drop_score=0.5,
            det_limit_side_len=960,
            rec_batch_num=1,
        )

    def extract_text(self, image: np.ndarray) -> Dict[str, float]:
        try:
            if image.shape[0] > 480 or image.shape[1] > 640:
                height, width = image.shape[:2]
                scale = min(480/height, 640/width)
                new_height, new_width = int(height * scale), int(width * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = self.ocr.ocr(image, cls=False)
            lines = []
            if result and isinstance(result, list) and len(result) > 0:
                for line in result:
                    if line:
                        for subline in line:
                            if subline and len(subline) >= 2:
                                text = subline[1][0].strip()
                                confidence = subline[1][1]
                                if 'ind' not in text.lower():
                                    lines.append((text, confidence))

            gc.collect()

            if lines:
                combined_text = ''.join(line[0] for line in lines)
                combined_confidence = sum(line[1] for line in lines) / len(lines)
                return {'text': combined_text, 'confidence': combined_confidence}
            return {'text': '', 'confidence': 0.0}
        except Exception as e:
            gc.collect()
            raise RuntimeError(f"PaddleOCR error: {str(e)}")