from pydantic import BaseModel
from typing import List, Optional

class DetectionResult(BaseModel):
    detection_id: int
    bbox: List[float]
    detection_confidence: float
    text: str
    confidence: float

class OCRResponse(BaseModel):
    success: bool
    detections_count: int
    message: Optional[str] = None
    results: List[DetectionResult] = []