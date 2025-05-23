# app/models/__init__.py
from .request import ImageRequest
from .response import OCRResponse, DetectionResult

__all__ = ["ImageRequest", "OCRResponse", "DetectionResult"]