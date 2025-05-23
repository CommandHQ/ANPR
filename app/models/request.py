from pydantic import BaseModel, HttpUrl
from typing import Optional

class ImageRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[HttpUrl] = None

    class Config:
        extra = "forbid"