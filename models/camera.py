from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

class CameraCaptureRequest(BaseModel):
    image_base64: str
    filename: Optional[str] = None
    inspector_id: Optional[str] = None

    @field_validator("image_base64")
    @classmethod
    def strip_data_uri(cls, value: str):
        if "," in value:
            return value.split(",", 1)[1]
        return value.strip()


class PreprocessResponse(BaseModel):
    filename: str
    original_size: List[int]
    output_size: List[int]
    brightness_factor: float
    contrast_factor: float
    preview_base64: str