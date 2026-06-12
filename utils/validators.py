from fastapi import HTTPException, UploadFile
from core.config import (
    ALLOWED_MIME_TYPES,
    MAX_IMAGE_BYTES
)

def validate_upload(upload: UploadFile):
    ct = upload.content_type or ""

    if ct not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{ct}'"
        )


def check_size(raw: bytes):
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds maximum size."
        )