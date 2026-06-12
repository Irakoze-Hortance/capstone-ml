import base64
import numpy as np

from io import BytesIO
from PIL import (
    Image,
    ImageOps,
    ImageEnhance,
    ImageFilter,
)

from fastapi import HTTPException

from core.config import IMG_SIZE


def decode_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(
            BytesIO(raw)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode image: {exc}",
        )


def preprocess_for_cnn(
    image: Image.Image,
) -> np.ndarray:

    resized = image.resize(
        IMG_SIZE,
        Image.LANCZOS,
    )

    arr = (
        np.asarray(
            resized,
            dtype=np.float32,
        )
        / 255.0
    )

    return np.expand_dims(arr, axis=0)


def preprocess_for_uint8(
    image: Image.Image,
) -> np.ndarray:

    resized = image.resize(
        IMG_SIZE,
        Image.LANCZOS,
    )

    arr = np.asarray(
        resized,
        dtype=np.uint8,
    )

    return np.expand_dims(arr, axis=0)


def enhance_for_preview(
    image: Image.Image,
    brightness: float = 1.15,
    contrast: float = 1.10,
):

    image = ImageOps.autocontrast(image)

    image = (
        ImageEnhance.Brightness(image)
        .enhance(brightness)
    )

    image = (
        ImageEnhance.Contrast(image)
        .enhance(contrast)
    )

    image = image.filter(
        ImageFilter.MedianFilter(size=3)
    )

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


def decode_base64(
    payload: str,
) -> bytes:

    try:
        return base64.b64decode(
            payload,
            validate=True,
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image payload.",
        )