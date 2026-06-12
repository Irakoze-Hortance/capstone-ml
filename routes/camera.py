import uuid

from fastapi import APIRouter
from io import BytesIO
import base64

from models.camera import (
    CameraCaptureRequest,
    PreprocessResponse,
)

from models.prediction import PredictionResult

from core.preprocessing import (
    decode_base64,
    decode_image,
    enhance_for_preview,
)

from core.inference import infer

from core.history import (
    append_history,
    build_observation,
    obs_to_result,
)

router = APIRouter(
    prefix="/camera",
    tags=["Camera"],
)


@router.post(
    "/predict",
    response_model=PredictionResult,
)
async def camera_predict(
    payload: CameraCaptureRequest,
):

    raw = decode_base64(
        payload.image_base64
    )

    image = decode_image(raw)

    verdict, confidence, probs, inf_ms = infer(image)

    obs = build_observation(
        query_filename=payload.filename
        or f"camera_{uuid.uuid4().hex[:8]}.jpg",
        verdict=verdict,
        confidence=confidence,
        probabilities=probs,
        inference_ms=inf_ms,
        inspector_id=payload.inspector_id,
    )

    append_history(obs)

    return obs_to_result(obs)


@router.post(
    "/preprocess",
    response_model=PreprocessResponse,
)
async def camera_preprocess(
    payload: CameraCaptureRequest,
):

    raw = decode_base64(
        payload.image_base64
    )

    image = decode_image(raw)

    enhanced = enhance_for_preview(image)

    buffer = BytesIO()

    enhanced.save(
        buffer,
        format="PNG",
    )

    preview = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return PreprocessResponse(
        filename=payload.filename
        or "camera_capture.png",
        original_size=[
            image.width,
            image.height,
        ],
        output_size=[
            enhanced.width,
            enhanced.height,
        ],
        brightness_factor=1.15,
        contrast_factor=1.10,
        preview_base64=preview,
    )