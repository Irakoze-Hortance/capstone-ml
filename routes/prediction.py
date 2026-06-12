import uuid

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import (
    APIRouter,
    File,
    Query,
    UploadFile,
    HTTPException,
)

from models.prediction import (
    PredictionResult,
    BatchPredictionResult,
)

from core.inference import infer
from core.history import (
    append_history,
    build_observation,
    obs_to_result,
)

from utils.validators import (
    validate_upload,
    check_size,
)

from core.preprocessing import decode_image
from core.config import MAX_BATCH_IMAGES

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post(
    "",
    response_model=PredictionResult,
)
async def predict(
    file: UploadFile = File(...),
    inspector_id: Optional[str] = Query(None),
):

    validate_upload(file)

    raw = await file.read()

    check_size(raw)

    image = decode_image(raw)

    verdict, confidence, probs, inf_ms = infer(image)

    obs = build_observation(
        query_filename=file.filename
        or f"upload_{uuid.uuid4().hex[:8]}.jpg",
        verdict=verdict,
        confidence=confidence,
        probabilities=probs,
        inference_ms=inf_ms,
        inspector_id=inspector_id,
    )

    append_history(obs)

    return obs_to_result(obs)


@router.post(
    "/batch",
    response_model=BatchPredictionResult,
)
async def predict_batch(
    files: List[UploadFile] = File(...),
    inspector_id: Optional[str] = Query(None),
):

    if len(files) > MAX_BATCH_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Maximum {MAX_BATCH_IMAGES} images allowed.",
        )

    batch_id = uuid.uuid4().hex

    results = []
    failed = []

    for upload in files:

        try:
            validate_upload(upload)

            raw = await upload.read()

            check_size(raw)

            image = decode_image(raw)

            verdict, confidence, probs, inf_ms = infer(image)

            obs = build_observation(
                query_filename=upload.filename
                or f"batch_{uuid.uuid4().hex[:8]}.jpg",
                verdict=verdict,
                confidence=confidence,
                probabilities=probs,
                inference_ms=inf_ms,
                inspector_id=inspector_id,
            )

            append_history(obs)

            results.append(obs_to_result(obs))

        except Exception as exc:

            failed.append(
                {
                    "filename": upload.filename or "unknown",
                    "error": str(exc),
                }
            )

    return BatchPredictionResult(
        batch_id=batch_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total=len(files),
        results=results,
        failed=failed,
    )