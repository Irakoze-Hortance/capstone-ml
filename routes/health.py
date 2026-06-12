import time

from datetime import datetime, timezone
from fastapi import APIRouter

from models.health import HealthResponse

from core.history import read_history
from core.model_loader import (
    MODEL_BACKEND,
    MODEL_VERSION,
)
from core.config import (
    CLASS_NAMES,
    CONF_THRESHOLD,
)

router = APIRouter(
    prefix="/health",
    tags=["Utility"],
)

SERVER_START = time.time()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Server health check",
)
def health():

    records = read_history()

    return HealthResponse(
        status="ok",
        model_backend=MODEL_BACKEND,
        model_version=MODEL_VERSION,
        class_names=CLASS_NAMES,
        conf_threshold=CONF_THRESHOLD,
        history_records=len(records),
        uptime_s=round(time.time() - SERVER_START, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )