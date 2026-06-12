from pydantic import BaseModel
from typing import List

class HealthResponse(BaseModel):
    status: str
    model_backend: str
    model_version: str
    class_names: List[str]
    conf_threshold: float
    history_records: int
    uptime_s: float
    timestamp: str