from pydantic import BaseModel
from typing import Optional

from .prediction import ClassProbability

class ObservationSummary(BaseModel):
    observation_id: str
    timestamp: str
    query_filename: str
    verdict: str
    confidence: float
    inspector_id: Optional[str] = None


class ObservationDetail(ObservationSummary):
    probabilities: ClassProbability
    inference_ms: float
    model_version: str