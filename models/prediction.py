from pydantic import BaseModel, Field
from typing import List, Dict

class ClassProbability(BaseModel):
    authentic: float = Field(..., ge=0, le=1)
    counterfeit: float = Field(..., ge=0, le=1)


class PredictionResult(BaseModel):
    observation_id: str
    timestamp: str
    query_filename: str
    verdict: str
    confidence: float
    probabilities: ClassProbability
    inference_ms: float
    model_version: str
    scope_note: str = (
        "Packaging-level verification only. Not a chemical analysis."
    )


class BatchPredictionResult(BaseModel):
    batch_id: str
    timestamp: str
    total: int
    results: List[PredictionResult]
    failed: List[Dict[str, str]] = []