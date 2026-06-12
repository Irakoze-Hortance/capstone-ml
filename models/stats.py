from pydantic import BaseModel
from typing import Optional

class StatsResponse(BaseModel):
    total_scans: int
    authentic_count: int
    counterfeit_count: int
    counterfeit_rate_pct: float
    mean_confidence: float
    mean_inference_ms: float
    scans_today: int
    unique_inspectors: int
    first_scan: Optional[str]
    last_scan: Optional[str]