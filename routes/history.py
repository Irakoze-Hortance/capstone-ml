from typing import List, Optional

from fastapi import (
    APIRouter,
    Query,
    HTTPException,
)

from models.history import (
    ObservationSummary,
    ObservationDetail,
)

from models.prediction import ClassProbability

from core.history import (
    read_history,
    rewrite_history,
)

from core.model_loader import MODEL_VERSION

router = APIRouter(
    prefix="/history",
    tags=["History"],
)


@router.get(
    "",
    response_model=List[ObservationSummary],
)
def history(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    verdict: Optional[str] = Query(None),
):

    records = read_history()[::-1]

    if verdict:
        records = [
            r
            for r in records
            if r["verdict"] == verdict
        ]

    records = records[
        skip : skip + limit
    ]

    return [
        ObservationSummary(
            observation_id=r["observation_id"],
            timestamp=r["timestamp"],
            query_filename=r["query_filename"],
            verdict=r["verdict"],
            confidence=r["confidence"],
            inspector_id=r.get("inspector_id"),
        )
        for r in records
    ]


@router.get(
    "/{observation_id}",
    response_model=ObservationDetail,
)
def history_detail(
    observation_id: str,
):

    for r in read_history():

        if r["observation_id"] == observation_id:

            return ObservationDetail(
                observation_id=r["observation_id"],
                timestamp=r["timestamp"],
                query_filename=r["query_filename"],
                verdict=r["verdict"],
                confidence=r["confidence"],
                inspector_id=r.get("inspector_id"),
                probabilities=ClassProbability(
                    authentic=r.get(
                        "prob_authentic",
                        0,
                    ),
                    counterfeit=r.get(
                        "prob_counterfeit",
                        0,
                    ),
                ),
                inference_ms=r["inference_ms"],
                model_version=r.get(
                    "model_version",
                    MODEL_VERSION,
                ),
            )

    raise HTTPException(
        status_code=404,
        detail="Observation not found.",
    )


@router.delete(
    "/{observation_id}",
)
def delete_history(
    observation_id: str,
):

    records = read_history()

    before = len(records)

    records = [
        r
        for r in records
        if r["observation_id"] != observation_id
    ]

    if len(records) == before:
        raise HTTPException(
            status_code=404,
            detail="Observation not found.",
        )

    rewrite_history(records)

    return {
        "deleted": observation_id,
        "remaining": len(records),
    }