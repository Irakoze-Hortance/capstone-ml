from datetime import (
    datetime,
    timezone,
)

import numpy as np

from fastapi import (
    APIRouter,
)

from fastapi.responses import Response

from models.stats import StatsResponse

from core.history import read_history

from core.visualization import (
    build_heatmap_png,
    build_confidence_dist_png,
)

router = APIRouter(
    tags=["Analytics"],
)


@router.get(
    "/stats",
    response_model=StatsResponse,
)
def stats():

    records = read_history()

    if not records:

        return StatsResponse(
            total_scans=0,
            authentic_count=0,
            counterfeit_count=0,
            counterfeit_rate_pct=0,
            mean_confidence=0,
            mean_inference_ms=0,
            scans_today=0,
            unique_inspectors=0,
            first_scan=None,
            last_scan=None,
        )

    auth_count = sum(
        1
        for r in records
        if r["verdict"] == "authentic"
    )

    counterfeit_count = sum(
        1
        for r in records
        if r["verdict"] == "counterfeit"
    )

    today = datetime.now(
        timezone.utc
    ).date().isoformat()

    scans_today = sum(
        1
        for r in records
        if r["timestamp"].startswith(today)
    )

    inspectors = {
        r["inspector_id"]
        for r in records
        if r.get("inspector_id")
    }

    timestamps = sorted(
        r["timestamp"]
        for r in records
    )

    return StatsResponse(
        total_scans=len(records),
        authentic_count=auth_count,
        counterfeit_count=counterfeit_count,
        counterfeit_rate_pct=round(
            counterfeit_count
            / len(records)
            * 100,
            2,
        ),
        mean_confidence=round(
            np.mean(
                [
                    r["confidence"]
                    for r in records
                ]
            ),
            4,
        ),
        mean_inference_ms=round(
            np.mean(
                [
                    r["inference_ms"]
                    for r in records
                ]
            ),
            2,
        ),
        scans_today=scans_today,
        unique_inspectors=len(inspectors),
        first_scan=timestamps[0],
        last_scan=timestamps[-1],
    )


@router.get("/heatmap")
def heatmap():

    png = build_heatmap_png(
        read_history()
    )

    return Response(
        content=png,
        media_type="image/png",
    )


@router.get("/heatmap/confidence")
def confidence_heatmap():

    png = build_confidence_dist_png(
        read_history()
    )

    return Response(
        content=png,
        media_type="image/png",
    )