import csv
import io
import json

from fastapi import (
    APIRouter,
    HTTPException,
)

from fastapi.responses import (
    StreamingResponse,
)

from core.history import read_history

router = APIRouter(
    prefix="/export",
    tags=["Export"],
)


@router.get("/csv")
def export_csv():

    records = read_history()

    if not records:
        raise HTTPException(
            status_code=404,
            detail="No history to export.",
        )

    fields = [
        "observation_id",
        "timestamp",
        "query_filename",
        "verdict",
        "confidence",
        "prob_authentic",
        "prob_counterfeit",
        "inference_ms",
        "model_version",
        "inspector_id",
    ]

    def generate():

        buffer = io.StringIO()

        writer = csv.DictWriter(
            buffer,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in records:
            writer.writerow(row)

        yield buffer.getvalue()

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=pharmacheck_history.csv"
        },
    )


@router.get("/json")
def export_json():

    records = read_history()

    if not records:
        raise HTTPException(
            status_code=404,
            detail="No history to export.",
        )

    return StreamingResponse(
        iter(
            [
                json.dumps(
                    records,
                    indent=2,
                    ensure_ascii=False,
                )
            ]
        ),
        media_type="application/json",
        headers={
            "Content-Disposition":
            "attachment; filename=pharmacheck_history.json"
        },
    )