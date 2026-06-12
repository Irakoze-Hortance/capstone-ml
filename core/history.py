import json
import uuid

from datetime import (
    datetime,
    timezone,
)

from core.config import (
    ARTIFACT_DIR,
    HISTORY_PATH,
)

from models.prediction import (
    PredictionResult,
    ClassProbability,
)

from core.model_loader import (
    MODEL_VERSION,
)


def ensure_artifacts():
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def read_history():

    ensure_artifacts()

    if not HISTORY_PATH.exists():
        return []

    records = []

    with HISTORY_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return records


def append_history(entry):

    ensure_artifacts()

    with HISTORY_PATH.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(entry)
            + "\n"
        )


def rewrite_history(records):

    ensure_artifacts()

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in records:

            f.write(
                json.dumps(row)
                + "\n"
            )


def build_observation(
    query_filename,
    verdict,
    confidence,
    probabilities,
    inference_ms,
    inspector_id=None,
):

    now = datetime.now(
        timezone.utc
    )

    return {
        "observation_id":
            now.strftime("%Y%m%d%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8],

        "timestamp":
            now.isoformat(),

        "query_filename":
            query_filename,

        "verdict":
            verdict,

        "confidence":
            round(confidence, 6),

        "prob_authentic":
            round(
                probabilities.authentic,
                6,
            ),

        "prob_counterfeit":
            round(
                probabilities.counterfeit,
                6,
            ),

        "inference_ms":
            round(
                inference_ms,
                3,
            ),

        "model_version":
            MODEL_VERSION,

        "inspector_id":
            inspector_id,
    }


def obs_to_result(obs):

    return PredictionResult(
        observation_id=obs[
            "observation_id"
        ],
        timestamp=obs[
            "timestamp"
        ],
        query_filename=obs[
            "query_filename"
        ],
        verdict=obs[
            "verdict"
        ],
        confidence=obs[
            "confidence"
        ],
        probabilities=ClassProbability(
            authentic=obs.get(
                "prob_authentic",
                0.0,
            ),
            counterfeit=obs.get(
                "prob_counterfeit",
                0.0,
            ),
        ),
        inference_ms=obs.get(
            "inference_ms",
            0.0,
        ),
        model_version=obs.get(
            "model_version",
            MODEL_VERSION,
        ),
    )