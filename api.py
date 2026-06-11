from __future__ import annotations

import base64
from csv import DictReader
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/Users/mac/Desktop/capstone-prep/Counterfeit_med_detection.v1i.multiclass")
HISTORY_PATH = Path("/Users/mac/Desktop/capstone-prep/artifacts/prediction_history.jsonl")
FEATURE_SIZE = (64, 64)
SPLITS = ["train", "valid", "test"]

app = FastAPI(
    title="Counterfeit Medicine Similarity API",
    description="Upload an image and compare it against all dataset images to find the closest match.",
    version="2.0.0",
)


@dataclass(frozen=True)
class DatasetItem:
    split: str
    filename: str
    labels: Dict[str, int]
    feature: np.ndarray


class MatchResult(BaseModel):
    observation_id: str
    timestamp: str
    query_filename: str
    best_match_filename: str
    best_match_split: str
    best_match_labels: Dict[str, int]
    similarity_score: float
    top_5_matches: List[Dict[str, object]]


class ObservationResponse(BaseModel):
    observation_id: str
    timestamp: str
    query_filename: str
    best_match_filename: str
    best_match_split: str
    similarity_score: float


class CameraCaptureRequest(BaseModel):
    image_base64: str
    filename: str | None = None


class PreprocessResponse(BaseModel):
    filename: str
    brightness_factor: float
    output_size: List[int]
    preview_base64: str


def ensure_artifacts_dir() -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_history() -> List[dict]:
    ensure_artifacts_dir()
    if not HISTORY_PATH.exists():
        return []

    observations: List[dict] = []
    with HISTORY_PATH.open("r", encoding="utf-8") as history_file:
        for line in history_file:
            line = line.strip()
            if line:
                observations.append(json.loads(line))
    return observations


def append_history(entry: dict) -> None:
    ensure_artifacts_dir()
    with HISTORY_PATH.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(entry) + "\n")


def load_labels(split: str) -> Dict[str, Dict[str, int]]:
    label_path = DATA_ROOT / split / "_classes.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    labels_by_filename: Dict[str, Dict[str, int]] = {}
    with label_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = DictReader(csv_file)
        for row in reader:
            filename = row.pop("filename")
            labels_by_filename[filename] = {name: int(value) for name, value in row.items()}
    return labels_by_filename


def _decode_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


def preprocess_for_similarity(raw: bytes, brightness_factor: float = 1.15) -> np.ndarray:
    image = _decode_image(raw)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = image.filter(ImageFilter.SHARPEN)
    image = image.resize(FEATURE_SIZE)
    feature = np.asarray(image).astype("float32").reshape(-1) / 255.0
    norm = float(np.linalg.norm(feature))
    if norm == 0.0:
        raise HTTPException(status_code=400, detail="Image could not be processed.")
    return feature / norm


def preprocess_for_ocr_preview(raw: bytes, brightness_factor: float = 1.15) -> tuple[Image.Image, float]:
    image = _decode_image(raw)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = image.filter(ImageFilter.SHARPEN)
    return image, brightness_factor


def build_dataset_index() -> List[DatasetItem]:
    dataset_index: List[DatasetItem] = []
    for split in SPLITS:
        split_dir = DATA_ROOT / split
        labels_by_filename = load_labels(split)
        for image_path in sorted(split_dir.iterdir()):
            if not image_path.is_file() or image_path.name == "_classes.csv":
                continue
            feature = preprocess_for_similarity(image_path.read_bytes())
            dataset_index.append(
                DatasetItem(
                    split=split,
                    filename=image_path.name,
                    labels=labels_by_filename.get(image_path.name, {}),
                    feature=feature,
                )
            )

    if not dataset_index:
        raise RuntimeError("No dataset images were indexed.")
    return dataset_index


DATASET_INDEX = build_dataset_index()


def decode_base64_image(image_base64: str) -> bytes:
    payload = image_base64.strip()
    if "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 camera image.") from exc


def build_similarity_heatmap(observations: List[dict]) -> bytes:
    if not observations:
        raise HTTPException(status_code=404, detail="No upload history found yet.")

    rows = []
    for split in SPLITS:
        split_rows = [row for row in observations if row.get("best_match_split") == split]
        if split_rows:
            similarities = [float(row["similarity_score"]) for row in split_rows]
            rows.append([split, len(split_rows), float(np.mean(similarities)), float(np.max(similarities))])

    if not rows:
        raise HTTPException(status_code=404, detail="No matching observations found.")

    values = np.array([[row[2]] for row in rows], dtype=float)
    counts = [row[1] for row in rows]
    best_scores = [row[3] for row in rows]
    labels = [row[0] for row in rows]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels))))
    heatmap = ax.imshow(values, cmap="Reds", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Similarity Heatmap by Dataset Split")
    ax.set_xlabel("Mean similarity")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0])
    ax.set_xticklabels(["similarity"])

    for row_index, _label in enumerate(labels):
        ax.text(
            0,
            row_index,
            f"n={counts[row_index]}\nbest={best_scores[row_index]:.3f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75),
        )

    fig.colorbar(heatmap, ax=ax, label="Mean similarity")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_images": len(DATASET_INDEX)}


@app.post("/predict", response_model=MatchResult)
async def predict(file: UploadFile = File(...)) -> MatchResult:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    raw = await file.read()
    query_feature = preprocess_for_similarity(raw)

    similarities = np.array([float(np.dot(query_feature, item.feature)) for item in DATASET_INDEX], dtype=float)
    ranked_indices = np.argsort(similarities)[::-1]
    top_index = int(ranked_indices[0])
    top_item = DATASET_INDEX[top_index]
    top_score = float(similarities[top_index])

    now = datetime.now(timezone.utc)
    observation_id = now.strftime("%Y%m%d%H%M%S%f")
    timestamp = now.isoformat()
    query_filename = file.filename or f"upload_{observation_id}.jpg"

    top_5_matches = []
    for rank in ranked_indices[:5]:
        item = DATASET_INDEX[int(rank)]
        top_5_matches.append(
            {
                "filename": item.filename,
                "split": item.split,
                "labels": item.labels,
                "similarity_score": float(similarities[int(rank)]),
            }
        )

    observation = {
        "observation_id": observation_id,
        "timestamp": timestamp,
        "query_filename": query_filename,
        "best_match_filename": top_item.filename,
        "best_match_split": top_item.split,
        "similarity_score": top_score,
    }
    append_history(observation)

    return MatchResult(
        observation_id=observation_id,
        timestamp=timestamp,
        query_filename=query_filename,
        best_match_filename=top_item.filename,
        best_match_split=top_item.split,
        best_match_labels=top_item.labels,
        similarity_score=top_score,
        top_5_matches=top_5_matches,
    )


@app.post("/camera/predict", response_model=MatchResult)
async def camera_predict(payload: CameraCaptureRequest) -> MatchResult:
    raw = decode_base64_image(payload.image_base64)
    query_feature = preprocess_for_similarity(raw)

    similarities = np.array([float(np.dot(query_feature, item.feature)) for item in DATASET_INDEX], dtype=float)
    ranked_indices = np.argsort(similarities)[::-1]
    top_index = int(ranked_indices[0])
    top_item = DATASET_INDEX[top_index]
    top_score = float(similarities[top_index])

    now = datetime.now(timezone.utc)
    observation_id = now.strftime("%Y%m%d%H%M%S%f")
    timestamp = now.isoformat()
    query_filename = payload.filename or f"camera_{observation_id}.jpg"

    top_5_matches = []
    for rank in ranked_indices[:5]:
        item = DATASET_INDEX[int(rank)]
        top_5_matches.append(
            {
                "filename": item.filename,
                "split": item.split,
                "labels": item.labels,
                "similarity_score": float(similarities[int(rank)]),
            }
        )

    observation = {
        "observation_id": observation_id,
        "timestamp": timestamp,
        "query_filename": query_filename,
        "best_match_filename": top_item.filename,
        "best_match_split": top_item.split,
        "similarity_score": top_score,
    }
    append_history(observation)

    return MatchResult(
        observation_id=observation_id,
        timestamp=timestamp,
        query_filename=query_filename,
        best_match_filename=top_item.filename,
        best_match_split=top_item.split,
        best_match_labels=top_item.labels,
        similarity_score=top_score,
        top_5_matches=top_5_matches,
    )


@app.post("/camera/preprocess", response_model=PreprocessResponse)
async def camera_preprocess(payload: CameraCaptureRequest) -> PreprocessResponse:
    raw = decode_base64_image(payload.image_base64)
    image, brightness_factor = preprocess_for_ocr_preview(raw)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    preview_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return PreprocessResponse(
        filename=payload.filename or "camera_capture.png",
        brightness_factor=brightness_factor,
        output_size=[image.width, image.height],
        preview_base64=preview_base64,
    )


@app.get("/history", response_model=List[ObservationResponse])
def history() -> List[ObservationResponse]:
    observations = read_history()
    return [ObservationResponse(**row) for row in observations]


@app.get("/heatmap")
def heatmap() -> Response:
    image_bytes = build_similarity_heatmap(read_history())
    return Response(content=image_bytes, media_type="image/png")


# Run locally:
# /usr/local/bin/python3 -m uvicorn api:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
