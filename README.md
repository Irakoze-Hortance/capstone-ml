# PharmaCheck API

A FastAPI-based machine learning system for counterfeit medicine packaging detection using a lightweight CNN (MobileNetV3) exported to TensorFlow Lite (INT8) for fast inference.

> Scope: This system performs packaging-level verification only. It does not analyze chemical composition.

---

## Features

- CNN-based classification (Authentic vs Counterfeit)
- Fast inference using TensorFlow Lite (optimized for edge and server)
- Camera-ready base64 image API
- Batch image prediction (up to 20 images)
- Scan history tracking (JSONL storage)
- Analytics (statistics, confidence distribution, heatmaps)
- Export results as CSV and JSON
- Fully Dockerized deployment
- REST API ready for Android integration

---

## Tech Stack

- FastAPI
- TensorFlow Lite (TFLite INT8 model)
- NumPy
- Pillow
- Matplotlib
- Docker

---

## Design

Figma Design:

https://www.figma.com/design/nlKueCQFq6y63viL3jSwd1/pharmacheck?node-id=0-1&t=1csNLmRkR3myup7s-1

![Application Screenshot](screen.png)

---

## Project Structure

```text
pharmacheck/
│
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── predict.py
│   │   ├── history.py
│   │   ├── stats.py
│   │   ├── camera.py
│   │   └── export.py
│   │
│   ├── core/
│   │   ├── model.py
│   │   ├── inference.py
│   │   └── utils.py
│   │
│   └── schemas/
│       ├── prediction.py
│       ├── history.py
│       └── stats.py
│
├── artifacts/
│   └── pharmacheck_mobilenetv3.tflite
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## Setup Instructions

### Prerequisites

Ensure the following are installed:

- Python 3.11+
- Docker
- Git

---

### Clone the Repository

```bash
git clone https://github.com/your-username/pharmacheck.git
cd pharmacheck
```

---

### Create and Activate a Virtual Environment

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Add the Model File

Place the TensorFlow Lite model in the `artifacts` directory:

```text
artifacts/
└── pharmacheck_mobilenetv3.tflite
```

---

### Run the Application

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

## Docker Setup

### Build the Docker Image

```bash
docker build -t pharmacheck-api .
```

### Run the Docker Container

```bash
docker run -p 8000:8000 pharmacheck-api
```

The API will be available at:

```text
http://localhost:8000
```

---

## API Endpoints

### Health

```http
GET /health
```

Returns server health information, model details, and history statistics.

---

### Single Prediction

```http
POST /predict
```

Upload a medicine packaging image for classification.

**Input**

- Multipart form-data
- `file`: image (JPEG, PNG, BMP, WebP)

---

### Batch Prediction

```http
POST /predict/batch
```

Upload up to 20 images in a single request.

---

### Camera Prediction

```http
POST /camera/predict
```

Predict from a base64-encoded image.

Example payload:

```json
{
  "image_base64": "base64_encoded_image",
  "filename": "sample.jpg",
  "inspector_id": "uuid"
}
```

---

### Camera Preprocessing Preview

```http
POST /camera/preprocess
```

Returns the enhanced image preview and preprocessing metadata.

---

### History

```http
GET /history
GET /history/{observation_id}
DELETE /history/{observation_id}
```

Manage scan observations.

---

### Statistics

```http
GET /stats
```

Returns:

- Total scans
- Counterfeit rate
- Average confidence
- Average inference time
- Inspector statistics

---

### Visualizations

```http
GET /heatmap
GET /heatmap/confidence
```

Generate PNG visualizations from scan history.

---

### Export

```http
GET /export/csv
GET /export/json
```

Export complete scan history.

---

## Model Information

| Property | Value |
|-----------|--------|
| Architecture | MobileNetV3-Small |
| Format | TensorFlow Lite (INT8) |
| Input Size | 224 × 224 |
| Classes | authentic, counterfeit |
| Deployment | Android + REST API |

---

## Notes

- The system verifies medicine packaging only.
- Predictions should be treated as decision-support information.
- Chemical analysis is outside the scope of this application.
- For production deployments, persistent storage (database or object storage) is recommended instead of local JSONL files.

---

## License

MIT License