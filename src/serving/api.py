"""FastAPI serving endpoint for drone detection."""

import io
import logging
import os
import time
from pathlib import Path

import mlflow
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Drone Detection API", version="0.1.0")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "drone-detection-yolo")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", None)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of predictions")
PREDICTIONS_ERRORS = Counter("predictions_errors_total", "Total prediction errors")
DETECTIONS_TOTAL = Counter("detections_total", "Total drones detected")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

_model: YOLO | None = None


def get_model() -> YOLO:
    """Load model from MLflow registry or local path."""
    global _model
    if _model is not None:
        return _model

    if LOCAL_MODEL_PATH and Path(LOCAL_MODEL_PATH).exists():
        logger.info("Loading model from local path: %s", LOCAL_MODEL_PATH)
        _model = YOLO(LOCAL_MODEL_PATH)
    else:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=["Production", "None"])
            if versions:
                run_id = versions[0].run_id
                logger.info("Loading model from MLflow registry, run_id=%s", run_id)
                artifact_path = client.download_artifacts(run_id, "model")
                pt_files = list(Path(artifact_path).glob("*.pt"))
                _model = YOLO(str(pt_files[0])) if pt_files else YOLO("yolov8n.pt")
            else:
                logger.warning("No model in registry, using pretrained yolov8n")
                _model = YOLO("yolov8n.pt")
        except Exception:
            logger.exception("Failed to load model from MLflow, falling back to yolov8n")
            _model = YOLO("yolov8n.pt")

    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: float = 0.25):
    """Run drone detection on an uploaded image."""
    PREDICTIONS_TOTAL.inc()
    start = time.time()

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        model = get_model()
        results = model.predict(source=image, conf=confidence)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": int(box.cls[0]),
                    "class_name": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                })

        DETECTIONS_TOTAL.inc(len(detections))
        latency = time.time() - start
        PREDICTION_LATENCY.observe(latency)
        logger.info("Prediction: %d detections, latency=%.3fs", len(detections), latency)

        return {"detections": detections, "count": len(detections)}
    except Exception:
        PREDICTIONS_ERRORS.inc()
        logger.exception("Prediction failed")
        raise
