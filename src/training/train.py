"""Train YOLOv8 model with MLflow tracking."""

import logging
import os
from pathlib import Path

import mlflow
from ultralytics import YOLO, settings as yolo_settings

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "drone-detection"
DATA_YAML = str(Path(os.getenv("DATA_YAML", "data/data.yaml")).resolve())
MODEL_NAME = os.getenv("MODEL_NAME", "drone-detection-yolo")


def train(epochs: int = 20, imgsz: int = 640, batch: int = 16) -> dict:
    """Train YOLOv8n and log to MLflow."""
    # Configure S3/MinIO credentials for artifact storage
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"))
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))

    # Disable Ultralytics built-in MLflow callback to avoid conflicts
    yolo_settings.update({"mlflow": False})

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "imgsz": imgsz, "batch": batch, "model": "yolov8n"})

        model = YOLO("yolov8n.pt")
        results = model.train(data=DATA_YAML, epochs=epochs, imgsz=imgsz, batch=batch, project="runs", name="train", exist_ok=True)

        # Log metrics
        metrics = {
            "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
            "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
            "precision": results.results_dict.get("metrics/precision(B)", 0),
            "recall": results.results_dict.get("metrics/recall(B)", 0),
        }
        mlflow.log_metrics(metrics)

        # Save model artifact to MinIO via MLflow
        best_model_path = str(Path(results.save_dir) / "weights" / "best.pt")
        mlflow.log_artifact(best_model_path, "model")

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)

        logger.info("Run ID: %s", mlflow.active_run().info.run_id)
        logger.info("Metrics: %s", metrics)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metrics = train()
    logger.info("Training complete: %s", metrics)
