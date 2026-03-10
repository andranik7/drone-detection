"""Evaluate a YOLO model and compare with the current production model."""

import logging
import os
import mlflow
from ultralytics import YOLO

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "drone-detection-yolo")


def get_production_metrics() -> dict | None:
    """Fetch metrics of the current production model from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        latest = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not latest:
            logger.info("No production model found in registry")
            return None
        run = client.get_run(latest[0].run_id)
        return run.data.metrics
    except Exception:
        logger.exception("Failed to fetch production metrics")
        return None


def evaluate_model(model_path: str, data_yaml: str) -> dict:
    """Run validation on a YOLO model and return metrics."""
    logger.info("Evaluating model: %s", model_path)
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    metrics = {
        "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def is_better(new_metrics: dict, old_metrics: dict | None, key: str = "mAP50") -> bool:
    """Check if new model outperforms the old one on a given metric."""
    if old_metrics is None:
        logger.info("No production model found, new model wins by default")
        return True
    new_val = new_metrics.get(key, 0)
    old_val = old_metrics.get(key, 0)
    logger.info("Comparing %s: new=%.4f vs old=%.4f", key, new_val, old_val)
    return new_val > old_val
