"""DAG: Continuous training - retrain and promote model if better."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

import sys
sys.path.insert(0, "/opt/airflow")


def retrain_task(**context):
    from src.training.train import train
    metrics = train(epochs=20, imgsz=640, batch=16)
    context["ti"].xcom_push(key="new_metrics", value=metrics)
    return metrics


def compare_task(**context):
    from src.training.evaluate import get_production_metrics, is_better
    new_metrics = context["ti"].xcom_pull(key="new_metrics", task_ids="retrain")
    old_metrics = get_production_metrics()
    if is_better(new_metrics, old_metrics):
        return "promote_model"
    return "skip_promotion"


def promote_task():
    import os
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()
    model_name = os.getenv("MODEL_NAME", "drone-detection-yolo")
    versions = client.get_latest_versions(model_name, stages=["None"])
    if versions:
        client.transition_model_version_stage(model_name, versions[0].version, "Production")
        print(f"Promoted model v{versions[0].version} to Production")


def skip_task():
    print("New model is not better. Skipping promotion.")


with DAG(
    dag_id="continuous_training",
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(weeks=1),
    catchup=False,
    tags=["ct"],
) as dag:

    retrain = PythonOperator(task_id="retrain", python_callable=retrain_task)

    compare = BranchPythonOperator(task_id="compare_models", python_callable=compare_task)

    promote = PythonOperator(task_id="promote_model", python_callable=promote_task)

    skip = PythonOperator(task_id="skip_promotion", python_callable=skip_task)

    retrain >> compare >> [promote, skip]
