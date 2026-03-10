"""DAG: Train YOLOv8 model and register in MLflow."""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow")


def train_task():
    from src.training.train import train
    metrics = train(epochs=20, imgsz=640, batch=16)
    print(f"Training metrics: {metrics}")
    return metrics


with DAG(
    dag_id="train_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["training"],
) as dag:

    train = PythonOperator(task_id="train_model", python_callable=train_task)
