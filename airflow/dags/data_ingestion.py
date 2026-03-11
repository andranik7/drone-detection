"""DAG: Download and validate drone detection dataset."""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow")


def download_task():
    from src.data.download import download_dataset, fix_data_yaml, verify_structure
    download_dataset()
    fix_data_yaml()
    stats = verify_structure()
    print(f"Dataset stats: {stats}")
    return stats


def preprocess_task():
    from pathlib import Path
    from src.data.preprocess import clean_split

    data_dir = Path("/opt/airflow/data")
    for split in ["train", "valid", "test"]:
        result = clean_split(data_dir / split)
        print(f"{split}: {result}")


with DAG(
    dag_id="data_ingestion",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["data"],
) as dag:

    download = PythonOperator(task_id="download_dataset", python_callable=download_task)
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_task)

    download >> preprocess
