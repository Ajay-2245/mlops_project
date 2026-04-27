"""
airflow/dags/data_pipeline_dag.py
───────────────────────────────────
Airflow 3.x compatible DAG for the Insurance Fraud Detection ML pipeline.

Compatible with: Apache Airflow 3.0.4+, Python 3.13
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Works locally (PROJECT_ROOT env var) and in Docker (/opt/airflow)
ROOT = Path(os.getenv("PROJECT_ROOT", os.getcwd()))

DEFAULT_ARGS = {
    "owner": "da5402-ajay",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run_python(script: str) -> None:
    """Run a Python script from the project root."""
    script_path = ROOT / script
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    if result.stdout:
        logging.info(result.stdout)
    if result.returncode != 0:
        logging.error(result.stderr)
        raise RuntimeError(
            f"Script {script} failed with exit code {result.returncode}\n{result.stderr}"
        )


def ingest_task(**kwargs):
    _run_python("src/data/ingest.py")


def validate_task(**kwargs):
    _run_python("src/data/validate.py")
    report_path = ROOT / "data/processed/validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    kwargs["ti"].xcom_push(key="validation_passed", value=report["passed"])
    kwargs["ti"].xcom_push(key="row_count", value=report.get("row_count", 0))
    logging.info(
        "Validation: passed=%s rows=%s",
        report["passed"], report.get("row_count", 0),
    )


def decide_on_validation(**kwargs):
    ti = kwargs["ti"]
    passed = ti.xcom_pull(task_ids="validate", key="validation_passed")
    logging.info("Validation passed: %s", passed)
    return "preprocess" if passed else "validation_failed"


def preprocess_task(**kwargs):
    _run_python("src/data/preprocess.py")


def train_task(**kwargs):
    _run_python("src/models/train.py")


def evaluate_task(**kwargs):
    _run_python("src/models/evaluate.py")
    metrics_path = ROOT / "data/processed/eval_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    kwargs["ti"].xcom_push(key="test_f1", value=metrics.get("test_f1", 0))
    kwargs["ti"].xcom_push(key="test_roc_auc", value=metrics.get("test_roc_auc", 0))
    logging.info(
        "Evaluation: F1=%.4f ROC-AUC=%.4f",
        metrics.get("test_f1", 0), metrics.get("test_roc_auc", 0),
    )


def register_model_task(**kwargs):
    """
    Alias the latest model version as 'champion'.
    Uses MLflow 3.x API (set_registered_model_alias).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    ti = kwargs["ti"]
    test_f1 = ti.xcom_pull(task_ids="evaluate", key="test_f1")
    test_roc_auc = ti.xcom_pull(task_ids="evaluate", key="test_roc_auc")

    logging.info("Test F1=%.4f | ROC-AUC=%.4f", test_f1, test_roc_auc)

    if test_f1 < 0.60 or test_roc_auc < 0.70:
        raise ValueError(
            f"Model does not meet acceptance criteria. "
            f"F1={test_f1:.4f} (need ≥0.60), AUC={test_roc_auc:.4f} (need ≥0.70)"
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = "insurance_fraud_model"

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logging.warning("No model versions found for %s", model_name)
        return

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest.version,
    )
    logging.info(
        "Model '%s' version %s aliased as 'champion'.",
        model_name, latest.version,
    )


def validation_failed_handler(**kwargs):
    logging.error("Data validation failed. Aborting pipeline.")
    raise ValueError("Data validation failed — pipeline aborted.")


def pipeline_success_notification(**kwargs):
    ti = kwargs["ti"]
    f1 = ti.xcom_pull(task_ids="evaluate", key="test_f1")
    auc = ti.xcom_pull(task_ids="evaluate", key="test_roc_auc")
    rows = ti.xcom_pull(task_ids="validate", key="row_count")
    logging.info(
        "Pipeline SUCCESS | rows=%s | F1=%.4f | AUC=%.4f",
        rows, f1 or 0, auc or 0,
    )


# ── DAG definition ─────────────────────────────────────────────────────────
with DAG(
    dag_id="insurance_fraud_ml_pipeline",
    default_args=DEFAULT_ARGS,
    description="End-to-end ML pipeline for Insurance Fraud Detection",
    schedule="0 2 * * *",          # Airflow 3.x uses schedule= not schedule_interval=
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "fraud", "da5402"],
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest",
        python_callable=ingest_task,
    )

    t_validate = PythonOperator(
        task_id="validate",
        python_callable=validate_task,
    )

    t_branch = BranchPythonOperator(
        task_id="branch_on_validation",
        python_callable=decide_on_validation,
    )

    t_validation_failed = PythonOperator(
        task_id="validation_failed",
        python_callable=validation_failed_handler,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_task,
    )

    t_train = PythonOperator(
        task_id="train",
        python_callable=train_task,
    )

    t_evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate_task,
    )

    t_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model_task,
    )

    t_notify = PythonOperator(
        task_id="notify_success",
        python_callable=pipeline_success_notification,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ── DAG flow ──────────────────────────────────────────────────────────
    t_ingest >> t_validate >> t_branch
    t_branch >> t_validation_failed
    t_branch >> t_preprocess >> t_train >> t_evaluate >> t_register >> t_notify