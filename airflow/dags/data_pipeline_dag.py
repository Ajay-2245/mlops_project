"""
airflow/dags/data_pipeline_dag.py
───────────────────────────────────
Airflow DAG for the full ML pipeline.

FIX: t_validation_failed now uses TriggerRule.ALL_DONE (default) instead of
     NONE_FAILED_MIN_ONE_SUCCESS, which was causing it to fire even when the
     branch chose the preprocess path.
"""

import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

ROOT = Path("/opt/airflow")

DEFAULT_ARGS = {
    "owner": "da5402-ajay",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}


def _run_python(script: str, **kwargs) -> None:
    result = subprocess.run(
        ["python", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    logging.info(result.stdout)
    if result.returncode != 0:
        logging.error(result.stderr)
        raise RuntimeError(f"Script {script} failed with code {result.returncode}")


def ingest_task(**kwargs):
    _run_python("src/data/ingest.py")


def validate_task(**kwargs):
    _run_python("src/data/validate.py")
    report_path = ROOT / "data/processed/validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    kwargs["ti"].xcom_push(key="validation_passed", value=report["passed"])
    kwargs["ti"].xcom_push(key="row_count", value=report.get("row_count", 0))


def decide_on_validation(**kwargs):
    ti = kwargs["ti"]
    passed = ti.xcom_pull(task_ids="validate", key="validation_passed")
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


def register_model_task(**kwargs):
    """Promote model to Production if it meets acceptance criteria."""
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

    client = MlflowClient(tracking_uri="http://mlflow:5000")
    model_name = "insurance_fraud_model"

    versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
    if not versions:
        logging.warning("No model versions found in None/Staging stage.")
        return

    latest = versions[-1]
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logging.info("Model version %s promoted to Production.", latest.version)


def pipeline_success_notification(**kwargs):
    ti = kwargs["ti"]
    f1 = ti.xcom_pull(task_ids="evaluate", key="test_f1")
    auc = ti.xcom_pull(task_ids="evaluate", key="test_roc_auc")
    rows = ti.xcom_pull(task_ids="validate", key="row_count")
    logging.info(
        "Pipeline SUCCESS | rows=%s | F1=%.4f | AUC=%.4f",
        rows, f1 or 0, auc or 0,
    )


def validation_failed_handler(**kwargs):
    logging.error("Data validation failed. Aborting pipeline.")
    raise ValueError("Data validation failed — pipeline aborted.")


with DAG(
    dag_id="insurance_fraud_ml_pipeline",
    default_args=DEFAULT_ARGS,
    description="End-to-end ML pipeline for Insurance Fraud Detection",
    schedule_interval="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "fraud", "da5402"],
) as dag:

    t_ingest = PythonOperator(task_id="ingest", python_callable=ingest_task)

    t_validate = PythonOperator(task_id="validate", python_callable=validate_task)

    t_branch = BranchPythonOperator(
        task_id="branch_on_validation",
        python_callable=decide_on_validation,
    )

    # FIX: use default ALL_SUCCESS trigger rule (removed NONE_FAILED_MIN_ONE_SUCCESS
    # which caused this task to fire even when branch chose the preprocess path)
    t_validation_failed = PythonOperator(
        task_id="validation_failed",
        python_callable=validation_failed_handler,
    )

    t_preprocess = PythonOperator(task_id="preprocess", python_callable=preprocess_task)
    t_train = PythonOperator(task_id="train", python_callable=train_task)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate_task)
    t_register = PythonOperator(task_id="register_model", python_callable=register_model_task)

    t_notify = PythonOperator(
        task_id="notify_success",
        python_callable=pipeline_success_notification,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    t_ingest >> t_validate >> t_branch
    t_branch >> t_validation_failed
    t_branch >> t_preprocess >> t_train >> t_evaluate >> t_register >> t_notify
