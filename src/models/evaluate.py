"""
src/models/evaluate.py
───────────────────────
Final evaluation on hold-out test set.

FIX: get_tracking_uri() reads MLFLOW_TRACKING_URI env var first
     (set to http://mlflow:5000 in Docker), falls back to params.yaml.
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = ROOT / "params.yaml"
PROCESSED = ROOT / "data/processed"


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def get_tracking_uri(mlflow_cfg: dict) -> str:
    """
    Read MLflow tracking URI from env var first (Docker: http://mlflow:5000),
    fall back to params.yaml (local dev: http://localhost:5000).
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        logger.info("Using MLflow URI from environment: %s", env_uri)
        return env_uri
    uri = mlflow_cfg["tracking_uri"]
    logger.info("Using MLflow URI from params.yaml: %s", uri)
    return uri


def main() -> None:
    params = load_params()
    threshold = params["model"]["threshold"]
    mlflow_cfg = params["mlflow"]

    # Load artefacts
    with open(PROCESSED / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(PROCESSED / "X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(PROCESSED / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    logger.info("Evaluating on test set — shape: %s", X_test.shape)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "test_accuracy": round(float((preds == y_test).mean()), 4),
        "test_precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "threshold": threshold,
        "test_size": len(y_test),
        "fraud_detected": int(preds.sum()),
        "actual_fraud": int(y_test.sum()),
    }

    logger.info("Test metrics: %s", metrics)

    report = classification_report(y_test, preds, target_names=["Legitimate", "Fraud"])
    logger.info("\n%s", report)

    # Confusion matrix CSV
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame({
        "actual": ["Legitimate", "Legitimate", "Fraud", "Fraud"],
        "predicted": ["Legitimate", "Fraud", "Legitimate", "Fraud"],
        "count": [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]],
    })
    cm_df.to_csv(PROCESSED / "confusion_matrix.csv", index=False)

    with open(PROCESSED / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # FIX: use env var so Docker containers connect to http://mlflow:5000
    tracking_uri = get_tracking_uri(mlflow_cfg)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name="evaluate_test"):
        mlflow.log_metrics(metrics)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_artifact(str(PROCESSED / "confusion_matrix.csv"))
        logger.info("Evaluation metrics logged to MLflow.")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()