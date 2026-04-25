"""
src/models/train.py
────────────────────
Trains the fraud-detection classifier with MLflow experiment tracking.

FIXES:
  - Removed use_label_encoder=False (removed in XGBoost >= 1.6)
  - Removed duplicate mlflow.log_artifact for model.pkl (already logged by log_model)
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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


def load_split(name: str) -> np.ndarray:
    with open(PROCESSED / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def build_model(cfg: dict):
    algo = cfg.get("algorithm", "random_forest")

    if algo == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
            random_state=42,
            n_jobs=-1,
        )
    elif algo == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            scale_pos_weight=cfg.get("scale_pos_weight", 3),
            eval_metric="logloss",
            random_state=42,
        )
    elif algo == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight=cfg["class_weight"],
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def evaluate_split(model, X, y, threshold: float, prefix: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        f"{prefix}_accuracy": round(accuracy_score(y, preds), 4),
        f"{prefix}_precision": round(precision_score(y, preds, zero_division=0), 4),
        f"{prefix}_recall": round(recall_score(y, preds, zero_division=0), 4),
        f"{prefix}_f1": round(f1_score(y, preds, zero_division=0), 4),
        f"{prefix}_roc_auc": round(roc_auc_score(y, proba), 4),
    }


def main() -> None:
    params = load_params()
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]
    threshold = model_cfg["threshold"]

    X_train = load_split("X_train")
    y_train = load_split("y_train")
    X_val = load_split("X_val")
    y_val = load_split("y_val")
    logger.info("Data loaded — train: %s | val: %s", X_train.shape, X_val.shape)

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=f"train_{model_cfg['algorithm']}") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        mlflow.log_params({
            "algorithm": model_cfg["algorithm"],
            "n_estimators": model_cfg.get("n_estimators"),
            "max_depth": model_cfg.get("max_depth"),
            "class_weight": model_cfg.get("class_weight"),
            "threshold": threshold,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "fraud_rate_train": float(y_train.mean()),
        })

        model = build_model(model_cfg)
        logger.info("Training %s ...", model_cfg["algorithm"])
        model.fit(X_train, y_train)
        logger.info("Training complete.")

        train_metrics = evaluate_split(model, X_train, y_train, threshold, "train")
        val_metrics = evaluate_split(model, X_val, y_val, threshold, "val")
        all_metrics = {**train_metrics, **val_metrics}

        mlflow.log_metrics(all_metrics)
        logger.info(
            "Val F1: %.4f | Val ROC-AUC: %.4f",
            val_metrics["val_f1"], val_metrics["val_roc_auc"],
        )

        with open(PROCESSED / "train_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        if hasattr(model, "feature_importances_"):
            mlflow.log_text(
                "\n".join(
                    f"feature_{i}: {v:.6f}"
                    for i, v in enumerate(model.feature_importances_)
                ),
                "feature_importances.txt",
            )

        # FIX: log_model already stores the model; removed duplicate log_artifact call
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=mlflow_cfg["registered_model_name"],
        )
        logger.info("Model logged to MLflow and registered.")

        # Persist locally for DVC pipeline
        with open(PROCESSED / "model.pkl", "wb") as f:
            pickle.dump(model, f)

    logger.info("Training pipeline complete. Run ID: %s", run.info.run_id)


if __name__ == "__main__":
    main()
