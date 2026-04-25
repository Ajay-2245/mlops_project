"""
src/models/train.py
────────────────────
MLflow 3.x compatible training script.

In MLflow 3.x, log_model() stores models in the new LoggedModel system
(visible in Models / Model Registry sections), NOT in the run Artifacts tab.
To show the model in Artifacts tab we also log it via log_artifact().
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
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


def load_preprocessor():
    with open(PROCESSED / "preprocessor.pkl", "rb") as f:
        return pickle.load(f)


def get_feature_names(preprocessor) -> list:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


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
            raise ImportError("xgboost is not installed.")
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
    preprocessor = load_preprocessor()
    feature_names = get_feature_names(preprocessor)

    logger.info("Data loaded — train: %s | val: %s", X_train.shape, X_val.shape)
    logger.info("Feature names resolved: %d features", len(feature_names))

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

        # ── Feature importances ──────────────────────────────────────────────
        if hasattr(model, "feature_importances_") and feature_names:
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

            mlflow.log_text(
                importance_df.head(20).to_string(index=False),
                "feature_importances.txt",
            )
            importance_df.to_csv(PROCESSED / "feature_importances.csv", index=False)
            logger.info("Top 5 features: %s", importance_df.head(5)["feature"].tolist())

        # ── Persist model locally ────────────────────────────────────────────
        model_path = PROCESSED / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # ── Artifacts tab: log pkl file directly ────────────────────────────
        # In MLflow 3.x, log_model goes to Models section NOT Artifacts tab.
        # log_artifact puts the file in the Artifacts tab.
        mlflow.log_artifact(str(model_path), artifact_path="model")
        logger.info("model.pkl logged to run Artifacts tab under 'model/' folder.")

        # ── Models section + Model Registry: log_model with name= ───────────
        # This is what populates the Models / Model Registry sections in MLflow 3.x.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=mlflow_cfg["registered_model_name"],
            input_example=X_train[:5],
        )
        logger.info("Model logged to Models section and registered in Model Registry.")

    logger.info("Training pipeline complete. Run ID: %s", run.info.run_id)


if __name__ == "__main__":
    main()