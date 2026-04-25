"""
src/models/predict.py
──────────────────────
Model loading and inference.

MLflow 3.x uses aliases instead of stages.
Load model via:  models:/insurance_fraud_model@champion
instead of:      models:/insurance_fraud_model/Production  (deprecated)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
PARAMS_FILE = ROOT / "params.yaml"

_model = None
_preprocessor = None


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = _load_preprocessor()
    return _preprocessor


def _load_model():
    """Try MLflow registry first (alias), fall back to local pkl."""
    try:
        params = load_params()
        mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
        model_name = params["mlflow"]["registered_model_name"]
        model_alias = params["mlflow"].get("model_alias", "champion")

        model_uri = f"models:/{model_name}@{model_alias}"
        logger.info("Loading model from MLflow registry: %s", model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded from MLflow registry.")
        return model
    except Exception as e:
        logger.warning("Registry load failed (%s). Falling back to local.", e)
        return _load_local_model()


def _load_local_model():
    model_path = PROCESSED / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run 'dvc repro' first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from local pickle.")
    return model


def _load_preprocessor():
    preprocessor_path = PROCESSED / "preprocessor.pkl"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"No preprocessor found at {preprocessor_path}. Run 'dvc repro' first."
        )
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("Preprocessor loaded.")
    return preprocessor


def reload_model():
    """Force reload model from registry (called by /model/reload endpoint)."""
    global _model
    _model = _load_model()
    logger.info("Model reloaded.")


def predict(features: dict, threshold: Optional[float] = None) -> dict:
    """
    Run inference on a single claim.

    Args:
        features: Raw feature dict from the API request.
        threshold: Decision threshold. Falls back to params.yaml if None.

    Returns:
        Dict with fraud_probability, is_fraud, risk_score, risk_tier, threshold_used.
    """
    if threshold is None:
        threshold = load_params()["model"]["threshold"]

    model = get_model()
    preprocessor = get_preprocessor()

    import pandas as pd
    df = pd.DataFrame([features])
    df.replace("?", float("nan"), inplace=True)

    # Apply same derived features as preprocessing
    from src.features.engineer import create_derived_features
    df = create_derived_features(df)

    X = preprocessor.transform(df)
    proba = float(model.predict_proba(X)[0, 1])
    is_fraud = proba >= threshold
    risk_score = round(proba * 100, 1)

    if proba >= 0.6:
        risk_tier = "HIGH"
    elif proba >= 0.3:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": bool(is_fraud),
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "threshold_used": threshold,
    }