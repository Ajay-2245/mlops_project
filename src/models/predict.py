"""
src/models/predict.py
──────────────────────
Stateless prediction helper used by the FastAPI backend.
Loads the model from MLflow Model Registry (or local fallback).
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# ── Singleton model cache ────────────────────────────────────────────────────
_model = None
_preprocessor = None


def _load_from_registry(
    model_name: str,
    stage: str = "Production",
    tracking_uri: str = "http://localhost:5000",
) -> Any:
    """Load model from MLflow model registry."""
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{stage}"
    logger.info("Loading model from MLflow registry: %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def _load_from_local() -> Any:
    """Fallback: load model from local pickle."""
    local_path = ROOT / "data/processed/model.pkl"
    if not local_path.exists():
        raise FileNotFoundError(f"No model found at {local_path}")
    with open(local_path, "rb") as f:
        return pickle.load(f)


def _load_preprocessor() -> Any:
    local_path = ROOT / "data/processed/preprocessor.pkl"
    if not local_path.exists():
        raise FileNotFoundError(f"No preprocessor found at {local_path}")
    with open(local_path, "rb") as f:
        return pickle.load(f)


def get_model():
    """Get the model, initializing from registry or local if needed."""
    global _model
    if _model is not None:
        return _model

    model_name = os.getenv("MODEL_NAME", "insurance_fraud_model")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    try:
        _model = _load_from_registry(model_name, model_stage, tracking_uri)
        logger.info("Model loaded from MLflow registry.")
    except Exception as e:
        logger.warning("Registry load failed (%s). Falling back to local.", e)
        _model = _load_from_local()
        logger.info("Model loaded from local pickle.")

    return _model


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = _load_preprocessor()
        logger.info("Preprocessor loaded.")
    return _preprocessor


def predict(
    features: Dict[str, Any],
    threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Run prediction pipeline on a single claim dict.
    Returns fraud probability, label, and risk score.
    """
    model = get_model()
    preprocessor = get_preprocessor()

    df = pd.DataFrame([features])

    # Apply feature engineering
    from src.features.engineer import create_derived_features
    df = create_derived_features(df)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[0, 1]
    label = int(proba >= threshold)

    # Risk score 0–100
    risk_score = round(proba * 100, 1)
    risk_tier = (
        "LOW" if risk_score < 30
        else "MEDIUM" if risk_score < 60
        else "HIGH"
    )

    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": bool(label),
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "threshold_used": threshold,
    }


def reload_model() -> None:
    """Force reload — called when a new model version is promoted."""
    global _model, _preprocessor
    _model = None
    _preprocessor = None
    logger.info("Model cache cleared — will reload on next prediction.")
