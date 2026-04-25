"""
backend/app/routers/health.py
──────────────────────────────
Health check endpoints required by the deployment guidelines.
/health  → liveness probe
/ready   → readiness probe (checks model availability)
/info    → model metadata
"""

import os
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.schemas.claim import HealthResponse, ModelInfoResponse

router = APIRouter()

_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    """Returns service health. Returns 200 if the server is running."""
    from src.monitoring.metrics import MODEL_LOADED

    model_loaded = MODEL_LOADED._value.get() == 1.0

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get("/ready", summary="Readiness probe")
async def ready():
    """
    Returns 200 only when the model is loaded and predictions can be served.
    Used by Docker orchestration to route traffic.
    """
    try:
        from src.models.predict import get_model, get_preprocessor
        get_model()
        get_preprocessor()
        return JSONResponse(status_code=200, content={"ready": True})
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": str(e)},
        )


@router.get("/info", response_model=ModelInfoResponse, summary="Model metadata")
async def info():
    """Returns metadata about the currently loaded model."""
    import yaml
    root = Path(__file__).resolve().parents[4]
    params_path = root / "params.yaml"
    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = yaml.safe_load(f)

    return ModelInfoResponse(
        model_name=os.getenv("MODEL_NAME", "insurance_fraud_model"),
        model_stage=os.getenv("MODEL_STAGE", "Production"),
        algorithm=params.get("model", {}).get("algorithm", "unknown"),
        threshold=params.get("model", {}).get("threshold", 0.4),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
