"""
backend/app/routers/predict.py
───────────────────────────────
Prediction endpoints.

FIX: threshold is now read from request.app.state.threshold (set once at
     startup in main.py) instead of re-reading params.yaml on every call.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from backend.app.schemas.claim import (
    BatchClaimRequest,
    BatchPredictionResponse,
    ClaimRequest,
    PredictionResponse,
)
from src.monitoring.metrics import (
    FRAUD_PROBABILITY,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
)

logger = logging.getLogger(__name__)
router = APIRouter()

RISK_MESSAGES = {
    "LOW": "This claim appears legitimate. No immediate action required.",
    "MEDIUM": "This claim has moderate fraud indicators. Consider a manual review.",
    "HIGH": "This claim shows high fraud indicators. Escalate for investigation.",
}


def _run_prediction(claim: ClaimRequest, threshold: float) -> PredictionResponse:
    from src.models.predict import predict

    features = claim.model_dump(by_alias=True)
    result = predict(features, threshold=threshold)

    FRAUD_PROBABILITY.observe(result["fraud_probability"])
    PREDICTION_COUNT.labels(result="fraud" if result["is_fraud"] else "legitimate").inc()

    return PredictionResponse(
        claim_id=str(uuid.uuid4()),
        fraud_probability=result["fraud_probability"],
        is_fraud=result["is_fraud"],
        risk_score=result["risk_score"],
        risk_tier=result["risk_tier"],
        threshold_used=result["threshold_used"],
        message=RISK_MESSAGES[result["risk_tier"]],
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraud for a single insurance claim",
)
async def predict_single(
    request: Request,
    claim: ClaimRequest,
    threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override decision threshold (default from params.yaml)",
    ),
):
    try:
        # FIX: use pre-loaded threshold from app state; no disk read per request
        effective_threshold = threshold if threshold is not None else request.app.state.threshold
        return _run_prediction(claim, effective_threshold)

    except FileNotFoundError as e:
        PREDICTION_ERRORS.labels(error_type="model_not_found").inc()
        logger.error("Model not found: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train and register a model first.",
        )
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="prediction_error").inc()
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch fraud prediction (up to 100 claims)",
)
async def predict_batch(request: Request, payload: BatchClaimRequest):
    try:
        threshold = request.app.state.threshold
        predictions = [_run_prediction(claim, threshold) for claim in payload.claims]
        fraud_count = sum(1 for p in predictions if p.is_fraud)

        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            fraud_count=fraud_count,
            legitimate_count=len(predictions) - fraud_count,
        )
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="batch_error").inc()
        logger.exception("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/model/reload",
    summary="Reload model from MLflow registry",
    tags=["Admin"],
)
async def reload_model():
    try:
        from src.models.predict import reload_model
        reload_model()
        logger.info("Model reloaded by API call.")
        return {"message": "Model reloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
