"""
backend/app/main.py
────────────────────
FastAPI application entry point.

FIX: threshold is now loaded once at startup and stored in app.state,
     so predict.py routers don't read params.yaml on every request.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.monitoring.metrics import CURRENT_THRESHOLD, MODEL_LOADED, REQUEST_COUNT, REQUEST_LATENCY

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

APP_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Insurance Fraud Detection API ...")

    # Load threshold once — routers read from app.state.threshold
    params_path = ROOT / "params.yaml"
    try:
        with open(params_path) as f:
            params = yaml.safe_load(f)
        app.state.threshold = params["model"]["threshold"]
        CURRENT_THRESHOLD.set(app.state.threshold)
        logger.info("Threshold loaded: %.2f", app.state.threshold)
    except Exception as e:
        app.state.threshold = 0.4
        logger.warning("Could not load threshold from params.yaml (%s). Using default 0.4", e)

    # Warm up model
    try:
        from src.models.predict import get_model, get_preprocessor
        get_model()
        get_preprocessor()
        MODEL_LOADED.set(1)
        logger.info("Model and preprocessor loaded successfully.")
    except Exception as e:
        MODEL_LOADED.set(0)
        logger.warning("Model not yet available: %s", e)

    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Insurance Fraud Detection API",
    description=(
        "Binary classification API that predicts whether an insurance "
        "claim is fraudulent. Part of DA5402 MLOps project."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
        logger.info(
            "%s %s %s %.3fs",
            request.method, request.url.path, response.status_code, duration,
        )
        return response
    except Exception as exc:
        duration = time.time() - start
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.url.path, status_code=500
        ).inc()
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

from backend.app.routers import health, predict  # noqa: E402

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Insurance Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENV", "production") == "development",
        log_level="info",
    )
