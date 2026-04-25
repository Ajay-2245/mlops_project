"""
src/monitoring/metrics.py
──────────────────────────
Prometheus instrumentation for the FastAPI backend.
Tracks: request counts, latency, fraud rate, prediction errors, drift score.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary

# ── Request metrics ──────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# ── Prediction metrics ───────────────────────────────────────────────────────
PREDICTION_COUNT = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions",
    ["result"],        # result = "fraud" | "legitimate"
)

FRAUD_PROBABILITY = Histogram(
    "fraud_prediction_probability",
    "Distribution of fraud probability scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PREDICTION_ERRORS = Counter(
    "fraud_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
)

# ── Model health ─────────────────────────────────────────────────────────────
MODEL_LOADED = Gauge(
    "fraud_model_loaded",
    "Whether the model is successfully loaded (1=yes, 0=no)",
)

CURRENT_THRESHOLD = Gauge(
    "fraud_current_threshold",
    "Current decision threshold used for predictions",
)

# ── Data drift ───────────────────────────────────────────────────────────────
FEATURE_DRIFT_SCORE = Gauge(
    "fraud_feature_drift_score",
    "PSI drift score per feature",
    ["feature_name"],
)

DRIFT_ALERT = Gauge(
    "fraud_drift_alert",
    "1 if drift detected, 0 otherwise",
)

# ── System / pipeline metrics ─────────────────────────────────────────────────
PIPELINE_RUNS = Counter(
    "fraud_pipeline_runs_total",
    "Total Airflow pipeline runs",
    ["status"],       # status = "success" | "failure"
)

LAST_RETRAIN_TIMESTAMP = Gauge(
    "fraud_last_retrain_timestamp",
    "Unix timestamp of last model retraining",
)
