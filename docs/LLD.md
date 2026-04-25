# Low-Level Design (LLD) — Insurance Fraud Detection System

## 1. API Endpoint Definitions

All endpoints are served by the FastAPI backend at `http://localhost:8000`.

---

### 1.1 `GET /health`
**Purpose:** Liveness probe — confirms the server process is running.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 342.1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` \| `"degraded"` \| `"unhealthy"` |
| `model_loaded` | bool | Whether the ML model is loaded in memory |
| `version` | string | API version string |
| `uptime_seconds` | float | Seconds since API startup |

---

### 1.2 `GET /ready`
**Purpose:** Readiness probe — 200 only when model can serve predictions.

**Response `200`:** `{"ready": true}`  
**Response `503`:** `{"ready": false, "reason": "<error>"}`

---

### 1.3 `POST /api/v1/predict`
**Purpose:** Submit a single insurance claim for fraud prediction.

**Query Param:** `threshold` (float 0–1, default 0.4)

**Request Body:** `ClaimRequest` — see Section 2.

**Response `200 OK`:** `PredictionResponse` — see Section 3.

| Code | Reason |
|------|--------|
| 422 | Validation error |
| 503 | Model not loaded |
| 500 | Prediction failure |

---

### 1.4 `POST /api/v1/predict/batch`
**Purpose:** Submit 1–100 claims in one request.

```json
{ "claims": [ <ClaimRequest>, ... ] }
```

**Response:**
```json
{ "predictions": [...], "total": 2, "fraud_count": 1, "legitimate_count": 1 }
```

---

### 1.5 `POST /api/v1/model/reload`
Force-reload model from MLflow registry.

---

### 1.6 `GET /metrics`
Prometheus text-format scrape endpoint.

---

## 2. Request Schema — ClaimRequest

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `months_as_customer` | int | 0–600 | Months as customer |
| `age` | int | 16–100 | Policyholder age |
| `policy_state` | string | — | US state code |
| `policy_csl` | string | — | Combined single limit |
| `policy_deductable` | int | ≥0 | Deductible amount |
| `policy_annual_premium` | float | ≥0 | Annual premium |
| `insured_sex` | enum | MALE\|FEMALE | Sex |
| `insured_education_level` | string | — | Education |
| `insured_occupation` | string | — | Occupation |
| `insured_relationship` | string | — | Relationship to policy |
| `capital-gains` | float | — | Capital gains |
| `capital-loss` | float | — | Capital loss |
| `incident_type` | string | — | Incident type |
| `collision_type` | string? | — | Collision type |
| `incident_severity` | string | — | Severity |
| `authorities_contacted` | string? | — | Authority contacted |
| `incident_state` | string | — | State of incident |
| `incident_city` | string | — | City of incident |
| `incident_hour_of_the_day` | int | 0–23 | Hour |
| `number_of_vehicles_involved` | int | 1–10 | Vehicles |
| `bodily_injuries` | int | 0–10 | Injuries |
| `witnesses` | int | 0–10 | Witnesses |
| `police_report_available` | enum | YES\|NO | Police report |
| `total_claim_amount` | float | ≥0 | Total claim USD |
| `injury_claim` | float | ≥0 | Injury component |
| `property_claim` | float | ≥0 | Property component |
| `vehicle_claim` | float | ≥0 | Vehicle component |
| `auto_make` | string | — | Vehicle make |
| `auto_year` | int | 1980–2025 | Vehicle year |

---

## 3. Response Schema — PredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| `claim_id` | string | UUID for this request |
| `fraud_probability` | float [0,1] | Model probability |
| `is_fraud` | bool | probability >= threshold |
| `risk_score` | float [0,100] | probability × 100 |
| `risk_tier` | enum | LOW / MEDIUM / HIGH |
| `threshold_used` | float | Applied threshold |
| `message` | string | Human-readable verdict |

---

## 4. Internal Module Interfaces

### `src/models/predict.predict(features, threshold)`
```python
def predict(features: dict, threshold: float = 0.4) -> dict:
    # Input:  raw claim dict
    # Output: {fraud_probability, is_fraud, risk_score, risk_tier, threshold_used}
```

### `src/features/engineer.create_derived_features(df)`
```python
# Adds: claim_to_premium_ratio, is_night_incident, multi_vehicle,
#        component_sum, claim_discrepancy, has_witnesses, no_police_report
```

---

## 5. Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fraud_api_requests_total` | Counter | method, endpoint, status_code | HTTP requests |
| `fraud_api_request_latency_seconds` | Histogram | endpoint | Latency |
| `fraud_predictions_total` | Counter | result | Prediction count |
| `fraud_prediction_probability` | Histogram | — | Score distribution |
| `fraud_prediction_errors_total` | Counter | error_type | Errors |
| `fraud_model_loaded` | Gauge | — | Model health |
| `fraud_feature_drift_score` | Gauge | feature_name | PSI drift |
| `fraud_drift_alert` | Gauge | — | 1 if drift detected |

---

## 6. Acceptance Criteria

| Criterion | Target |
|-----------|--------|
| Test F1 | ≥ 0.60 |
| Test ROC-AUC | ≥ 0.70 |
| API latency p95 | < 200 ms |
| Error rate | < 5% |
| Fraud recall | ≥ 0.65 |
