# Insurance Fraud Detection — Design Document

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP (port 3000)
┌──────────────────────────────▼──────────────────────────────────────────┐
│                    FRONTEND — Nginx + HTML/JS                            │
│  • Claim input form          • Risk gauge visualisation                  │
│  • Pipeline console tab      • Responsive / mobile-friendly              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ REST API (port 8000)
┌──────────────────────────────▼──────────────────────────────────────────┐
│               BACKEND — FastAPI (Python 3.11)                            │
│  POST /api/v1/predict         GET /health   GET /ready                  │
│  POST /api/v1/predict/batch   GET /info     GET /metrics                 │
│  Prometheus instrumentation   Pydantic validation   Logging              │
└──────┬──────────────────────────────────────┬───────────────────────────┘
       │ mlflow.sklearn.load_model             │ prometheus_client
┌──────▼──────────────┐             ┌─────────▼──────────────────────────┐
│  MLFLOW SERVER       │             │  PROMETHEUS + GRAFANA               │
│  • Model registry    │             │  • Scrapes /metrics every 10s       │
│  • Experiment runs   │             │  • Drift score gauges               │
│  • Artifact store    │             │  • Latency histograms               │
│  port 5000           │             │  • Fraud rate over time             │
└──────▲──────────────┘             └────────────────────────────────────┘
       │ log_model / transition stage
┌──────┴──────────────────────────────────────────────────────────────────┐
│              ML PIPELINE (DVC + Airflow)                                 │
│                                                                          │
│  [ingest] → [validate] → [preprocess] → [train] → [evaluate] →          │
│  [register_model]                                                        │
│                                                                          │
│  • DVC tracks every artifact hash (data, model, preprocessor)           │
│  • Airflow schedules daily re-runs                                       │
│  • MLflow logs every experiment run                                      │
└─────────────────────────────────────────────────────────────────────────┘
       ↑
  data/raw/insurance_claims.csv  (Kaggle: buntyshah/auto-insurance-claims-data)
```

---

## 2. High-Level Design (HLD)

### 2.1 Design Goals
| Goal | Implementation |
|---|---|
| Loose coupling | Frontend ↔ Backend connected only via REST (configurable `API_BASE`) |
| Reproducibility | Every run tied to a DVC commit hash + MLflow run ID |
| Scalability | FastAPI async + Uvicorn workers; Docker Compose ready for Swarm |
| Observability | Prometheus metrics on every endpoint + Grafana dashboards |
| Automation | Airflow DAG triggers full pipeline nightly |

### 2.2 Design Paradigm
The backend follows the **functional paradigm** — each module is a stateless function pipeline. The prediction pathway is:

```
HTTP Request → Pydantic Validation → Feature Engineering → Preprocessor.transform() → Model.predict_proba() → Threshold → JSON Response
```

### 2.3 System Components

**Frontend** (`frontend/`)
- Single-page HTML/JS application served by Nginx
- Calls backend REST API via `fetch()`
- Two views: Predict form + ML Pipeline console
- No framework dependency — pure vanilla JS

**Backend** (`backend/`)
- FastAPI with two routers: `health` and `predict`
- Prometheus middleware on every request
- Pydantic v2 schemas for strict input validation
- Model loaded at startup from MLflow registry (falls back to local pickle)

**ML Pipeline** (`src/`)
- `data/ingest.py` — Kaggle API download
- `data/validate.py` — schema + quality checks
- `data/preprocess.py` — feature engineering + splits
- `models/train.py` — training with full MLflow tracking
- `models/evaluate.py` — test-set evaluation + DVC metrics
- `features/engineer.py` — derived features (claim ratio, night incident, etc.)

**MLflow** — experiment tracking, model registry, artifact store

**DVC** — pipeline DAG, data/model versioning, reproducibility

**Airflow** — daily schedule, branching on validation failure, model promotion

**Prometheus + Grafana** — real-time monitoring, drift alerts

### 2.4 Data Flow
```
Kaggle CSV → ingest → raw CSV → validate → preprocess →
  (X_train, X_val, X_test, preprocessor, baseline_stats) →
  train → model.pkl + MLflow run → evaluate → metrics.json →
  register → Production stage in MLflow registry →
  FastAPI loads model → serves predictions
```

---

## 3. Low-Level Design (LLD) — API Endpoint Specification

### Base URL: `http://localhost:8000`

---

#### `GET /health`
**Description:** Liveness probe  
**Response 200:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 120.5
}
```

---

#### `GET /ready`
**Description:** Readiness probe — returns 503 if model not loaded  
**Response 200:** `{"ready": true}`  
**Response 503:** `{"ready": false, "reason": "<error>"}`

---

#### `GET /info`
**Description:** Returns model metadata  
**Response 200:**
```json
{
  "model_name": "insurance_fraud_model",
  "model_stage": "Production",
  "algorithm": "random_forest",
  "threshold": 0.4,
  "mlflow_tracking_uri": "http://localhost:5000"
}
```

---

#### `POST /api/v1/predict`
**Description:** Predict fraud for a single insurance claim  
**Query param:** `threshold` (float, optional, 0.0–1.0) — overrides default  

**Request body (ClaimRequest):**
```json
{
  "months_as_customer": 36,
  "age": 35,
  "policy_state": "OH",
  "policy_csl": "250/500",
  "policy_deductable": 500,
  "policy_annual_premium": 1200.0,
  "umbrella_limit": 0,
  "insured_sex": "MALE",
  "insured_education_level": "MD",
  "insured_occupation": "craft-repair",
  "insured_hobbies": "chess",
  "insured_relationship": "husband",
  "capital-gains": 0,
  "capital-loss": 0,
  "incident_type": "Single Vehicle Collision",
  "collision_type": "Front Collision",
  "incident_severity": "Major Damage",
  "authorities_contacted": "Police",
  "incident_state": "OH",
  "incident_city": "Columbus",
  "incident_hour_of_the_day": 14,
  "number_of_vehicles_involved": 1,
  "bodily_injuries": 1,
  "witnesses": 0,
  "police_report_available": "YES",
  "total_claim_amount": 65000,
  "injury_claim": 10000,
  "property_claim": 5000,
  "vehicle_claim": 50000,
  "auto_make": "Saab",
  "auto_year": 2012
}
```

**Response 200 (PredictionResponse):**
```json
{
  "claim_id": "uuid-v4",
  "fraud_probability": 0.7234,
  "is_fraud": true,
  "risk_score": 72.3,
  "risk_tier": "HIGH",
  "threshold_used": 0.4,
  "message": "This claim shows high fraud indicators. Escalate for investigation."
}
```
**Response 422:** Pydantic validation error  
**Response 503:** Model not available  
**Response 500:** Prediction runtime error  

---

#### `POST /api/v1/predict/batch`
**Description:** Batch prediction for up to 100 claims  
**Request body:**
```json
{ "claims": [ <ClaimRequest>, ... ] }
```
**Response 200 (BatchPredictionResponse):**
```json
{
  "predictions": [ <PredictionResponse>, ... ],
  "total": 2,
  "fraud_count": 1,
  "legitimate_count": 1
}
```

---

#### `POST /api/v1/model/reload`
**Description:** Force-reload the model from MLflow registry  
**Response 200:** `{"message": "Model reloaded successfully."}`

---

#### `GET /metrics`
**Description:** Prometheus metrics exposition endpoint  
**Response:** Prometheus text format (scraped by Prometheus server)

---

## 4. Feature Engineering — Derived Features

| Feature | Formula | Rationale |
|---|---|---|
| `claim_to_premium_ratio` | total_claim / annual_premium | High ratio = suspicious |
| `is_night_incident` | hour ≥ 22 or hour ≤ 6 | Night incidents more likely fraudulent |
| `multi_vehicle` | vehicles_involved > 1 | Staged accident indicator |
| `component_sum` | injury + property + vehicle | Cross-check with total |
| `claim_discrepancy` | |total − component_sum| | Accounting mismatch = red flag |
| `has_witnesses` | witnesses > 0 | Witnesses reduce fraud likelihood |
| `no_police_report` | police_report == "NO" | No report = suspicious for major incidents |

---

## 5. Acceptance Criteria

| Metric | Threshold | Rationale |
|---|---|---|
| Test F1 Score | ≥ 0.60 | Balances precision and recall for imbalanced fraud data |
| Test ROC-AUC | ≥ 0.70 | Discriminative ability across all thresholds |
| API latency (p95) | < 200 ms | As per business metric in guidelines |
| Error rate | < 5% | Prometheus alert threshold |

---

## 6. Test Plan

### Test Cases

| ID | Type | Description | Expected Result | Status |
|---|---|---|---|---|
| TC-01 | Unit | Valid claim → `200 OK` with all fields | prediction returned | PASS |
| TC-02 | Unit | Missing `age` field → `422` | validation error | PASS |
| TC-03 | Unit | `age=150` → `422` | out-of-range error | PASS |
| TC-04 | Unit | `insured_sex="OTHER"` → `422` | enum validation | PASS |
| TC-05 | Unit | Batch of 2 claims → correct totals | fraud_count matches | PASS |
| TC-06 | Unit | Batch > 100 → `422` | size limit enforced | PASS |
| TC-07 | Integration | `/health` returns `200` | model_loaded=true | PASS |
| TC-08 | Integration | `/metrics` returns Prometheus text | scraping works | PASS |
| TC-09 | E2E | Full DVC pipeline runs without error | eval_metrics.json created | — |
| TC-10 | E2E | Airflow DAG completes all tasks | model promoted to Production | — |

### Test Report Template
```
Total test cases: 10
Unit (automated): 8
Integration (automated): 2  
E2E (manual): 2

Results (pytest run):
  Passed: 8  Failed: 0  Skipped: 0

Acceptance criteria met: F1 ≥ 0.60, AUC ≥ 0.70 → ✅
```

---

## 7. User Manual (Non-Technical)

### How to Submit a Claim for Fraud Check

1. Open your browser and go to **http://localhost:3000**
2. You will see the **InsureGuard** dashboard
3. Fill in the claim form:
   - **Policy Information** — how long the customer has been with the company, their state, deductible
   - **Insured Person Details** — age, sex, education, occupation
   - **Incident Details** — what happened, when, where, severity
   - **Claim Amounts** — total and breakdown
   - **Vehicle Details** — make and year
4. Click **"Analyse Claim"**
5. The result panel will show:
   - A **risk score from 0–100**
   - A **risk tier**: LOW (green), MEDIUM (yellow), or HIGH (red)
   - A **recommendation** on whether to review or escalate
6. For testing, click **"Load Example"** to pre-fill a sample claim
7. To check a new claim, click **"Clear"** and repeat

### How to Monitor the System
- **Airflow** (http://localhost:8080) — view pipeline runs, check for errors
- **MLflow** (http://localhost:5000) — view experiment results, compare models
- **Grafana** (http://localhost:3001) — live monitoring dashboard (login: admin/admin)
