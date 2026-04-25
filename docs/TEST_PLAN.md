# Test Plan — Insurance Fraud Detection System

## 1. Overview

| Item | Detail |
|------|--------|
| Project | Insurance Fraud Detection MLOps System |
| Version | 1.0.0 |
| Tools | pytest, httpx, unittest.mock |

---

## 2. Test Categories

### 2.1 Unit Tests (`backend/tests/test_api.py`)

| ID | Test Case | Input | Expected | Status |
|----|-----------|-------|----------|--------|
| U01 | GET `/` returns 200 | — | `{"service": ...}` | PASS |
| U02 | GET `/health` returns status | — | `{status, model_loaded, ...}` | PASS |
| U03 | GET `/metrics` accessible | — | 200 + Prometheus text | PASS |
| U04 | POST `/api/v1/predict` valid claim | Valid ClaimRequest | 200 + PredictionResponse | PASS |
| U05 | POST `/api/v1/predict` missing `age` | ClaimRequest minus age | 422 | PASS |
| U06 | POST `/api/v1/predict` age=150 | age out of range | 422 | PASS |
| U07 | POST `/api/v1/predict` invalid sex | sex="OTHER" | 422 | PASS |
| U08 | POST `/api/v1/predict/batch` 2 claims | BatchClaimRequest | 200, total=2 | PASS |
| U09 | POST `/api/v1/predict/batch` 101 claims | Oversized batch | 422 | PASS |
| U10 | PredictionResponse has all fields | Valid response | All required keys present | PASS |
| U11 | risk_score in [0, 100] | Valid response | 0 ≤ score ≤ 100 | PASS |
| U12 | fraud_probability in [0, 1] | Valid response | 0.0 ≤ prob ≤ 1.0 | PASS |

### 2.2 Data Pipeline Tests

| ID | Test Case | Expected |
|----|-----------|----------|
| D01 | `validate.py` on clean data | Passes all checks |
| D02 | `validate.py` with missing target column | Fails + logs error |
| D03 | `preprocess.py` output shapes | X_train.shape[1] == X_test.shape[1] |
| D04 | `preprocess.py` no data leakage | Preprocessor fit only on X_train |
| D05 | Baseline stats JSON has all features | All numerical columns present |

### 2.3 Model Tests

| ID | Test Case | Expected |
|----|-----------|----------|
| M01 | Model trains without error | `model.pkl` created |
| M02 | Test F1 ≥ 0.60 | Acceptance criterion |
| M03 | Test ROC-AUC ≥ 0.70 | Acceptance criterion |
| M04 | Fraud recall ≥ 0.65 | Business criterion |
| M05 | Threshold 0.4 gives higher recall than 0.5 | Recall increases |

### 2.4 Integration Tests

| ID | Test Case | Expected |
|----|-----------|----------|
| I01 | Backend container starts | `/health` returns 200 |
| I02 | Frontend serves index.html | HTTP 200 |
| I03 | Frontend can reach backend | CORS headers present |
| I04 | MLflow server accessible | Port 5000 responds |
| I05 | Airflow DAG visible in UI | DAG listed in Airflow |
| I06 | Prometheus scrapes backend | `fraud_api_requests_total` visible |

---

## 3. Acceptance Criteria

A release is accepted when:
- All unit tests PASS (0 failures)
- Test F1 ≥ 0.60
- Test ROC-AUC ≥ 0.70
- All Docker containers start without error
- `/health` returns `"status": "healthy"`
- API p95 latency < 200 ms under 10 concurrent requests

---

## 4. Running Tests

```bash
# Unit tests
pytest backend/tests/ -v

# With coverage
pytest backend/tests/ -v --cov=backend --cov=src --cov-report=html

# DVC pipeline test
dvc repro --dry
```
