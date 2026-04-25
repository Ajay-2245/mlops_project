# Insurance Fraud Detection — MLOps Pipeline

End-to-end MLOps project for binary fraud classification on insurance claims.
Built for DA5402 at IIT Madras.

## Dataset

**Source:** https://www.kaggle.com/datasets/mastmustu/insurance-claims-fraud-data

> **No Kaggle API required.** Download the CSV manually, rename it to
> `insurance_fraud.csv`, and place it at `data/raw/insurance_fraud.csv`.

## Architecture

```
Streamlit UI (8501) → FastAPI Backend (8000) → MLflow Model Registry (5000)
                                    ↑
                             Prometheus (9090) → Grafana (3001)

Airflow (8080) orchestrates: ingest → validate → preprocess → train → evaluate → register
```

## Quick Start

```bash
# 1. Place dataset
mkdir -p data/raw
cp /path/to/insurance_fraud.csv data/raw/

# 2. Start all services
docker-compose up -d

# 3. Run the ML pipeline
pip install -r requirements.txt
dvc repro

# 4. Open the UI
open http://localhost:8501
```

## Services

| Service    | URL                    | Credentials   |
|------------|------------------------|---------------|
| Streamlit  | http://localhost:8501  | —             |
| FastAPI    | http://localhost:8000/docs | —         |
| MLflow     | http://localhost:5000  | —             |
| Airflow    | http://localhost:8080  | admin / admin |
| Grafana    | http://localhost:3001  | admin / admin |
| Prometheus | http://localhost:9090  | —             |

## DVC Pipeline

```
ingest → validate → preprocess → train → evaluate
```

Run with: `dvc repro`
Compare runs: `dvc metrics diff`

## Bug Fixes Applied (v2)

- XGBoost `use_label_encoder` removed (crashed on XGBoost ≥ 1.6)
- Drift detector now uses real training histogram (not synthetic Gaussian)
- Baseline stats computed on train split only (was full dataset — data leakage)
- Airflow `validation_failed` trigger rule fixed
- Threshold loaded once at API startup (was reading params.yaml per request)
- Root `requirements.txt` now installed in CI
- Unused `LabelEncoder` import removed
- Duplicate MLflow model artifact log removed
