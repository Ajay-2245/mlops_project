# High-Level Design (HLD) — Insurance Fraud Detection System

## 1. Architecture Overview

The system follows a **loosely-coupled microservices architecture** where the frontend UI, backend inference engine, and MLOps tooling are independent blocks connected via REST APIs and Docker networking.

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│              Frontend UI  (Nginx :3000)                         │
│         HTML + Vanilla JS — calls /api/v1/predict               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST (JSON)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               BACKEND — FastAPI (:8000)                         │
│  /api/v1/predict   /api/v1/predict/batch   /health   /metrics   │
│  ┌───────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │ Pydantic      │  │ Prediction Engine │  │ Prometheus     │   │
│  │ Validation    │  │ (sklearn model)   │  │ Instrumentation│   │
│  └───────────────┘  └──────────────────┘  └────────────────┘   │
└──────────┬────────────────────┬────────────────────────────────┘
           │ load model         │ scrape /metrics
           ▼                    ▼
┌──────────────────┐  ┌──────────────────────────────────────────┐
│  MLflow Server   │  │   Prometheus (:9090) + Grafana (:3001)   │
│  (:5000)         │  │   Dashboards: latency, fraud rate, drift  │
│  Model Registry  │  └──────────────────────────────────────────┘
│  Experiment Logs │
└──────────────────┘
           ▲
           │ log runs / promote versions
           │
┌──────────────────────────────────────────────────────────────────┐
│              ML PIPELINE (DVC + Airflow)                         │
│                                                                   │
│  ingest → validate → preprocess → train → evaluate → register   │
│                                                                   │
│  Airflow Scheduler: daily 02:00 UTC (:8080)                      │
│  DVC: data + model versioning, pipeline DAG                      │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Data Storage                    │
│  data/raw/insurance_claims.csv   │
│  data/processed/*.pkl            │
│  (DVC-tracked, Git LFS for large)│
└──────────────────────────────────┘
```

---

## 2. Technology Stack

| Layer              | Technology          | Purpose                              |
|--------------------|---------------------|--------------------------------------|
| Frontend           | HTML5 + Nginx       | Claim input form, risk visualisation |
| Backend API        | FastAPI + Uvicorn   | REST inference engine                |
| ML Model           | scikit-learn RF     | Binary fraud classification          |
| Experiment Track   | MLflow 2.x          | Parameters, metrics, model registry  |
| Data Pipeline      | Apache Airflow 2.8  | Orchestration, scheduling            |
| Data Versioning    | DVC                 | Reproducible data + model versions   |
| Monitoring         | Prometheus + Grafana| Real-time metrics, alerting          |
| Containerisation   | Docker + Compose    | Environment parity, deployment       |
| CI                 | GitHub Actions      | Lint, test, Docker build on push     |
| Source Control     | Git + Git LFS       | Code + large file versioning         |

---

## 3. Design Choices & Rationale

### 3.1 Loose Coupling
The frontend calls the backend exclusively through `POST /api/v1/predict`. There is no shared state. The backend URL is configurable via the `API_BASE` constant in `index.html`. This means either side can be deployed independently.

### 3.2 Random Forest as Default Classifier
The dataset has ~1000 rows with a class imbalance (~24% fraud). Random Forest with `class_weight=balanced` handles both characteristics well without requiring complex oversampling. XGBoost is available as a swap-in via `params.yaml`.

### 3.3 Configurable Decision Threshold (0.4)
Fraud detection is **recall-sensitive** — a missed fraud is more costly than a false alarm. We set the threshold to 0.4 (below 0.5) to increase recall at a small precision cost. This is tunable in `params.yaml` without code changes.

### 3.4 DVC for Reproducibility
Every experiment is reproducible by its Git commit hash + MLflow run ID. Running `dvc repro` on any commit regenerates exactly the same artefacts.

### 3.5 Airflow for Automation
The Airflow DAG includes a branching validation step: if data quality fails, the pipeline aborts before any model training, preventing garbage-in/garbage-out.

---

## 4. Data Flow

```
Kaggle CSV
   │
   ▼ (ingest.py)
data/raw/insurance_claims.csv   ─── DVC tracked ───►  Git hash
   │
   ▼ (validate.py)
validation_report.json          ─── schema / nulls / distribution
   │
   ▼ (preprocess.py + engineer.py)
data/processed/                 ─── X_train, X_val, X_test, preprocessor.pkl
                                    baseline_stats.json (for drift)
   │
   ▼ (train.py)
model.pkl                       ─── MLflow logged + registered
train_metrics.json              ─── DVC metrics
   │
   ▼ (evaluate.py)
eval_metrics.json               ─── test F1, AUC, confusion matrix
   │
   ▼ (Airflow: register_model)
MLflow Production stage         ─── FastAPI loads from here
```

---

## 5. Security Considerations
- Non-root Docker user (`appuser`) in backend container
- CORS configured (tighten `allow_origins` in production)
- No raw PII stored in logs
- Secrets (Kaggle API key) passed via environment variables, never committed

---

## 6. Scalability
- Uvicorn runs with `--workers 2` (increase for higher throughput)
- Batch endpoint supports up to 100 claims per request
- Stateless backend — horizontally scalable behind a load balancer
