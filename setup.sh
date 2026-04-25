#!/usr/bin/env bash
# setup.sh — First-time project setup
# Run this once after cloning the repository.
# Usage: bash setup.sh

set -e
echo "======================================================"
echo "  Insurance Fraud Detection — Project Setup"
echo "======================================================"

# ── 1. Python virtual environment ─────────────────────────────────────────────
echo ""
echo ">>> [1/6] Creating Python virtual environment ..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "    ✅ Python environment ready."

# ── 2. Git init ───────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/6] Initialising Git ..."
if [ ! -d ".git" ]; then
  git init
  git add .
  git commit -m "feat: initial project structure"
  echo "    ✅ Git initialised."
else
  echo "    ✅ Git already initialised."
fi

# ── 3. DVC init ───────────────────────────────────────────────────────────────
echo ""
echo ">>> [3/6] Initialising DVC ..."
dvc init --no-scm 2>/dev/null || dvc init
git add .dvc .dvcignore
git commit -m "feat: initialise DVC" 2>/dev/null || true
echo "    ✅ DVC initialised."

# ── 4. Check Kaggle credentials ───────────────────────────────────────────────
echo ""
echo ">>> [4/6] Checking Kaggle credentials ..."
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "    ✅ kaggle.json found."
else
  echo "    ⚠️  kaggle.json NOT found."
  echo "       Download from: https://www.kaggle.com/settings → API → Create New Token"
  echo "       Place at:      ~/.kaggle/kaggle.json"
  echo "       Then run:      chmod 600 ~/.kaggle/kaggle.json"
fi

# ── 5. Run DVC pipeline ───────────────────────────────────────────────────────
echo ""
echo ">>> [5/6] Running DVC pipeline (ingest → validate → preprocess → train → evaluate) ..."
echo "    This may take a few minutes on first run ..."
dvc repro
echo "    ✅ Pipeline complete."

# ── 6. Docker Compose ─────────────────────────────────────────────────────────
echo ""
echo ">>> [6/6] Starting Docker services ..."
echo "    (Requires Docker Desktop to be running)"
docker compose up --build -d
echo "    ✅ Services started."

echo ""
echo "======================================================"
echo "  🚀 Setup Complete!"
echo "======================================================"
echo ""
echo "  Frontend UI  →  http://localhost:3000"
echo "  API Docs     →  http://localhost:8000/docs"
echo "  MLflow       →  http://localhost:5000"
echo "  Airflow      →  http://localhost:8080  (admin/admin)"
echo "  Grafana      →  http://localhost:3001  (admin/admin)"
echo "  Prometheus   →  http://localhost:9090"
echo ""
