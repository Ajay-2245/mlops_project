FROM apache/airflow:2.10.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    scikit-learn==1.5.0 \
    xgboost==2.0.3 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    openpyxl==3.1.2 \
    mlflow==3.1.1 \
    pyyaml==6.0.1 \
    dvc==3.50.0