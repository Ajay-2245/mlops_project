"""
src/data/preprocess.py
───────────────────────
Reads raw data, applies feature engineering, splits into train/val/test,
fits the sklearn preprocessor on train only, and persists all artefacts.
Supports both .xlsx and .csv input files.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.engineer import (
    build_preprocessor,
    compute_baseline_stats,
    create_derived_features,
    encode_target,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = ROOT / "params.yaml"
PROCESSED = ROOT / "data/processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()
    data_cfg = params["data"]
    feat_cfg = params["features"]

    raw_path = ROOT / data_cfg["raw_path"]
    logger.info("Loading raw data from %s", raw_path)

    # Support both Excel and CSV
    if raw_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(raw_path)
    else:
        df = pd.read_csv(raw_path)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    drop_cols = [c for c in feat_cfg["drop_columns"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Dropped columns: %s", drop_cols)

    target_col = feat_cfg["target_column"]
    y = encode_target(df[target_col])
    df.drop(columns=[target_col], inplace=True)
    logger.info("Target distribution — fraud=1: %.2f%%", y.mean() * 100)

    df = create_derived_features(df)

    derived_num = [
        "claim_to_premium_ratio", "is_night_incident", "multi_vehicle",
        "component_sum", "claim_discrepancy", "has_witnesses", "no_police_report",
    ]
    derived_num = [c for c in derived_num if c in df.columns]

    cat_cols = [c for c in feat_cfg["categorical_columns"] if c in df.columns]
    num_cols = [c for c in feat_cfg["numerical_columns"] if c in df.columns]

    # ── Train / val / test split ─────────────────────────────────────────────
    random_state = data_cfg["random_state"]
    test_size = data_cfg["test_size"]
    val_size = data_cfg["val_size"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_trainval,
    )
    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # Baseline stats from train only (no data leakage)
    baseline_stats = compute_baseline_stats(X_train, num_cols + derived_num)
    with open(PROCESSED / "baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)
    logger.info("Baseline stats saved (training set only).")

    # Fit preprocessor on train only
    preprocessor = build_preprocessor(num_cols, cat_cols, derived_num)
    preprocessor.fit(X_train)
    logger.info("Preprocessor fitted on training data.")

    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    splits = {
        "X_train": X_train_t, "X_val": X_val_t, "X_test": X_test_t,
        "y_train": y_train.values, "y_val": y_val.values, "y_test": y_test.values,
    }
    for name, arr in splits.items():
        out = PROCESSED / f"{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(arr, f)
        logger.info("Saved %s → shape %s", out.name, arr.shape)

    with open(PROCESSED / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Preprocessor saved.")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
