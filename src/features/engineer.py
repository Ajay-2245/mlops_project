"""
src/features/engineer.py
─────────────────────────
Feature engineering transforms applied before model training.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def encode_target(series: pd.Series) -> pd.Series:
    """Encode target: Y → 1, N → 0."""
    return series.map({"Y": 1, "N": 0}).astype(int)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-relevant engineered features."""
    df = df.copy()

    if "total_claim_amount" in df.columns and "policy_annual_premium" in df.columns:
        df["claim_to_premium_ratio"] = (
            df["total_claim_amount"] / (df["policy_annual_premium"] + 1)
        )

    if "incident_hour_of_the_day" in df.columns:
        df["is_night_incident"] = df["incident_hour_of_the_day"].apply(
            lambda h: 1 if (h >= 22 or h <= 6) else 0
        )

    if "number_of_vehicles_involved" in df.columns:
        df["multi_vehicle"] = (df["number_of_vehicles_involved"] > 1).astype(int)

    if all(c in df.columns for c in ["injury_claim", "property_claim", "vehicle_claim"]):
        df["component_sum"] = (
            df["injury_claim"] + df["property_claim"] + df["vehicle_claim"]
        )
        if "total_claim_amount" in df.columns:
            df["claim_discrepancy"] = abs(
                df["total_claim_amount"] - df["component_sum"]
            )

    if "witnesses" in df.columns:
        df["has_witnesses"] = (df["witnesses"] > 0).astype(int)

    if "police_report_available" in df.columns:
        df["no_police_report"] = (df["police_report_available"] == "NO").astype(int)

    logger.info("Derived features added. New shape: %s", df.shape)
    return df


def build_preprocessor(
    numerical_columns: list,
    categorical_columns: list,
    derived_numerical: list | None = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer:
    - Numerical: median imputation + standard scaling
    - Categorical: constant imputation + one-hot encoding
    """
    if derived_numerical is None:
        derived_numerical = []

    all_numerical = numerical_columns + derived_numerical

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, all_numerical),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def compute_baseline_stats(df: pd.DataFrame, numerical_columns: list) -> dict:
    """
    Compute per-feature histogram from training data for drift detection.
    Stores actual bin edges and counts (not just summary stats) so that
    the drift detector uses the real training distribution, not a synthetic one.
    """
    stats = {}
    for col in numerical_columns:
        if col not in df.columns:
            continue
        values = df[col].dropna().values
        if len(values) == 0:
            continue
        counts, bin_edges = np.histogram(values, bins=10)
        stats[col] = {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
            # Summary stats kept for reference / display
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "q25": float(np.percentile(values, 25)),
            "q50": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75)),
        }
    logger.info("Baseline histogram stats computed for %d features", len(stats))
    return stats
