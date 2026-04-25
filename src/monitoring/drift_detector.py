"""
src/monitoring/drift_detector.py
─────────────────────────────────
PSI-based data drift detection using actual training histograms.

FIX: replaced np.random.normal synthetic baseline with stored histogram
     bins/counts from the real training distribution. This makes PSI
     scores deterministic and accurate regardless of distribution shape.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.monitoring.metrics import DRIFT_ALERT, FEATURE_DRIFT_SCORE

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "data/processed/baseline_stats.json"
DRIFT_THRESHOLD = 0.1   # PSI > 0.1 → moderate drift; > 0.2 → significant


def _psi_from_histogram(
    baseline_counts: np.ndarray,
    baseline_edges: np.ndarray,
    actual: np.ndarray,
) -> float:
    """
    Compute PSI between a stored baseline histogram and new live data.

    Uses the baseline's own bin edges so the bucketing is consistent
    across every call — no randomness involved.
    """
    if len(actual) < 10:
        return 0.0

    expected_pcts = baseline_counts / baseline_counts.sum()
    actual_counts = np.histogram(actual, bins=baseline_edges)[0]
    actual_pcts = actual_counts / max(actual_counts.sum(), 1)

    # Clip to avoid log(0)
    expected_pcts = np.clip(expected_pcts, 1e-6, None)
    actual_pcts = np.clip(actual_pcts, 1e-6, None)

    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi)


def load_baseline() -> Dict:
    if not BASELINE_PATH.exists():
        logger.warning("Baseline stats not found at %s", BASELINE_PATH)
        return {}
    with open(BASELINE_PATH) as f:
        return json.load(f)


def detect_drift(
    live_data: pd.DataFrame,
    features_to_check: List[str] | None = None,
) -> Dict[str, float]:
    """
    Compute PSI for each numerical feature and update Prometheus gauges.

    Args:
        live_data: DataFrame of recent incoming prediction requests.
        features_to_check: Feature names to check. Defaults to all baseline features.

    Returns:
        Dict mapping feature_name → PSI score.
    """
    baseline = load_baseline()
    if not baseline:
        return {}

    if features_to_check is None:
        features_to_check = list(baseline.keys())

    drift_scores = {}
    max_drift = 0.0

    for feature in features_to_check:
        if feature not in baseline or feature not in live_data.columns:
            continue

        stats = baseline[feature]

        # Guard: old-format baseline (summary stats only, no histogram)
        if "bin_edges" not in stats or "counts" not in stats:
            logger.warning(
                "Feature '%s' baseline has no histogram data. "
                "Re-run preprocessing to regenerate baseline_stats.json.",
                feature,
            )
            continue

        baseline_edges = np.array(stats["bin_edges"])
        baseline_counts = np.array(stats["counts"], dtype=float)
        actual = live_data[feature].dropna().values

        if len(actual) < 10:
            logger.debug(
                "Too few samples for '%s' (%d) — skipping drift check.", feature, len(actual)
            )
            continue

        psi = round(_psi_from_histogram(baseline_counts, baseline_edges, actual), 4)
        drift_scores[feature] = psi
        max_drift = max(max_drift, psi)

        FEATURE_DRIFT_SCORE.labels(feature_name=feature).set(psi)
        if psi > DRIFT_THRESHOLD:
            logger.warning("Drift detected for '%s': PSI=%.4f", feature, psi)

    DRIFT_ALERT.set(1 if max_drift > DRIFT_THRESHOLD else 0)

    logger.info(
        "Drift check complete. Features: %d | Max PSI: %.4f | Alert: %s",
        len(drift_scores), max_drift, max_drift > DRIFT_THRESHOLD,
    )
    return drift_scores
