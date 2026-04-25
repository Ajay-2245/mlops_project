"""
src/data/validate.py
─────────────────────
Validates the raw dataset for schema, missing values,
and class distribution. Writes a JSON report.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = ROOT / "params.yaml"

EXPECTED_COLUMNS = [
    "months_as_customer", "age", "policy_number", "policy_bind_date",
    "policy_state", "policy_csl", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "insured_zip", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "insured_relationship",
    "capital-gains", "capital-loss", "incident_date", "incident_type",
    "collision_type", "incident_severity", "authorities_contacted",
    "incident_state", "incident_city", "incident_location",
    "incident_hour_of_the_day", "number_of_vehicles_involved",
    "property_damage",                                          # NEW
    "bodily_injuries", "witnesses", "police_report_available",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "auto_make", "auto_model", "auto_year", "fraud_reported",
]


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def validate(df: pd.DataFrame, target: str) -> dict:
    report = {"passed": True, "warnings": [], "errors": []}

    # ── Schema check ────────────────────────────────────────────────────────
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        report["errors"].append(f"Missing columns: {missing_cols}")
        report["passed"] = False
    else:
        logger.info("Schema check PASSED (%d columns)", len(df.columns))

    # ── Row count ───────────────────────────────────────────────────────────
    report["row_count"] = len(df)
    if len(df) < 100:
        report["errors"].append(f"Too few rows: {len(df)}")
        report["passed"] = False
    logger.info("Row count: %d", len(df))

    # ── Missing values ──────────────────────────────────────────────────────
    null_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    high_null = {k: v for k, v in null_pct.items() if v > 30}
    report["null_percentages"] = null_pct
    if high_null:
        report["warnings"].append(f"High null columns (>30%): {high_null}")
        logger.warning("High nulls: %s", high_null)
    else:
        logger.info("Missing value check PASSED")

    # ── Target distribution ─────────────────────────────────────────────────
    if target in df.columns:
        dist = df[target].value_counts(normalize=True).round(4).to_dict()
        report["target_distribution"] = dist
        logger.info("Target distribution: %s", dist)
        fraud_rate = dist.get("Y", dist.get(1, 0))
        if fraud_rate < 0.05:
            report["warnings"].append(f"Very low fraud rate: {fraud_rate:.2%}")

    # ── Duplicate rows ──────────────────────────────────────────────────────
    dup_count = df.duplicated().sum()
    report["duplicate_rows"] = int(dup_count)
    if dup_count > 0:
        report["warnings"].append(f"{dup_count} duplicate rows found")
        logger.warning("%d duplicate rows", dup_count)

    return report


def main() -> None:
    params = load_params()
    raw_path = ROOT / params["data"]["raw_path"]
    out_path = ROOT / "data/processed/validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data from %s", raw_path)
    # Read Excel or CSV based on file extension
    if raw_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(raw_path)
    else:
        df = pd.read_csv(raw_path)

    report = validate(df, params["features"]["target_column"])

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report written to %s", out_path)

    if not report["passed"]:
        logger.error("Data validation FAILED: %s", report["errors"])
        sys.exit(1)

    logger.info("Data validation PASSED")


if __name__ == "__main__":
    main()
