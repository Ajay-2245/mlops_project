"""
src/data/ingest.py
──────────────────
Validates that the raw Excel file has been manually placed at data/raw/.
Dataset: https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection

Place the downloaded file at: data/raw/insurance_fraud.xlsx
"""

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


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()
    raw_path = ROOT / params["data"]["raw_path"]

    if not raw_path.exists():
        logger.error(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Raw data file not found at: %s\n"
            "\n"
            "  Steps to fix:\n"
            "  1. Go to: https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection\n"
            "  2. Click Download to get the file\n"
            "  3. Rename it to: insurance_fraud.xlsx\n"
            "  4. Place it at:  data/raw/insurance_fraud.xlsx\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            raw_path,
        )
        sys.exit(1)

    # Quick sanity check — load first few rows
    try:
        df = pd.read_excel(raw_path, nrows=5)
        logger.info(
            "Raw data found at %s — %d columns detected: %s",
            raw_path,
            len(df.columns),
            list(df.columns),
        )
    except Exception as e:
        logger.error("Failed to read Excel file at %s: %s", raw_path, e)
        sys.exit(1)

    size_kb = raw_path.stat().st_size / 1024
    logger.info("Ingestion check passed. File size: %.1f KB", size_kb)


if __name__ == "__main__":
    main()
