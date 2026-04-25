"""
backend/tests/test_api.py
──────────────────────────
Unit and integration test suite for the FastAPI backend.
Run with: pytest backend/tests/ -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────
VALID_CLAIM = {
    "months_as_customer": 36,
    "age": 35,
    "policy_state": "OH",
    "policy_csl": "250/500",
    "policy_deductable": 500,
    "policy_annual_premium": 1200.0,
    "umbrella_limit": 0,
    "insured_sex": "MALE",
    "insured_education_level": "MD",
    "insured_occupation": "craft-repair",
    "insured_hobbies": "chess",
    "insured_relationship": "husband",
    "capital-gains": 0,
    "capital-loss": 0,
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Front Collision",
    "incident_severity": "Major Damage",
    "authorities_contacted": "Police",
    "incident_state": "OH",
    "incident_city": "Columbus",
    "incident_hour_of_the_day": 14,
    "number_of_vehicles_involved": 1,
    "bodily_injuries": 1,
    "witnesses": 0,
    "police_report_available": "YES",
    "total_claim_amount": 65000,
    "injury_claim": 10000,
    "property_claim": 5000,
    "vehicle_claim": 50000,
    "auto_make": "Saab",
    "auto_year": 2012,
}

MOCK_PREDICTION = {
    "fraud_probability": 0.72,
    "is_fraud": True,
    "risk_score": 72.0,
    "risk_tier": "HIGH",
    "threshold_used": 0.4,
}


@pytest.fixture
def client():
    """Create test client with mocked model."""
    with patch("src.models.predict.get_model") as mock_model, \
         patch("src.models.predict.get_preprocessor") as mock_preprocessor:

        mock_model.return_value = MagicMock()
        mock_preprocessor.return_value = MagicMock()

        with patch("src.models.predict.predict", return_value=MOCK_PREDICTION):
            from backend.app.main import app
            with TestClient(app) as c:
                yield c


# ── Health tests ──────────────────────────────────────────────────────────────
class TestHealth:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()

    def test_health_endpoint_exists(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_metrics_endpoint_accessible(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200


# ── Prediction tests ──────────────────────────────────────────────────────────
class TestPrediction:
    def test_valid_claim_returns_prediction(self, client):
        with patch("src.models.predict.predict", return_value=MOCK_PREDICTION):
            response = client.post("/api/v1/predict", json=VALID_CLAIM)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_score" in data
        assert "risk_tier" in data
        assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH")

    def test_prediction_with_custom_threshold(self, client):
        with patch("src.models.predict.predict", return_value=MOCK_PREDICTION):
            response = client.post("/api/v1/predict?threshold=0.8", json=VALID_CLAIM)
        assert response.status_code == 200

    def test_missing_required_field_returns_422(self, client):
        bad_claim = {k: v for k, v in VALID_CLAIM.items() if k != "age"}
        response = client.post("/api/v1/predict", json=bad_claim)
        assert response.status_code == 422

    def test_invalid_age_returns_422(self, client):
        bad_claim = {**VALID_CLAIM, "age": 150}
        response = client.post("/api/v1/predict", json=bad_claim)
        assert response.status_code == 422

    def test_invalid_sex_returns_422(self, client):
        bad_claim = {**VALID_CLAIM, "insured_sex": "OTHER"}
        response = client.post("/api/v1/predict", json=bad_claim)
        assert response.status_code == 422

    def test_batch_prediction(self, client):
        with patch("src.models.predict.predict", return_value=MOCK_PREDICTION):
            payload = {"claims": [VALID_CLAIM, VALID_CLAIM]}
            response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert "fraud_count" in data
        assert "legitimate_count" in data

    def test_batch_too_large_returns_422(self, client):
        payload = {"claims": [VALID_CLAIM] * 101}
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 422


# ── Schema tests ──────────────────────────────────────────────────────────────
class TestSchemas:
    def test_prediction_response_has_required_fields(self, client):
        with patch("src.models.predict.predict", return_value=MOCK_PREDICTION):
            response = client.post("/api/v1/predict", json=VALID_CLAIM)
        data = response.json()
        required = ["claim_id", "fraud_probability", "is_fraud", "risk_score",
                    "risk_tier", "threshold_used", "message"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_risk_score_in_range(self, client):
        with patch("src.models.predict.predict", return_value=MOCK_PREDICTION):
            response = client.post("/api/v1/predict", json=VALID_CLAIM)
        data = response.json()
        assert 0 <= data["risk_score"] <= 100

    def test_fraud_probability_in_range(self, client):
        with patch("src.models.predict.predict", return_value=MOCK_PREDICTION):
            response = client.post("/api/v1/predict", json=VALID_CLAIM)
        data = response.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0
