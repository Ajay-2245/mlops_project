"""
backend/app/schemas/claim.py
─────────────────────────────
Pydantic models for request validation and response serialization.
Updated to include property_damage column.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ClaimRequest(BaseModel):
    """Input schema for a single insurance claim prediction request."""

    # Policy information
    months_as_customer: int = Field(..., ge=0, le=600)
    age: int = Field(..., ge=16, le=100)
    policy_state: str
    policy_csl: str
    policy_deductable: int = Field(..., ge=0)
    policy_annual_premium: float = Field(..., ge=0)
    umbrella_limit: int = Field(default=0)

    # Insured person
    insured_sex: Literal["MALE", "FEMALE"]
    insured_education_level: str
    insured_occupation: str
    insured_hobbies: Optional[str] = None
    insured_relationship: str
    capital_gains: float = Field(default=0, alias="capital-gains")
    capital_loss: float = Field(default=0, alias="capital-loss")

    # Incident details
    incident_type: str
    collision_type: Optional[str] = None
    incident_severity: str
    authorities_contacted: Optional[str] = None
    incident_state: str
    incident_city: str
    incident_hour_of_the_day: int = Field(..., ge=0, le=23)
    number_of_vehicles_involved: int = Field(..., ge=1, le=10)
    property_damage: Optional[Literal["YES", "NO"]] = None   # NEW
    bodily_injuries: int = Field(..., ge=0, le=10)
    witnesses: int = Field(..., ge=0, le=10)
    police_report_available: Literal["YES", "NO"]

    # Claim amounts
    total_claim_amount: float = Field(..., ge=0)
    injury_claim: float = Field(default=0, ge=0)
    property_claim: float = Field(default=0, ge=0)
    vehicle_claim: float = Field(default=0, ge=0)

    # Vehicle
    auto_make: str
    auto_year: int = Field(..., ge=1980, le=2025)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
                "property_damage": "NO",
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
        }


class PredictionResponse(BaseModel):
    claim_id: Optional[str] = None
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    threshold_used: float
    message: str


class BatchClaimRequest(BaseModel):
    claims: list[ClaimRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    fraud_count: int
    legitimate_count: int


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_stage: str
    algorithm: str
    threshold: float
    mlflow_tracking_uri: str
