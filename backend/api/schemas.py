from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

class RiskInputEnvelope(BaseModel):
    location_query: str = Field(..., description="Target Indian location query string.")
    target_date: str = Field(..., description="Trip date in YYYY-MM-DD format.")
    simulation_override: Literal[
        "☀️ Live Production Mode", 
        "🟢 Minimal Risk Tester", 
        "🍏 Low Risk Tester", 
        "🟡 Elevated Risk Tester", 
        "🟠 Severe Risk Tester", 
        "🚨 Critical Hazard Tester"
    ] = Field(..., description="Simulation override categories.")

class RiskInferenceResponse(BaseModel):
    status: str
    resolved_name: str
    latitude: float
    longitude: float
    predicted_hazard_score: float
    risk_category: str
    destination_type: str
    destination_description: str
    model_version: str
    forecast_date: str
    processed_features: Dict[str, Any]