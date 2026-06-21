from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List  # <-- Added 'List' here

# =====================================================================
# 🛡️ PART 1 SCHEMAS: RISK ENGINE (DO NOT TOUCH)
# =====================================================================
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

# =====================================================================
# 💰 PART 2 SCHEMAS: BUDGET ENGINE (NEW)
# =====================================================================
class BudgetPredictionRequest(BaseModel):
    location_query: str = Field(..., example="Manali")
    target_date: str = Field(..., example="2026-06-17")
    num_days: int = Field(default=3, ge=1, description="Total days of the trip")
    num_people: int = Field(default=2, ge=1, description="Number of travelers")
    travel_style: str = Field(default="Standard", example="Luxury")
    transport_mode: str = Field(default="Train", example="Flight")
    max_budget: float = Field(default=30000.0, ge=0.0, description="User's max budget in INR")

class BudgetBreakdown(BaseModel):
    accommodation: float
    food: float
    local_commute: float
    transport: float
    activities: float
    emergency_buffer: float

class FinancialForecastResponse(BaseModel):
    estimated_total: float
    budget_status: str
    financial_stress_score: float   
    budget_summary: str             
    applied_taxes: List[str]
    breakdown: BudgetBreakdown