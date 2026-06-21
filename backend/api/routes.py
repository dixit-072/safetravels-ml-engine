import os
import joblib # Using joblib instead of pickle because your training script used joblib!
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# --- IMPORTING YOUR REAL PIPELINE ---
from backend.api.services import gather_and_engineer_features
from backend.api.schemas import BudgetPredictionRequest, FinancialForecastResponse
from backend.api.budget_engine import calculate_dynamic_budget

router = APIRouter()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = "models/travel_risk_model.pkl"
SCHEMA_PATH = "models/model_feature_schema.pkl"

model = None
feature_schema = None
model_loaded = False

# Load the Champion XGBoost Model
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCHEMA_PATH):
        model = joblib.load(MODEL_PATH)
        feature_schema = joblib.load(SCHEMA_PATH)
        model_loaded = True
        logging.info("🚀 XGBoost Champion Model successfully verified in memory.")
except Exception as e:
    logging.error(f"🛑 Error loading model files: {e}")
    model_loaded = False

class RoutePredictionRequest(BaseModel):
    source_query: str = Field(default="New Delhi", example="New Delhi") # <--- NEW: The starting point!
    location_query: str = Field(..., example="Goa")                     # (This is your destination)
    target_date: str = Field(..., example="2026-06-11")
    simulation_override: str = Field(..., example="☀️ Live Production Mode")


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "inference_engine_core",
        "model_loaded": model_loaded,
        "model_version": "v3.0_XGBoost" # Upgraded version name!
    }


@router.post("/predict")
async def predict_route_risk(payload: RoutePredictionRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="ML Core engine binaries unavailable.")
    
    try:
        # 1. Use your powerful services.py pipeline to fetch REAL data
        resolved_name, lat, lon, elevation, is_mountain, is_coastal, raw_features = gather_and_engineer_features(
            source_query=payload.source_query,
            location_query=payload.location_query,
            target_date_str=payload.target_date,
            override_mode=payload.simulation_override
        )
        
        # 2. Align features array EXACTLY with how XGBoost was trained
        input_df = pd.DataFrame([raw_features])
        if feature_schema is not None:
            input_df = input_df.reindex(columns=feature_schema, fill_value=0.0)
        
        # 3. 🧠 THE BRAIN: Ask XGBoost for the true prediction
        prediction_score = model.predict(input_df)[0]
        prediction_score = max(0.0, min(100.0, float(prediction_score)))
        
        # 4. Risk Evaluation Tier Assignments
        if prediction_score < 25:
            risk_category = "Minimal Risk 🟢"
        elif prediction_score < 45:
            risk_category = "Low Risk 🍏"
        elif prediction_score < 65:
            risk_category = "Moderate Risk 🟡"
        elif prediction_score < 85:
            risk_category = "Elevated Risk 🟠"
        else:
            risk_category = "Critical Hazard 🚨"

        # Determine UI Destination Type
        if is_mountain:
            dest_type = "⛰️ Mountain/High Altitude"
        elif is_coastal:
            dest_type = "🏖️ Coastal Zone"
        else:
            dest_type = "🛣️ General Transit Route"
            
        return {
            "status": "SUCCESS",
            "source_name": raw_features.get("source_resolved_name", payload.source_query), # <--- NEW
            "route_distance_km": raw_features.get("route_distance_km", 0.0),               # <--- NEW
            "route_duration_hrs": raw_features.get("route_duration_hrs", 0.0),             # <--- NEW
            "resolved_name": resolved_name,
            "destination_type": dest_type,
            "destination_description": f"XGBoost AI Pipeline executing evaluations for {resolved_name}.",
            "latitude": lat,
            "longitude": lon,
            "predicted_hazard_score": round(float(prediction_score), 2),
            "risk_category": risk_category,
            "model_version": "v3.0_XGBoost",
            "forecast_date": payload.target_date,
            "processed_features": raw_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error logs: {str(e)}")


# =====================================================================
# 💰 PART 2: BUDGET FORECASTER ENDPOINT (UNCHANGED)
# =====================================================================
@router.post("/predict_budget", response_model=FinancialForecastResponse)
async def predict_budget(request: BudgetPredictionRequest):
    try:
        user_inputs = request.dict() 
        ml_telemetry = {
            "elevation": 2100.0,             
            "rain": 12.5,                    
            "temp_max": 28.0,                
            "predicted_hazard_score": 55.0,  
            "festival_boost": 0.0            
        }
        financial_forecast = calculate_dynamic_budget(user_inputs, ml_telemetry)
        return financial_forecast

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Budget Engine failed: {str(e)}")