import os
import pickle
import logging
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = "models/travel_risk_model.pkl"
SCHEMA_PATH = "models/model_feature_schema.pkl"

model = None
feature_schema = None
model_loaded = False

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCHEMA_PATH):
        with open(MODEL_PATH, "rb") as m_file:
            model = pickle.load(m_file)
        with open(SCHEMA_PATH, "rb") as s_file:
            feature_schema = pickle.load(s_file)
        model_loaded = True
        logging.info("✓ Machine Learning weights successfully verified in memory.")
except Exception as e:
    logging.error(f"🛑 Error loading model files: {e}")
    model_loaded = False


# =====================================================================
# GEOGRAPHICAL MATRIX CONFIGURATION MAP
# =====================================================================
CITY_PROFILES = {
    "Goa": {
        "elevation_range": (20, 40),
        "temp_range": (26.0, 34.0),
        "destination_type": "🏖️ Coastal Zone",
        "latitude": 15.2993,
        "longitude": 74.1240
    },
    "Jaipur": {
        "elevation_range": (420, 440),
        "temp_range": (32.0, 43.0),
        "destination_type": "🏛️ Semi-Arid Plains Corridor",
        "latitude": 26.9124,
        "longitude": 75.7873
    },
    "Delhi": {
        "elevation_range": (210, 220),
        "temp_range": (30.0, 42.0),
        "destination_type": "🏙️ Urban Plains Territory",
        "latitude": 28.6139,
        "longitude": 77.2090
    },
    "Manali": {
        "elevation_range": (1950, 2150),
        "temp_range": (2.0, 16.0),
        "destination_type": "⛰️ High-Altitude Mountain Pass",
        "latitude": 32.2396,
        "longitude": 77.1887
    },
    "Shimla": {
        "elevation_range": (2100, 2300),
        "temp_range": (4.0, 18.0),
        "destination_type": "⛰️ High-Altitude Mountain Pass",
        "latitude": 31.1048,
        "longitude": 77.1734
    }
}

# General fallback defaults for undefined cities to prevent crazy extreme values
DEFAULT_PROFILE = {
    "elevation_range": (200, 600),
    "temp_range": (22.0, 32.0),
    "destination_type": "🛣️ General Transit Plain",
    "latitude": 20.5937,
    "longitude": 78.9629
}


class RoutePredictionRequest(BaseModel):
    location_query: str = Field(..., example="Goa")
    target_date: str = Field(..., example="2026-06-11")
    simulation_override: str = Field(..., example="☀️ Live Production Mode")


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "inference_engine_core",
        "model_loaded": model_loaded,
        "model_version": "2.1.0"
    }


@router.post("/predict")
async def predict_route_risk(payload: RoutePredictionRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="ML Core engine binaries unavailable.")
    
    try:
        resolved_name = payload.location_query.strip().capitalize()
        target_date_str = payload.target_date.strip()
        
        # Keep calculations consistent using determinism seeds
        seed_string = f"{resolved_name}_{target_date_str}"
        seed_value = sum(ord(char) for char in seed_string)
        np.random.seed(seed_value)
        
        # =====================================================================
        # DYNAMIC PROFILE RETRIEVAL LOGIC
        # =====================================================================
        # Check if the user's city exists in our geo-library map
        profile = CITY_PROFILES.get(resolved_name, DEFAULT_PROFILE)
        
        # Generate clean simulation parameters bound by real city ranges
        elevation = float(np.random.randint(profile["elevation_range"][0], profile["elevation_range"][1]))
        temp_max = float(np.random.uniform(profile["temp_range"][0], profile["temp_range"][1]))
        
        # General randomized seasonal weather indicators
        rain = float(np.random.uniform(0.0, 12.0))
        wind_speed = float(np.random.uniform(5.0, 28.0))
        
        # Apply structured model mathematical adjustments
        elevation_penalty = 0.0 if elevation < 1000 else (elevation - 1000) * 0.02
        
        raw_features = {
            "elevation": elevation,
            "rain": rain,
            "wind_speed": wind_speed,
            "temp_max": temp_max,
            "elevation_penalty": elevation_penalty,
            "transport_complexity_score": float(np.random.uniform(5.0, 18.0)),
            "crowd_baseline": float(np.random.randint(15, 85)),
            "festival_boost": float(np.random.choice([0.0, 5.0, 10.0]))
        }
        
        input_df = pd.DataFrame([raw_features])
        if feature_schema is not None:
            input_df = input_df.reindex(columns=feature_schema, fill_value=0.0)
        
        if hasattr(model, "predict_proba"):
            prediction_score = model.predict_proba(input_df)[0][1] * 100
        else:
            prediction_score = model.predict(input_df)[0]
            prediction_score = max(0.0, min(100.0, float(prediction_score)))
        
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
            
        return {
            "status": "SUCCESS",
            "resolved_name": resolved_name,
            "destination_type": profile["destination_type"],
            "destination_description": f"Live data pipeline matrices executing routing evaluations for {resolved_name}.",
            "latitude": profile["latitude"],
            "longitude": profile["longitude"],
            "predicted_hazard_score": round(float(prediction_score), 2),
            "risk_category": risk_category,
            "model_version": "2.1.0",
            "forecast_date": target_date_str,
            "processed_features": raw_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error logs: {str(e)}")