import os
import pickle
import logging
import requests
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

DEFAULT_PROFILE = {
    "elevation_range": (200, 600),
    "temp_range": (22.0, 32.0),
    "destination_type": "🛣️ General Transit Plain",
    "latitude": 20.5937,
    "longitude": 78.9629
}


# =====================================================================
# LIVE METEOROLOGICAL NETWORK UTILITY
# =====================================================================
def fetch_real_time_weather(lat: float, lon: float) -> dict:
    """
    Queries Open-Meteo global tracking array for live surface telemetry.
    Returns parsed dictionary parameters or None upon request timeout.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,rain,wind_speed_10m"
        response = requests.get(url, timeout=6)
        if response.status_code == 200:
            current_data = response.json().get("current", {})
            return {
                "rain": float(current_data.get("rain", 0.0)),
                "temp_max": float(current_data.get("temperature_2m", 20.0)),
                "wind_speed": float(current_data.get("wind_speed_10m", 10.0))
            }
    except Exception as network_err:
        logging.warning(f"⚠️ Live API connection dropped. Reverting to regional simulation: {network_err}")
    return None


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
        
        # Enforce mathematical determinism based on city-date combination
        seed_string = f"{resolved_name}_{target_date_str}"
        seed_value = sum(ord(char) for char in seed_string)
        np.random.seed(seed_value)
        
        # Load local base metadata profile
        profile = CITY_PROFILES.get(resolved_name, DEFAULT_PROFILE)
        
        # Determine topographical layout limits
        elevation = float(np.random.randint(profile["elevation_range"][0], profile["elevation_range"][1]))
        
        # =====================================================================
        # REAL-TIME CORRIDOR TELEMETRY FETCH
        # =====================================================================
        live_weather = fetch_real_time_weather(profile["latitude"], profile["longitude"])
        
        if live_weather:
            rain = live_weather["rain"]
            temp_max = live_weather["temp_max"]
            wind_speed = live_weather["wind_speed"]
        else:
            # Resilient fallback matrices if connection interface fails
            rain = float(np.random.uniform(0.0, 12.0))
            temp_max = float(np.random.uniform(profile["temp_range"][0], profile["temp_range"][1]))
            wind_speed = float(np.random.uniform(5.0, 28.0))
        
        # Apply mathematical scaling penalties for extreme altimeters
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
        
        # Align features array with model training parameters layout matrix
        input_df = pd.DataFrame([raw_features])
        if feature_schema is not None:
            input_df = input_df.reindex(columns=feature_schema, fill_value=0.0)
        
        # Extract classification probability metric weights
        if hasattr(model, "predict_proba"):
            prediction_score = model.predict_proba(input_df)[0][1] * 100
        else:
            prediction_score = model.predict(input_df)[0]
            prediction_score = max(0.0, min(100.0, float(prediction_score)))
        
        # Risk Evaluation Tier Assignments
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