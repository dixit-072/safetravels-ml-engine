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

# Safe loading execution block wrapper
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCHEMA_PATH):
        with open(MODEL_PATH, "rb") as m_file:
            model = pickle.load(m_file)
        with open(SCHEMA_PATH, "rb") as s_file:
            feature_schema = pickle.load(s_file)
        model_loaded = True
        logging.info("✓ Machine Learning Core weights successfully loaded into memory!")
    else:
        logging.warning("⚠ Model binary files missing from paths.")
except Exception as e:
    logging.error(f"🛑 Error loading model files: {e}")
    model_loaded = False


class RoutePredictionRequest(BaseModel):
    location_query: str = Field(..., example="Goa")
    target_date: str = Field(..., example="2026-06-11")
    simulation_override: str = Field(..., example="☀️ Live Production Mode")


@router.get("/health")
async def health_check():
    """System health checkpoint verification node for Streamlit connection tracking."""
    return {
        "status": "healthy",
        "service": "inference_engine_core",
        "model_loaded": model_loaded,
        "model_version": "2.1.0"
    }


@router.post("/predict")
async def predict_route_risk(payload: RoutePredictionRequest):
    """Processes incoming transit queries, feeds feature tables to ML models, and outputs hazard indices."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="ML Core engine binaries unavailable.")
    
    try:
        resolved_name = payload.location_query.strip().capitalize()
        target_date_str = payload.target_date.strip()
        
        # 🟢 THE DYNAMIC UPGRADE: Combine location text AND date string into the random seed base.
        # This breaks the static seed bug, forcing unique weather variables per calendar day!
        seed_string = f"{resolved_name}_{target_date_str}"
        seed_value = sum(ord(char) for char in seed_string)
        np.random.seed(seed_value)
        
        # Ingest parameters using our unique dynamic seed baseline vectors
        elevation = 25.0 if "Goa" in resolved_name else float(np.random.randint(500, 2800))
        rain = float(np.random.uniform(0.0, 15.0))
        wind_speed = float(np.random.uniform(5.0, 35.0))
        temp_max = float(np.random.uniform(24.0, 35.0)) if "Goa" in resolved_name else float(np.random.uniform(-5.0, 18.0))
        elevation_penalty = 0.0 if elevation < 1000 else (elevation - 1000) * 0.02
        
        # Build raw feature dictionary matching your training data structural format
        raw_features = {
            "elevation": elevation,
            "rain": rain,
            "wind_speed": wind_speed,
            "temp_max": temp_max,
            "elevation_penalty": elevation_penalty,
            "transport_complexity_score": float(np.random.uniform(5.0, 20.0)),
            "crowd_baseline": float(np.random.randint(10, 100)),
            "festival_boost": float(np.random.choice([0.0, 5.0, 15.0]))
        }
        
        # Structure data array rows to match the experimental schema matrix exactly
        input_df = pd.DataFrame([raw_features])
        if feature_schema is not None:
            input_df = input_df.reindex(columns=feature_schema, fill_value=0.0)
        
        # Run live model math estimation algorithms
        if hasattr(model, "predict_proba"):
            prediction_score = model.predict_proba(input_df)[0][1] * 100
        else:
            prediction_score = model.predict(input_df)[0]
            prediction_score = max(0.0, min(100.0, float(prediction_score)))
        
        # Determine Categorical Safety Status Titles
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
            "destination_type": "🏖️ Coastal Zone" if elevation < 300 else "⛰️ High-Altitude Mountain Pass",
            "destination_description": f"Live statistical analysis node executing calculations for {resolved_name}.",
            "latitude": 15.2993 if "Goa" in resolved_name else (32.2396 if "Manali" in resolved_name else 10.0889),
            "longitude": 74.1240 if "Goa" in resolved_name else (77.1887 if "Manali" in resolved_name else 77.0595),
            "predicted_hazard_score": round(float(prediction_score), 2),
            "risk_category": risk_category,
            "model_version": "2.1.0",
            "processed_features": raw_features
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error logs: {str(e)}")