import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from backend.api.schemas import RiskInputEnvelope, RiskInferenceResponse
from backend.api.services import gather_and_engineer_features
from dotenv import load_dotenv  # ✅ Added environment handler layer

# Initialize environment variables at engine startup
load_dotenv()

# 🔌 Synchronized import pointing to your custom storage module layout
from store_user_query.store import entry_store

router = APIRouter()

MODEL_PATH = os.path.join("models", "travel_risk_model.pkl")
SCHEMA_PATH = os.path.join("models", "model_feature_schema.pkl")

# Initialize and lock model assets at runtime startup
if os.path.exists(MODEL_PATH) and os.path.exists(SCHEMA_PATH):
    try:
        CHAMPION_MODEL = joblib.load(MODEL_PATH)
        ORDERED_SCHEMA = joblib.load(SCHEMA_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to unfreeze model artifacts: {e}")
else:
    CHAMPION_MODEL, ORDERED_SCHEMA = None, None

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "inference_engine_core", "model_loaded": CHAMPION_MODEL is not None}

@router.post("/predict", response_model=RiskInferenceResponse)
def predict_route_hazard(envelope: RiskInputEnvelope):
    if CHAMPION_MODEL is None or ORDERED_SCHEMA is None:
        raise HTTPException(status_code=503, detail="Inference engine offline: ML model binaries missing or corrupt.")

    try:
        # 1. Invoke feature extraction engine (Live Geocoding + Weather Forecast Telemetry)
        resolved_name, lat, lon, elevation, is_mountain, is_coastal, feature_dict = gather_and_engineer_features(
            envelope.location_query, envelope.target_date, envelope.simulation_override
        )
        
        # 2. Enforce exact structural sorting array order required by XGBoost contract
        input_df = pd.DataFrame([feature_dict])[ORDERED_SCHEMA].astype(float)
        
        # 3. Run true prediction using our champion model binary weights
        raw_prediction = float(CHAMPION_MODEL.predict(input_df)[0])
        score = round(max(0.0, min(100.0, raw_prediction)), 2)

        # 4. Realigned precisely to match your notebook's 4-tier stress distribution contract
        if score < 30.0: 
            tier = "Low"
        elif score < 45.0: 
            tier = "Moderate"
        elif score < 60.0: 
            tier = "Elevated"
        else: 
            tier = "Critical"

        # 5. Compile infrastructure and terrain descriptions
        if is_mountain:
            dest_type = "⛰️ Mountain Hill Station"
            dest_desc = f"High-elevation alpine terrain ({int(elevation)}m). Slopes are highly susceptible to rain-triggered mass wasting landslides and road network closures."
        elif is_coastal:
            dest_type = "🏖️ Coastal Maritime Zone"
            dest_desc = "Low-lying shoreline plain segment. Highly sensitive to monsoonal depressions, flash flooding, and severe weather infrastructure strain."
        else:
            dest_type = "🏛️ Historic Plains City"
            dest_desc = "Stable topography corridor layout. Equipped with resilient arterial transport links with significantly lower environmental sensitivities."

        # 6. Assemble the structured API output payload response
        response_data = {
            "status": "SUCCESS", 
            "resolved_name": resolved_name, 
            "latitude": lat, 
            "longitude": lon,
            "predicted_hazard_score": score, 
            "risk_category": tier, 
            "destination_type": dest_type, 
            "destination_description": dest_desc,
            "model_version": "2.0.6", 
            "forecast_date": envelope.target_date, 
            "processed_features": feature_dict
        }
        
        # 7. 🛡️ Bulletproof Storage Execution Gate
        # Intercepts successful inferences and pipes them safely via configurations
        try:
            entry_store(response_data, envelope.location_query)
        except Exception as db_err:
            # Captures and prints database/CSV faults to the console but prevents API crashes!
            print(f"⚠ Non-blocking Persistence Storage Fault Detected: {db_err}")
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))