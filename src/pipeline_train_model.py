import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set up logging formatting
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = "data"
MODELS_DIR = "models"

def run_model_training_pipeline():
    logging.info("--- [STAGE 08A] INITIATING ML MODEL TOURNAMENT VALIDATION ENGINE ---")
    
    input_file = os.path.join(DATA_DIR, "master_feature_table_with_hazards.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing master modeling array table: '{input_file}'")

    df = pd.read_csv(input_file)
    
    # 🧬 Phase 3: High-Alpha Structural Feature Isolation Contract
    feature_columns = [
        'rain', 'snowfall', 'wind_speed', 'temp_max', 'precipitation',
        'elevation', 'mountain_flag', 'coastal_flag', 'high_altitude_flag',
        'nearest_landslide_km', 'landslide_density_per_1000sqkm',
        'crowd_baseline', 'festival_boost', 'school_vacation_flag', 'long_weekend_flag', 'is_weekend',
        'transport_complexity_score', 'budget_stress_index', 'elevation_penalty'
    ]
    
    X = df[feature_columns].astype(float)
    y = df['overall_hazard_score'].astype(float)
    
    logging.info(f" -> Feature Contract Realigned: {X.shape[0]} historical rows x {X.shape[1]} raw features.")
    
    # Phase 4: Train/Test Random State Validation Hold-Out Split (80/20 Grid)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Check framework environment accessibility for the advanced XGBoost package wrapper
    try:
        from xgboost import XGBRegressor
        xgb_available = True
        logging.info(" 🚀 XGBoost framework library discovered and activated inside the workspace.")
    except ImportError:
        xgb_available = False
        logging.warning(" ⚠️ XGBoost package not found in this kernel partition. Defaulting to standard models.")

    model_pool = {
        "Linear Regression Baseline": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    if xgb_available:
        model_pool["XGBoost Regressor"] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

    evaluation_records = []
    trained_model_objects = {}

    logging.info(" --- EXECUTING PERFORMANCE TOURNAMENT & CROSS-VALIDATION LOOP ---")
    for label, model_obj in model_pool.items():
        logging.info(f" -> Active Candidate Training: {label}...")
        
        # Train model matrix partitions
        model_obj.fit(X_train, y_train)
        trained_model_objects[label] = model_obj
        
        # Forecast test evaluation partition splits
        preds = model_obj.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # 5-Fold Cross Validation check to verify mathematical consistency and prevent overfitting
        cv_scores = cross_val_score(model_obj, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        
        evaluation_records.append({
            "Model Candidate": label, "Test R2": r2, "Mean CV R2": cv_scores.mean(), "Test MAE": mae, "Test RMSE": rmse
        })

    df_leaderboard = pd.DataFrame(evaluation_records).sort_values(by="Test R2", ascending=False).reset_index(drop=True)
    logging.info(f"\n🏆 TOURNAMENT COMPLETED:\n{df_leaderboard.to_string(index=False)}")

    # Isolate winning champion object instance details
    champion_label = df_leaderboard.iloc[0]["Model Candidate"]
    champion_model = trained_model_objects[champion_label]
    logging.info(f"🥇 Champion Candidate Discovered: '{champion_label}'")

    # 🛡️ LEAKAGE AUDIT: Strict Chronological Time-Based Out-of-Sample Split Validation Gate
    logging.info(" --- INITIATING CHRONOLOGICAL DATA-LEAKAGE STRESS TEST ---")
    df['date'] = pd.to_datetime(df['date'])
    
    train_mask = df['date'] < '2025-01-01'
    test_mask = df['date'] >= '2025-01-01'
    
    # ✅ FIX: Synchronized to prevent array dimension mismatches during validation
    X_train_chrono = df.loc[train_mask, feature_columns].astype(float)
    y_train_chrono = df.loc[train_mask, 'overall_hazard_score'].astype(float)
    X_test_chrono = df.loc[test_mask, feature_columns].astype(float)
    y_test_chrono = df.loc[test_mask, 'overall_hazard_score'].astype(float)

    rf_chrono = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_chrono.fit(X_train_chrono, y_train_chrono)
    chrono_r2 = r2_score(y_test_chrono, rf_chrono.predict(X_test_chrono))
    
    logging.info(f" -> Chronological Split Train Matrix Window: 2021-2024 ({X_train_chrono.shape[0]} rows)")
    logging.info(f" -> Chronological Split Test Out-of-Sample Window: 2025 ({X_test_chrono.shape[0]} rows)")
    logging.info(f" -> Out-of-Sample Chronological Generalization R2: {round(chrono_r2, 4)}")

    # 💾 Phase 11: Production Serialization Interface Gateway
    model_out = os.path.join(MODELS_DIR, "travel_risk_model.pkl")
    schema_out = os.path.join(MODELS_DIR, "model_feature_schema.pkl")
    
    joblib.dump(champion_model, model_out)
    joblib.dump(feature_columns, schema_out)
    
    logging.info(" =====================================================================")
    logging.info(f" ✅ CHAMPION BINARY EXPORTED SUCCESSFULLY: '{model_out}'")
    logging.info(f" ✅ FEATURE INDEX SCHEMA CONTRACT LOCKED: '{schema_out}'")
    logging.info(" =====================================================================")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    run_model_training_pipeline()