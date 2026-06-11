import os
import requests
import pandas as pd
import numpy as np
import logging
import holidays
from datetime import datetime
from dotenv import load_dotenv  # ✅ Added environment configuration tracking handler

# Ingest configuration mappings at runtime setup initialization
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Explicit file system anchors
DATA_DIR = "data"
MASTER_DATA_PATH = os.path.join(DATA_DIR, "master_feature_table_with_hazards.csv")

def get_historical_crowd_lookup():
    """Extracts mean baseline crowd metrics per location from master data files."""
    fallback = 45.0
    lookup = {}
    if os.path.exists(MASTER_DATA_PATH):
        try:
            df = pd.read_csv(MASTER_DATA_PATH)
            if "location" in df.columns and "crowd_baseline_component" in df.columns:
                grouped = df.groupby("location")["crowd_baseline_component"].mean().to_dict()
                lookup = {str(k).strip().lower(): float(v) for k, v in grouped.items()}
        except Exception as e:
            logging.warning(f"Crowd map generation failure: {e}")
    return lookup, fallback

CROWD_LOOKUP, CROWD_FALLBACK = get_historical_crowd_lookup()

def gather_and_engineer_features(location_query: str, target_date_str: str, override_mode: str) -> tuple:
    """
    Orchestrates real-time telemetry harvesting and transforms it into the
    exact 19-column feature space contract required by our frozen XGBoost model.
    """
    # 1. Secure Coordinate Resolution via Nominatim Environmental Vectors
    geo_url = os.getenv("GEOCUT_BASE_URL", "https://nominatim.openstreetmap.org/search")
    app_agent = os.getenv("NETWORK_USER_AGENT", "SafeTravels_Backend_Engine_v2")
    
    headers = {"User-Agent": app_agent}
    try:
        geo_res = requests.get(geo_url, params={"q": location_query, "format": "json", "limit": 1}, headers=headers, timeout=5)
        if geo_res.status_code != 200 or not geo_res.json():
            raise ValueError(f"Geographic lookup tracking failed for query: '{location_query}'")
        loc_data = geo_res.json()[0]
    except Exception as e:
        raise ValueError(f"Upstream Geocoding Gateway Timeout: {str(e)}")
    
    lat, lon = float(loc_data["lat"]), float(loc_data["lon"])
    resolved_name = loc_data["display_name"].split(",")[0]
    address_text = loc_data.get("display_name", "").lower()

    if not (6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0):
        raise ValueError(f"Target location '{resolved_name}' falls outside sovereign Indian map boundaries.")

    # 2. Live Elevation Telemetry Gathering
    elevation_api_url = os.getenv("ELEVATION_API_URL", "https://api.open-meteo.com/v1/elevation")
    try:
        el_res = requests.get(f"{elevation_api_url}?latitude={lat}&longitude={lon}", timeout=5)
        elevation = float(el_res.json()["elevation"][0]) if el_res.status_code == 200 else 400.0
    except Exception:
        elevation = 400.0

    # Parse our targeted travel window date
    trip_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    today_date = datetime.now().date()
    
    # Calculate exactly how many days ahead the target trip sits from today (y)
    days_ahead = (trip_date - today_date).days
    
    # Secure bounding: clamp index between 0 and 15 (Open-Meteo's max 16-day limit)
    forecast_idx = max(0, min(15, days_ahead))

    rain, snowfall, wind_speed, temp_max, precipitation = 0.0, 0.0, 12.0, 24.0, 0.0
    
    # Live Weather Forecasting Ingestion Layer
    weather_api_url = os.getenv("WEATHER_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
    try:
        weather_url = f"{weather_api_url}?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max&timezone=auto&forecast_days=16"
        w_res = requests.get(weather_url, timeout=5)
        
        if w_res.status_code == 200:
            daily = w_res.json().get("daily", {})
            
            # Extract lists safely with clean fallbacks if lists are cut short
            rain_list = daily.get("rain_sum", [0.0])
            snow_list = daily.get("snowfall_sum", [0.0])
            wind_list = daily.get("wind_speed_10m_max", [12.0])
            temp_list = daily.get("temperature_2m_max", [24.0])
            precip_list = daily.get("precipitation_sum", [0.0])
            
            # Use our calculated forecast index securely based on current array sizes
            idx = forecast_idx if forecast_idx < len(rain_list) else 0
            
            rain = float(rain_list[idx])
            snowfall = float(snow_list[idx])
            wind_speed = float(wind_list[idx])
            temp_max = float(temp_list[idx])
            precipitation = float(precip_list[idx])
            
    except Exception as e:
        logging.warning(f"Live forecast acquisition drop: {e}. Defaulting to weather baselines.")

    # 3. Apply Quality Control Laboratory Override Grid Rules
    if override_mode == "🟢 Minimal Risk Tester":
        rain, wind_speed, elevation = 0.0, 2.0, 10.0
    elif override_mode == "🍏 Low Risk Tester":
        rain, wind_speed, elevation = 5.0, 10.0, 300.0
    elif override_mode == "🟡 Elevated Risk Tester":
        rain, wind_speed = 110.0, 45.0  
    elif override_mode == "🟠 Severe Risk Tester":
        rain, wind_speed = 160.0, 65.0
    elif override_mode == "🚨 Critical Hazard Tester":
        rain, wind_speed, elevation = 250.0, 95.0, 3000.0

    # 4. Feature Extraction & Alignment Logic Engine
    is_mountain = 1 if elevation > 1000.0 or any(k in address_text for k in ["himachal", "uttarakhand", "jammu", "kashmir", "sikkim", "hill", "manali", "shimla"]) else 0
    is_high_alt = 1 if elevation > 1800.0 else 0
    is_coastal = 1 if any(k in address_text for k in ["coast", "beach", "goa", "mumbai", "chennai", "kerala"]) else 0
    
    el_penalty = round(elevation / 100.0, 2) if is_mountain else 0.0
    complexity_score = round(70.0 + (elevation / 500.0), 2) if is_mountain else (20.0 if is_coastal else 25.0)
    travel_cost = 70 if is_mountain else (40 if is_coastal else 30)
    budget_stress = int(round((travel_cost * 0.6) + (complexity_score * 0.3) + (el_penalty * 0.1)))

    day_of_week = trip_date.weekday()
    is_weekend = 1 if day_of_week in [5, 6] else 0
    
    india_h = holidays.India(years=[trip_date.year])
    is_holiday = 1 if target_date_str in india_h else 0
    h_name = india_h.get(target_date_str, "None")
    
    festival_boost = 0.0
    if is_holiday:
        f_map = {"Republic Day": 5, "Independence Day": 5, "Gandhi Jayanti": 5, "Good Friday": 5, "Holi": 8, "Maha Shivaratri": 5}
        festival_boost = f_map.get(h_name, 0.0)
        if any(k in h_name for k in ["Diwali", "Deepavali", "New Year"]): festival_boost = 15.0
        if "Christmas" in h_name: festival_boost = 10.0

    school_vacation = 1 if trip_date.month in [5, 6] else 0
    school_vacation = 1 if trip_date.month in [5, 6] else 0
    matched_crowd_base = CROWD_LOOKUP.get(location_query.strip().lower(), CROWD_FALLBACK)

    # 5. Structure Final Input Matrix Object Map
    payload = {
        'rain': float(rain), 'snowfall': float(snowfall), 'wind_speed': float(wind_speed), 'temp_max': float(temp_max), 'precipitation': float(precipitation),
        'elevation': float(elevation), 'mountain_flag': int(is_mountain), 'coastal_flag': int(is_coastal), 'high_altitude_flag': int(is_high_alt),
        'nearest_landslide_km': 4.5 if is_mountain else 150.0, 'landslide_density_per_1000sqkm': 3.2 if is_high_alt else 0.0,
        'crowd_baseline': float(matched_crowd_base), 'festival_boost': float(festival_boost), 'school_vacation_flag': int(school_vacation),
        'long_weekend_flag': 0, 'is_weekend': int(is_weekend), 'transport_complexity_score': float(complexity_score),
        'budget_stress_index': float(budget_stress), 'elevation_penalty': float(el_penalty)
    }
    
    return resolved_name, lat, lon, elevation, is_mountain, is_coastal, payload