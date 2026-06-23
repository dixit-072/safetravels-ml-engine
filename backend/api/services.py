import os
import math
import requests
import pandas as pd
import numpy as np
import logging
import holidays
import time
from datetime import datetime
from dotenv import load_dotenv 

# Ingest configuration mappings
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = "data"
MASTER_DATA_PATH = os.path.join(DATA_DIR, "master_feature_table_with_hazards.csv")

# 1. Global Utility: Straight-line distance (The Fallback Math)
def calculate_straight_line(lat1, lon1, lat2, lon2):
    """Calculates the physical distance between two points using the Haversine formula."""
    R = 6371.0 
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

# 2. Global Utility: Crowd lookup
def get_historical_crowd_lookup():
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

def geocode_location(query: str):
    geo_url = os.getenv("GEOCUT_BASE_URL", "https://nominatim.openstreetmap.org/search")
    
    # 🛡️ THE FIX: Nominatim strictly REQUIRES an email address in the User-Agent.
    # Be sure to change 'your_email@example.com' to your real email address!
    app_agent = os.getenv("NETWORK_USER_AGENT", "SafeTravels_Backend_Engine_v3 (kaushaldixit783@gmail.com)")
    
    headers = {
        "User-Agent": app_agent,
        "Accept-Language": "en" # Forces English results to prevent random parsing crashes
    }
    
    try:
        # Increased timeout slightly to 10s to prevent accidental timeouts on slow connections
        res = requests.get(geo_url, params={"q": query, "format": "json", "limit": 1}, headers=headers, timeout=10)
        
        # 🚨 INSTANT DEBUGGING: If you are blocked, this tells you exactly why in your terminal
        if res.status_code in [403, 429]:
            print(f"🛑 API BLOCKED (HTTP {res.status_code}): Nominatim is rate-limiting you!")
            raise ValueError(f"Geocoding API blocked the request (HTTP {res.status_code}).")
            
        if res.status_code == 200:
            data_list = res.json()
            if data_list: # Check if the list actually has data inside
                data = data_list[0]
                return float(data["lat"]), float(data["lon"]), data["display_name"].split(",")[0], data.get("display_name", "").lower()
            else:
                print(f"⚠️ NOT FOUND: Nominatim database returned empty results for '{query}'")
        else:
            print(f"⚠️ SERVER ERROR: Nominatim returned status {res.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ NETWORK CRASH: {e}")
        
    # If the code reaches here, it means it failed to find a valid location
    raise ValueError(f"Could not find coordinates for: {query}")

# 3. Main Pipeline
def gather_and_engineer_features(source_query: str, location_query: str, target_date_str: str, override_mode: str) -> tuple:
    
    # 1. Geocode the Origin
    src_lat, src_lon, src_name, src_address = geocode_location(source_query)
    
    # 🛡️ THE FIX: Force Python to wait 1.2 seconds to obey Nominatim's strict rate limit
    time.sleep(1.2)
    
    # 2. Geocode the Destination
    lat, lon, resolved_name, address_text = geocode_location(location_query)

    # =========================================================================
    # 1. LIVE ELEVATION TELEMETRY (Checking BOTH Source and Destination)
    # =========================================================================
    elevation_api_url = os.getenv("ELEVATION_API_URL", "https://api.open-meteo.com/v1/elevation")
    try:
        # Get Destination Elevation
        el_res_dest = requests.get(f"{elevation_api_url}?latitude={lat}&longitude={lon}", timeout=5)
        dest_elevation = float(el_res_dest.json()["elevation"][0]) if el_res_dest.status_code == 200 else 400.0
        
        # Get Source Elevation
        el_res_src = requests.get(f"{elevation_api_url}?latitude={src_lat}&longitude={src_lon}", timeout=5)
        src_elevation = float(el_res_src.json()["elevation"][0]) if el_res_src.status_code == 200 else 400.0
        
        # Find the highest point of the trip
        max_trip_elevation = max(dest_elevation, src_elevation)
        
        # We keep the destination elevation to display on the UI
        elevation = dest_elevation 
        
    except Exception:
        max_trip_elevation = 400.0
        elevation = 400.0
        logging.warning("Elevation API failed. Defaulting to 400m.")

    # ⛰️ Establish the dynamic multiplier based on the HIGHEST point
    if max_trip_elevation > 1800.0:
        terrain_multiplier = 1.4  # Adds 40% extra time for high-altitude traffic/breaks
    elif max_trip_elevation > 1000.0:
        terrain_multiplier = 1.2  # Adds 20% extra time for hilly terrain
    else:
        terrain_multiplier = 1.0  # Plains / Highways

    # =========================================================================
    # 2. 🚗 OPENROUTE SERVICE INTEGRATION (Now Topography-Aware!)
    # =========================================================================
    distance_km = 0.0
    duration_hrs = 0.0
    try:
        api_key = os.getenv("ORS_API_KEY") 
        if not api_key:
            raise ValueError("API Key is missing from .env")

        headers = {
            'Authorization': api_key,
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8'
        }
        
        ors_url = f"https://api.openrouteservice.org/v2/directions/driving-car?start={src_lon},{src_lat}&end={lon},{lat}"
        route_res = requests.get(ors_url, headers=headers, timeout=25)
        
        if route_res.status_code == 200:
            features = route_res.json()["features"][0]
            summary = features["properties"]["summary"]
            
            distance_km = round(summary["distance"] / 1000.0, 1)
            base_duration = summary["duration"] / 3600.0
            if distance_km > 600 and terrain_multiplier > 1.0:
                # Keep only 20% of the penalty for long cross-country drives
                terrain_multiplier = 1.0 + ((terrain_multiplier - 1.0) * 0.2)

            duration_hrs = round(base_duration * terrain_multiplier, 1)
            
            logging.info(f"✅ Route mapped: {distance_km}km, {duration_hrs}hrs (Multiplier: {terrain_multiplier}x)")
        else:
            raise ValueError(f"API Error {route_res.status_code}: {route_res.text}")
            
    except Exception as e:
        logging.warning(f"Routing failed ({e}). Engaging Fallback.")
        distance_km = calculate_straight_line(src_lat, src_lon, lat, lon)
        # Apply the same dynamic multiplier to the fallback math!
        duration_hrs = round((distance_km / 45.0) * terrain_multiplier, 1)

    # =========================================================================
    # 3. LIVE WEATHER FORECASTING INGESTION LAYER
    # =========================================================================
    trip_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    today_date = datetime.now().date()
    days_ahead = (trip_date - today_date).days
    forecast_idx = max(0, min(15, days_ahead))

    rain, snowfall, wind_speed, temp_max, temp_min, precipitation = 0.0, 0.0, 12.0, 24.0, 15.0, 0.0
    weather_api_url = os.getenv("WEATHER_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
    try:
        weather_url = f"{weather_api_url}?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max&timezone=auto&forecast_days=16"
        w_res = requests.get(weather_url, timeout=5)
        
        if w_res.status_code == 200:
            daily = w_res.json().get("daily", {})
            rain_list = daily.get("rain_sum", [0.0])
            snow_list = daily.get("snowfall_sum", [0.0])
            wind_list = daily.get("wind_speed_10m_max", [12.0])
            temp_max_list = daily.get("temperature_2m_max", [24.0])
            temp_min_list = daily.get("temperature_2m_min", [15.0]) 
            precip_list = daily.get("precipitation_sum", [0.0])
            
            idx = forecast_idx if forecast_idx < len(rain_list) else 0
            
            rain = float(rain_list[idx])
            snowfall = float(snow_list[idx])
            wind_speed = float(wind_list[idx])
            temp_max = float(temp_max_list[idx])
            temp_min = float(temp_min_list[idx]) 
            precipitation = float(precip_list[idx])
    except Exception as e:
        logging.warning(f"Live forecast drop: {e}.")

    # =========================================================================
    # 4. FEATURE EXTRACTION & ALIGNMENT LOGIC ENGINE
    # =========================================================================
    # We update the is_mountain flag to rely heavily on our elevation data
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
    matched_crowd_base = CROWD_LOOKUP.get(location_query.strip().lower(), CROWD_FALLBACK)

    # =========================================================================
    # 5. STRUCTURE FINAL INPUT MATRIX OBJECT MAP
    # =========================================================================
    payload = {
        'rain': float(rain), 
        'snowfall': float(snowfall), 
        'wind_speed': float(wind_speed), 
        'temp_max': float(temp_max), 
        'temp_min': float(temp_min), 
        'precipitation': float(precipitation),
        'elevation': float(elevation), 
        'mountain_flag': int(is_mountain), 
        'coastal_flag': int(is_coastal), 
        'high_altitude_flag': int(is_high_alt),
        'nearest_landslide_km': 4.5 if is_mountain else 150.0, 
        'landslide_density_per_1000sqkm': 3.2 if is_high_alt else 0.0,
        'crowd_baseline': float(matched_crowd_base), 
        'festival_boost': float(festival_boost), 
        'school_vacation_flag': int(school_vacation),
        'long_weekend_flag': 0, 
        'is_weekend': int(is_weekend), 
        'transport_complexity_score': float(complexity_score),
        'budget_stress_index': float(budget_stress), 
        'elevation_penalty': float(el_penalty),
        
        # 🆕 ROUTING METRICS FOR THE FRONTEND
        'route_distance_km': distance_km,
        'route_duration_hrs': duration_hrs,
        'source_resolved_name': src_name,
        'source_lat': src_lat,
        'source_lon': src_lon 
    }
    
    return resolved_name, lat, lon, elevation, is_mountain, is_coastal, payload