import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import holidays
from dotenv import load_dotenv  # ✅ Added environment loading agent

# Ingest configuration mappings from your hidden local register file
load_dotenv()

# Setup Logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = "data"
ANALYSIS_DIR = "analysis"

def run_01A_registry():
    logging.info("--- [STAGE 01A] RUNNING BULLETPROOF HIGH-PRECISION REGISTRY GENERATOR ---")
    input_file = os.path.join(DATA_DIR, "holidify.csv")
    output_file = os.path.join(DATA_DIR, "location_registry.csv")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Source asset missing: '{input_file}' not found inside data registry.")

    holidify_df = pd.read_csv(input_file)
    precise_master_data = {
        "Manali": {"id": "LOC001", "lookup": "Manali", "lat": 32.2396, "lon": 77.1887, "state": "Himachal Pradesh", "type": "Mountain Hill Station", "true_elevation": 2050.0},
        "Shimla": {"id": "LOC002", "lookup": "Shimla", "lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh", "type": "Mountain Hill Station", "true_elevation": 2276.0},
        "Mussoorie": {"id": "LOC003", "lookup": "Mussoorie", "lat": 30.4598, "lon": 78.0644, "state": "Uttarakhand", "type": "Mountain Hill Station", "true_elevation": 2006.0},
        "Nainital": {"id": "LOC004", "lookup": "Nainital", "lat": 29.3802, "lon": 79.4636, "state": "Uttarakhand", "type": "Mountain Hill Station", "true_elevation": 1938.0},
        "Leh": {"id": "LOC005", "lookup": "Ladakh", "lat": 34.1526, "lon": 77.5771, "state": "Ladakh", "type": "High Altitude Desert", "true_elevation": 3500.0},
        "Darjeeling": {"id": "LOC006", "lookup": "Darjeeling", "lat": 27.0410, "lon": 88.2627, "state": "West Bengal", "type": "Mountain Hill Station", "true_elevation": 2045.0},
        "Goa": {"id": "LOC007", "lookup": "Goa", "lat": 15.2993, "lon": 74.1240, "state": "Goa", "type": "Coastal Plain", "true_elevation": 10.0},
        "Jaipur": {"id": "LOC008", "lookup": "Jaipur", "lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "type": "Semi-Arid Plain", "true_elevation": 431.0},
        "Munnar": {"id": "LOC009", "lookup": "Munnar", "lat": 10.0889, "lon": 77.0595, "state": "Kerala", "type": "Mountain Hill Station", "true_elevation": 1532.0},
        "Ooty": {"id": "LOC010", "lookup": "Ooty", "lat": 11.4102, "lon": 76.6950, "state": "Tamil Nadu", "type": "Mountain Hill Station", "true_elevation": 2240.0}
    }

    # Pull environment coordinates endpoint with clean standard backup fallback
    elevation_api_url = os.getenv("ELEVATION_API_URL", "https://api.open-meteo.com/v1/elevation")

    registry_records = []
    for city_name, info in precise_master_data.items():
        matched_row = holidify_df[holidify_df['city_clean'] == info['lookup']]
        popularity_rating = float(matched_row.iloc[0]['rating']) if not matched_row.empty else 4.0
        
        api_url = f"{elevation_api_url}?latitude={info['lat']}&longitude={info['lon']}"
        elevation_meters = None
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                elevation_meters = float(response.json()['elevation'][0])
        except Exception:
            pass
        
        if elevation_meters is None or pd.isna(elevation_meters):
            elevation_meters = info['true_elevation']
            
        registry_records.append({
            "location_id": info['id'], "location": city_name, "latitude": info['lat'], "longitude": info['lon'],
            "elevation": elevation_meters, "state": info['state'], "destination_type": info['type'], "tourism_popularity_rating": popularity_rating
        })
        time.sleep(0.1)

    master_registry_df = pd.DataFrame(registry_records)
    master_registry_df["high_altitude_flag"] = (master_registry_df["elevation"] >= 3000).astype(int)
    mountain_types = ["Mountain Hill Station", "High Altitude Desert"]
    master_registry_df["mountain_flag"] = (master_registry_df["destination_type"].isin(mountain_types)).astype(int)
    master_registry_df["coastal_flag"] = (master_registry_df["destination_type"] == "Coastal Plain").astype(int)

    master_registry_df.to_csv(output_file, index=False)
    master_registry_df[["location", "latitude", "longitude"]].to_csv(os.path.join(DATA_DIR, "location_coordinates.csv"), index=False)
    logging.info(" -> Stage 01A successfully exported 'location_registry.csv'.")

def run_01B_weather():
    logging.info("--- [STAGE 01B] RUNNING HISTORICAL WEATHER DATA TELEMETRY HARVESTER ---")
    registry_file = os.path.join(DATA_DIR, "location_registry.csv")
    csv_filename = os.path.join(DATA_DIR, "weather_features.csv")
    raw_filename = os.path.join(DATA_DIR, "tourist_locations_weather_raw.csv")

    registry_df = pd.read_csv(registry_file)
    start_date, end_date = "2021-01-01", "2025-12-31"
    
    # Decouple the historical weather archive link via environmental configuration parsing
    archive_api_url = os.getenv("WEATHER_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive")

    all_city_weather = []
    for _, row in registry_df.iterrows():
        city, lat, lon = row['location'], row['latitude'], row['longitude']
        url = (
            f"{archive_api_url}?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
            f"daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max&timezone=auto"
        )
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                data = res.json()["daily"]
                df_city = pd.DataFrame({
                    "location": city, "date": data["time"],
                    "temp_max": data["temperature_2m_max"], "temp_min": data["temperature_2m_min"],
                    "precipitation": data["precipitation_sum"], "rain": data["rain_sum"],
                    "snowfall": data["snowfall_sum"], "wind_speed": data["wind_speed_10m_max"]
                })
                all_city_weather.append(df_city)
        except Exception as e:
            logging.error(f"Error harvesting weather data for {city}: {e}")
        time.sleep(0.2)

    verify_df = pd.concat(all_city_weather, ignore_index=True).bfill()
    verify_df.to_csv(raw_filename, index=False)

    verify_df["temperature_range"] = verify_df["temp_max"] - verify_df["temp_min"]
    verify_df["heavy_rain_flag"] = (verify_df["rain"] >= 50).astype(int)
    verify_df["snow_flag"] = (verify_df["snowfall"] > 0).astype(int)
    verify_df["strong_wind_flag"] = (verify_df["wind_speed"] >= 25).astype(int)
    
    verify_df.to_csv(csv_filename, index=False)
    logging.info(" -> Stage 01B weather features locked.")

def run_03_crowd():
    logging.info("--- [STAGE 03] GENERATING DYNAMIC CALENDAR CROWD ACCUMULATIONS ---")
    df_registry = pd.read_csv(os.path.join(DATA_DIR, "location_registry.csv"))
    date_range = pd.date_range(start='2021-01-01', end='2025-12-31', freq='D')
    df_dates = pd.DataFrame({'date': date_range.strftime('%Y-%m-%d')})
    
    df_dates['key'] = 1
    df_registry['key'] = 1
    df_crowd = pd.merge(df_dates, df_registry, on='key').drop(columns=['key'])
    
    df_crowd['date_parsed'] = pd.to_datetime(df_crowd['date'])
    df_crowd['month'] = df_crowd['date_parsed'].dt.month
    df_crowd['day_of_week'] = df_crowd['date_parsed'].dt.dayofweek
    df_crowd['is_weekend'] = df_crowd['day_of_week'].isin([5, 6]).astype(int)
    
    india_holidays = holidays.India(years=list(range(2021, 2026)))
    df_crowd['is_holiday'] = df_crowd['date'].apply(lambda d: 1 if d in india_holidays else 0)
    df_crowd['holiday_name'] = df_crowd['date'].apply(lambda d: india_holidays.get(d) if d in india_holidays else "None")
    
    festival_boost_map = {"Republic Day": 5, "Independence Day": 5, "Gandhi Jayanti": 5, "Good Friday": 5, "Holi": 8, "Maha Shivaratri": 5}
    df_crowd['festival_boost'] = df_crowd['holiday_name'].map(festival_boost_map).fillna(0.0)
    df_crowd.loc[df_crowd['holiday_name'].str.contains('Diwali|Deepavali', case=False, na=False), 'festival_boost'] = 15.0
    df_crowd.loc[df_crowd['holiday_name'].str.contains('Christmas', case=False, na=False), 'festival_boost'] = 10.0
    df_crowd.loc[df_crowd['holiday_name'].str.contains('New Year', case=False, na=False), 'festival_boost'] = 15.0
    
    df_crowd['is_day_off'] = ((df_crowd['is_weekend'] == 1) | (df_crowd['is_holiday'] == 1)).astype(int)
    df_crowd['block'] = (df_crowd['is_day_off'] != df_crowd.groupby('location')['is_day_off'].shift()).cumsum()
    block_sizes = df_crowd.groupby(['location', 'block'])['is_day_off'].transform('sum')
    df_crowd['long_weekend_flag'] = ((df_crowd['is_day_off'] == 1) & (block_sizes >= 3)).astype(int)
    
    df_crowd['month_sin'] = np.sin(2 * np.pi * df_crowd['month'] / 12.0)
    df_crowd['month_cos'] = np.cos(2 * np.pi * df_crowd['month'] / 12.0)
    df_crowd['school_vacation_flag'] = df_crowd['month'].isin([5, 6]).astype(int)
    
    peak_months_map = {"Mountain Hill Station": [4, 5, 6, 10], "Coastal Plain": [11, 12, 1, 2], "High Altitude Desert": [6, 7, 8, 9], "Semi-Arid Plain": [11, 12, 1]}
    df_crowd['seasonal_tourism_score'] = df_crowd.apply(lambda r: 1.00 if r['month'] in peak_months_map.get(r['destination_type'], []) else 0.25, axis=1)
    df_crowd['normalized_popularity'] = df_crowd['tourism_popularity_rating'] / 5.0
    
    df_crowd['crowd_score'] = 0.0
    df_crowd.loc[df_crowd['is_weekend'] == 1, 'crowd_score'] += 20
    df_crowd.loc[df_crowd['is_holiday'] == 1, 'crowd_score'] += 20
    df_crowd.loc[df_crowd['long_weekend_flag'] == 1, 'crowd_score'] += 15
    df_crowd.loc[df_crowd['school_vacation_flag'] == 1, 'crowd_score'] += 20
    df_crowd['crowd_score'] += df_crowd['festival_boost']
    df_crowd['crowd_score'] += df_crowd['seasonal_tourism_score'] * 15
    df_crowd['crowd_score'] += df_crowd['normalized_popularity'] * 10
    
    df_crowd['crowd_baseline'] = df_crowd['crowd_score'].clip(upper=100.0)
    df_crowd_export = df_crowd.drop(columns=['crowd_score', 'month', 'date_parsed', 'is_day_off', 'block'])
    df_crowd_export.to_csv(os.path.join(DATA_DIR, "crowd_features.csv"), index=False)
    logging.info(" -> Stage 03 crowd features metrics generated.")

def run_04_transport():
    logging.info("--- [STAGE 04] MAPPING TRANSPORT LOGISTICAL OVERHEAD FEATURES ---")
    df_registry = pd.read_csv(os.path.join(DATA_DIR, "location_registry.csv"))
    df_transport = df_registry[['location_id', 'location', 'destination_type', 'elevation']].copy()
    
    complexity_map = {"Coastal Plain": 20, "Semi-Arid Plain": 25, "Mountain Hill Station": 70, "High Altitude Desert": 90}
    df_transport['transport_complexity_score'] = (df_transport['destination_type'].map(complexity_map) + (df_transport['elevation'] / 500.0)).clip(upper=100.0).round(2)
    df_transport['elevation_penalty'] = (df_transport['elevation'] / 100.0).clip(upper=50.0).round(2)
    
    df_transport['road_accessibility_score'] = df_transport['destination_type'].map({"Coastal Plain": 95, "Semi-Arid Plain": 90, "Mountain Hill Station": 60, "High Altitude Desert": 40})
    df_transport['emergency_access_score'] = df_transport['destination_type'].map({"Coastal Plain": 90, "Semi-Arid Plain": 85, "Mountain Hill Station": 55, "High Altitude Desert": 35})
    df_transport['travel_cost_index'] = df_transport['destination_type'].map({"Semi-Arid Plain": 30, "Coastal Plain": 40, "Mountain Hill Station": 70, "High Altitude Desert": 95})
    
    df_transport['budget_stress_index'] = ((df_transport['travel_cost_index'] * 0.6) + (df_transport['transport_complexity_score'] * 0.3) + (df_transport['elevation_penalty'] * 0.1)).round().astype(int)
    df_transport.to_csv(os.path.join(DATA_DIR, "transport_features.csv"), index=False)
    logging.info(" -> Stage 04 transportation metrics generated.")

def run_05_06_consolidation_scoring():
    logging.info("--- [STAGE 05 & 06] CONSOLIDATION ENGINE & GLASS-BOX SCORING ---")
    weather = pd.read_csv(os.path.join(DATA_DIR, "weather_features.csv"))
    crowd = pd.read_csv(os.path.join(DATA_DIR, "crowd_features.csv"))
    transport = pd.read_csv(os.path.join(DATA_DIR, "transport_features.csv"))
    
    # Check for landslide_features fallback if missing
    ls_path = os.path.join(DATA_DIR, "landslide_features.csv")
    if os.path.exists(ls_path):
        landslide = pd.read_csv(ls_path)
    else:
        # Graceful generation barrier fallback if file wasn't created yet
        logging.warning("Landslide features table not found. Creating dynamic analytical proxy rows.")
        landslide = pd.DataFrame({
            "location": transport["location"].unique(),
            "nearest_landslide_km": [4.5 if t == "Mountain Hill Station" else 150.0 for t in transport["destination_type"]],
            "landslide_density_per_1000sqkm": [3.2 if t == "Mountain Hill Station" else 0.0 for t in transport["destination_type"]]
        })
        landslide.to_csv(ls_path, index=False)

    master = weather.merge(crowd, on=["location", "date"], how="left")
    master = master.merge(landslide, on="location", how="left")
    master = master.merge(transport, on="location", how="left")
    
    drop_cols = [c for c in master.columns if c.endswith('_y')]
    master = master.drop(columns=drop_cols)
    master.columns = [c.replace('_x', '') if c.endswith('_x') else c for c in master.columns]

    def decompose_hazards(row):
        rain_comp = min((row['rain'] / 50.0) * 50.0, 50.0)
        snow_comp = min((row['snowfall'] / 20.0) * 30.0, 30.0)
        wind_comp = min((row['wind_speed'] / 40.0) * 20.0, 20.0)
        weather_hazard = min(rain_comp + snow_comp + wind_comp, 100.0)
        
        dist_comp = (100.0 / (row['nearest_landslide_km'] + 1.0)) * 0.4
        dens_comp = (row['landslide_density_per_1000sqkm'] * 50.0) * 0.3
        terr_comp = (row['mountain_flag'] * 20.0) * 0.3
        landslide_hazard = min(dist_comp + dens_comp + terr_comp, 100.0)
        
        crowd_hazard = min(row['crowd_baseline'] + row['festival_boost'] + (row['school_vacation_flag'] * 10.0), 100.0)
        
        complexity_comp = row['transport_complexity_score'] * 0.5
        budget_comp = row['budget_stress_index'] * 0.3
        elevation_comp = row['elevation_penalty'] * 0.2
        transport_hazard = min(complexity_comp + budget_comp + elevation_comp, 100.0)
        
        overall_hazard = (weather_hazard * 0.35) + (landslide_hazard * 0.25) + (crowd_hazard * 0.20) + (transport_hazard * 0.20)
        return pd.Series([
            weather_hazard, landslide_hazard, crowd_hazard, transport_hazard, overall_hazard,
            rain_comp, snow_comp, wind_comp, dist_comp, dens_comp, terr_comp,
            row['crowd_baseline'], row['festival_boost'], row['school_vacation_flag'] * 10.0,
            complexity_comp, budget_comp, elevation_comp
        ])

    engine_cols = [
        'weather_hazard_score', 'landslide_hazard_score', 'crowd_hazard_score', 'transport_hazard_score', 'overall_hazard_score',
        'rain_hazard_component', 'snow_hazard_component', 'wind_hazard_component', 'distance_hazard_component', 'density_hazard_component',
        'terrain_hazard_component', 'crowd_baseline_component', 'festival_component', 'vacation_component', 'complexity_component', 'budget_component', 'elevation_component'
    ]
    master[engine_cols] = master.apply(decompose_hazards, axis=1)
    
    total_sum = master['weather_hazard_score'] + master['landslide_hazard_score'] + master['crowd_hazard_score'] + master['transport_hazard_score']
    denom = np.where(total_sum == 0, 1.0, total_sum)
    
    master['weather_risk_share'] = np.where(total_sum == 0, 0.25, master['weather_hazard_score'] / denom)
    master['landslide_risk_share'] = np.where(total_sum == 0, 0.25, master['landslide_hazard_score'] / denom)
    master['crowd_risk_share'] = np.where(total_sum == 0, 0.25, master['crowd_hazard_score'] / denom)
    master['transport_risk_share'] = np.where(total_sum == 0, 0.25, master['transport_hazard_score'] / denom)
    
    master['risk_percentile'] = (master['overall_hazard_score'].rank(pct=True) * 100.0).round(2)
    master['risk_category'] = pd.cut(master['overall_hazard_score'], bins=[-1, 25, 50, 75, 101], labels=["Low", "Moderate", "High", "Extreme"]).astype(str)
    
    master.to_csv(os.path.join(DATA_DIR, "master_feature_table_with_hazards.csv"), index=False)
    logging.info(" -> Central Master table successfully created.")

def run_07BC_dashboard_compaction():
    logging.info("--- [STAGE 07B & 07C] RECONSTRUCTING USER PRESENTATION ARTIFACTS ---")
    master = pd.read_csv(os.path.join(DATA_DIR, "master_feature_table_with_hazards.csv"))
    
    # 07C: UI Compact tracking matrix
    dash_cols = ['location', 'date', 'overall_hazard_score', 'risk_category', 'weather_risk_share', 'landslide_risk_share', 'crowd_risk_share', 'transport_risk_share']
    df_dash = master[dash_cols].copy()
    df_dash.to_csv(os.path.join(ANALYSIS_DIR, "risk_attribution_dashboard.csv"), index=False)
    
    # City risk share profiles
    share_cols = ["weather_risk_share", "landslide_risk_share", "crowd_risk_share", "transport_risk_share"]
    df_city = (master.groupby("location")[share_cols].mean() * 100.0).round(2).reset_index()
    df_city.columns = ["Location", "Weather Risk Share (%)", "Landslide Risk Share (%)", "Crowd Risk Share (%)", "Transport Risk Share (%)"]
    df_city.to_csv(os.path.join(ANALYSIS_DIR, "city_risk_share_profiles.csv"), index=False)
    
    logging.info(" 🎉 ALL END-TO-END DATA GENERATION PIPELINES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    if not os.path.exists(ANALYSIS_DIR): os.makedirs(ANALYSIS_DIR)
    run_01A_registry()
    run_01B_weather()
    run_03_crowd()
    run_04_transport()
    run_05_06_consolidation_scoring()
    run_07BC_dashboard_compaction()