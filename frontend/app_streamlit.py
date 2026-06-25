import streamlit as st
import requests
import pandas as pd
import logging
import os
import numpy as np
import gspread
import pydeck as pdk
import json
import base64
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from google.oauth2.service_account import Credentials
from datetime import datetime
from dotenv import find_dotenv, load_dotenv

from summary import generate_semantic_narrative  
from budget_ui import render_budget_tab          

# This MUST be the first Streamlit command
st.set_page_config(page_title="SafeTravels | Smart Route Planner", page_icon="🚗", layout="wide")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv() # Keeps other local .env stuff working (like your database)

#  THE SECURE STREAMLIT SECRETS WAY
try:
    # Streamlit will automatically find .streamlit/secrets.toml
    ORS_API_KEY = st.secrets["ORS_API_KEY"]
except Exception:
    ORS_API_KEY = None

FASTAPI_URL = "https://safetravels-ml-engine.onrender.com/predict"  
HEALTH_URL = "https://safetravels-ml-engine.onrender.com/health"
MAX_HISTORY = 20

# --- 2. SETUP GOOGLE SHEETS LOCATIONS ---
try:
    SPREADSHEET_LINK = st.secrets.get("spreadsheet", "https://docs.google.com/spreadsheets/d/1KFiu3DzOSlDGEsh4vCYnfdUd6Po-0qL3CttLbe7wm1Q/edit")
    if st.secrets.get("SPREADSHEET_NAME"):
        SPREADSHEET_NAME = st.secrets.get("SPREADSHEET_NAME")
        WORKSHEET_NAME = st.secrets.get("GOOGLE_SHEET_TAB", "prediction_responses")
    else:
        SPREADSHEET_NAME = "SafeTravels_Cloud_Logs"
        WORKSHEET_NAME = st.secrets.get("worksheet", "prediction_responses")
except Exception:
    SPREADSHEET_LINK = "https://docs.google.com/spreadsheets/d/1KFiu3DzOSlDGEsh4vCYnfdUd6Po-0qL3CttLbe7wm1Q/edit"
    SPREADSHEET_NAME = "SafeTravels_Cloud_Logs"
    WORKSHEET_NAME = "prediction_responses"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.sidebar.title("🎮 Menu Options")
app_view = st.sidebar.radio("Switch Dashboard View:", ["🔮 Route Risk Checker", "📊 Travel Data Analytics"])
st.sidebar.markdown("---")

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource(ttl=600) 
def get_gspread_client(): 
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        
        if os.path.exists("google_creds.json"):
            with open("google_creds.json") as f:
                creds_dict = json.load(f)
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return gspread.authorize(credentials)
            
        elif "GCP_CREDS_B64" in st.secrets:
            b64_token = st.secrets["GCP_CREDS_B64"]
            clean_token = str(b64_token).strip().replace('"', '').replace("'", "")
            decoded_json = base64.b64decode(clean_token).decode("utf-8")
            creds_dict = json.loads(decoded_json)
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return gspread.authorize(credentials)
            
        else:
            logging.error("No credentials found in local file or st.secrets")
            return None
            
    except Exception as e:
        logging.error(f"Google Auth Failed: {e}")
        return None
    

def get_mathematical_ground_truth(telemetry):
    def get_val(key, default=0.0):
        return float(telemetry.get(key, default))

    rain_comp = min((get_val('rain') / 50.0) * 50.0, 50.0)
    snow_comp = min((get_val('snowfall') / 20.0) * 30.0, 30.0)
    wind_comp = min((get_val('wind_speed') / 40.0) * 20.0, 20.0)
    weather_hazard = min(rain_comp + snow_comp + wind_comp, 100.0)
    
    dist_factor = 100.0 / (get_val('nearest_landslide_km') + 1.0)
    dens_factor = get_val('landslide_density_per_1000sqkm') * 50.0
    terr_factor = get_val('mountain_flag') * 20.0
    landslide_hazard = min((dist_factor * 0.4) + (dens_factor * 0.3) + (terr_factor * 0.3), 100.0)
    
    crowd_hazard = min(get_val('crowd_baseline') + get_val('festival_boost') + (get_val('school_vacation_flag') * 10.0), 100.0)
    
    transport_hazard = min((get_val('transport_complexity_score') * 0.5) + (get_val('budget_stress_index') * 0.3) + (get_val('elevation_penalty') * 0.2), 100.0)
    
    return round((weather_hazard * 0.35) + (landslide_hazard * 0.25) + (crowd_hazard * 0.20) + (transport_hazard * 0.20), 2)

def fetch_budget_cloud_logs():
    client = get_gspread_client() 
    if not client: return pd.DataFrame()
    try:
        sheet = client.open_by_url(SPREADSHEET_LINK)
        worksheet = sheet.worksheet("budget_forecasts")
        records = worksheet.get_all_records()
        df = pd.DataFrame(records).dropna(how='all')
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        logging.error(f"Error reading budget tab: {e}")
        return pd.DataFrame()

def fetch_cloud_prediction_logs():
    client = get_gspread_client() 
    if not client: return pd.DataFrame()
    try:
        sheet = client.open_by_url(SPREADSHEET_LINK)
        worksheet = sheet.worksheet(WORKSHEET_NAME)
        records = worksheet.get_all_records()
        df = pd.DataFrame(records).dropna(how='all')
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        logging.error(f"Error reading prediction tab: {e}")
        return pd.DataFrame()

def write_cloud_prediction_log(row_data: list):
    client = get_gspread_client()
    if not client: return False
    try:
        try:
            sheet = client.open_by_url(SPREADSHEET_LINK).worksheet(WORKSHEET_NAME)
        except Exception:
            sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
        
        synchronized_payload = [str(x) if not isinstance(x, (int, float, dict)) else (json.dumps(x) if isinstance(x, dict) else float(x)) for x in row_data]
        synchronized_payload.append("SUCCESS")
        sheet.append_row(synchronized_payload)
        return True
    except Exception as e:
        return False

def write_budget_cloud_log(row_data: list):
    client = get_gspread_client()
    if not client: return False
    try:
        try:
            sheet = client.open_by_url(SPREADSHEET_LINK)
        except Exception:
            sheet = client.open(SPREADSHEET_NAME)
        
        BUDGET_TAB_NAME = "budget_forecasts"
        try:
            worksheet = sheet.worksheet(BUDGET_TAB_NAME)
        except Exception:
            worksheet = sheet.add_worksheet(title=BUDGET_TAB_NAME, rows="1000", cols="15")
            worksheet.append_row(["Timestamp", "Location", "Travel Date", "Days", "People", "Style", "Transport", "Max Budget", "Estimated Total", "Stress Score (%)", "Status"])
        
        worksheet.append_row(row_data)
        logging.info("✓ Budget forecast logged successfully to Google Sheets.")
        return True
    except Exception as e:
        logging.error(f"🛑 Failed to append budget log to Google Sheets: {e}")
        return False

# 🌟 NEW CLEAN RAINFALL TRANSLATOR (Guaranteed to fit in UI box)
def translate_rainfall(mm_value):
    try:
        mm = float(mm_value)
    except (ValueError, TypeError):
        return "Unknown ☁️", "N/A"
        
    # Standard IMD (India Meteorological Department) classifications
    if mm < 0.1:
        return "Clear ☀️", "Trace"
    elif mm < 2.5:
        return "Drizzle 🌦️", "Very Light"
    elif mm < 7.6:
        return "Showers 🌧️", "Light Rain"      # 🔹 Fixed Light/Light repetition
    elif mm < 35.6:
        return "Steady 🌧️", "Moderate"        # 🔹 Fixed Moderate/Moderate repetition
    elif mm < 64.5:
        return "Downpour ⛈️", "Rather Heavy"
    elif mm < 115.6:
        return "Intense ⛈️", "Heavy Rain"
    elif mm < 204.4:
        return "Extreme 🌊", "Very Heavy"
    else:
        return "Hazard 🚨", "Cloudburst"

@st.cache_data
def load_cached_destinations():
    default_cities = ["Manali", "Shimla", "Mussoorie", "Nainital", "Leh", "Darjeeling", "Goa", "Jaipur", "Munnar", "Ooty"]
    try:
        csv_path = "data/master_feature_table_with_hazards.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return sorted(df["location"].unique().tolist())
    except Exception: pass
    return default_cities

def check_backend_online(url):
    try:
        res = requests.get(url, timeout=2)
        return res.status_code == 200
    except Exception:
        return False

CORE_DESTINATIONS = load_cached_destinations()
BACKEND_ONLINE = check_backend_online(HEALTH_URL)

# --- 4. MAIN DASHBOARD UI ---
if app_view == "🔮 Route Risk Checker":
    st.title("🚗 SafeTravels AI Engine")
    st.markdown("### Smart Weather, Terrain & Financial Analysis System")
    st.write("---")

    st.header("🧳 Plan Your Trip")
    
    with st.container(border=True):
        c_title, c_toggle = st.columns([4, 1])
        with c_title:
            st.markdown("#### 🌍 Route Selection")
        with c_toggle:
            st.write("") 
            manual_search = st.toggle("✏️ Manual Entry", value=False, help="Turn on to type a destination not in the default list")

        col_src, col_dest, col_date = st.columns(3)
        
        with col_src:
            source_query = st.text_input("🛫 Origin", value="New Delhi")
            
        with col_dest:
            if manual_search:
                user_query = st.text_input("🛬 Destination", value="Manali", placeholder="Type any city...")
            else:
                user_query = st.selectbox("🛬 Destination", options=CORE_DESTINATIONS, index=1)
                
        with col_date:
            travel_date = st.date_input("📅 Date", min_value=datetime.now().date(), value=datetime.now().date())
            
    final_query_for_state = "Manali" if user_query.strip().upper().startswith("TEST_") else user_query
    st.session_state.shared_query = final_query_for_state
    st.session_state.shared_date = travel_date.strftime("%Y-%m-%d")
    st.session_state.shared_source = source_query 

    st.write("---")

    st.markdown("""
        <style>
            div[data-testid="stTabs"] button { flex: 1; }
        </style>
    """, unsafe_allow_html=True)

    tab_risk, tab_budget = st.tabs(["🛡️ Route Safety & Risk Profile", "💰 Intelligent Budget Forecaster"])

    with tab_risk:
        app_mode = "☀️ Live Production Mode"
        
        st.sidebar.markdown("### 📡 Connection Status")
        if BACKEND_ONLINE:
            st.sidebar.success("🟢 System Online")
        else:
            st.sidebar.error("🔴 System Offline")

        st.write("") 
        spacer_left, btn_col, spacer_right = st.columns([1, 2, 1])
        with btn_col:
            trigger_inference = st.button("Check Route Safety Profile", type="primary", use_container_width=True)

        st.write("") 
        col_inputs, col_advisory = st.columns([1.2, 1], gap="large")

        if trigger_inference:
            if not BACKEND_ONLINE:
                st.error("🛑 Cannot Complete Request: The calculation backend is currently offline.")
            else:
                user_input_clean = user_query.strip().upper()
                is_test_mode = user_input_clean.startswith("TEST_")
                final_query = "Manali" if is_test_mode else user_query
                final_mode = app_mode

                if is_test_mode:
                    try:
                        forced_score = float(user_input_clean.split("_")[1])
                        forced_score = max(0.0, min(100.0, forced_score))
                    except (IndexError, ValueError):
                        forced_score = 50.0

                    if forced_score < 25: final_mode = "🟢 Minimal Risk Tester"
                    elif forced_score < 45: final_mode = "🍏 Low Risk Tester"
                    elif forced_score < 65: final_mode = "🟡 Elevated Risk Tester"
                    elif forced_score < 85: final_mode = "🟠 Severe Risk Tester"
                    else: final_mode = "🚨 Critical Hazard Tester"

                request_body = {
                    "source_query": source_query.strip().title(), 
                    "location_query": final_query,
                    "target_date": travel_date.strftime("%Y-%m-%d"),
                    "simulation_override": final_mode
                }
                
                with st.spinner("Analyzing live weather and route variables..."):
                    try:
                        response = requests.post(FASTAPI_URL, json=request_body, timeout=45)
                        if response.status_code == 200:
                            res_data = response.json()
                            st.session_state["saved_risk_report"] = res_data
                            st.session_state["saved_is_test"] = is_test_mode
                            st.session_state["saved_forced_score"] = forced_score if is_test_mode else None
                            st.session_state["saved_final_query"] = final_query 
                            st.session_state["log_this_run"] = True
                        else:
                            st.error(f"🛑 Error: {response.json().get('detail', 'Could not fetch safety report.')}")
                    except Exception as e:
                        st.error(f"🛑 Connection Timeout: Details: {e}")

        # --- 5. RENDER ROUTE RESULTS ---
        if "saved_risk_report" in st.session_state:
            res_data = st.session_state["saved_risk_report"]
            is_test_mode = st.session_state["saved_is_test"]
            
            if res_data and res_data.get("status") == "SUCCESS":
                
                if is_test_mode:
                    forced_score = st.session_state["saved_forced_score"]
                    res_data["predicted_hazard_score"] = forced_score
                    if forced_score < 25:
                        res_data["risk_category"] = "Minimal"
                        res_data["processed_features"].update({"rain": 0.0, "wind_speed": 5.0, "temp_max": 22.0})
                    elif forced_score < 45:
                        res_data["risk_category"] = "Low"
                        res_data["processed_features"].update({"rain": 2.4, "wind_speed": 12.0, "temp_max": 18.5})
                    elif forced_score < 65:
                        res_data["risk_category"] = "Moderate"
                        res_data["processed_features"].update({"rain": 14.2, "wind_speed": 22.4, "temp_max": 14.0})
                    elif forced_score < 85:
                        res_data["risk_category"] = "Elevated"
                        res_data["processed_features"].update({"rain": 38.5, "wind_speed": 34.1, "temp_max": 9.5})
                    else:
                        res_data["risk_category"] = "Critical"
                        res_data["processed_features"].update({"rain": 85.4, "wind_speed": 48.2, "temp_max": 4.1})

                telemetry = res_data.get("processed_features", {})
                st.session_state["latest_telemetry"] = telemetry  
                score = res_data.get("predicted_hazard_score")
                tier = res_data.get("risk_category")
                
                normalized_features = {
                    "rain": float(telemetry.get('rain', 0.0)),
                    "wind_speed": float(telemetry.get('wind_speed', 0.0)),
                    "temp_max": float(telemetry.get('temp_max', 0.0)),
                    "temp_min": float(telemetry.get('temp_min', 0.0)),
                    "elevation": float(telemetry.get('elevation', 0.0)),
                    "resolved_name": res_data.get("resolved_name", "Specified Destination"),
                    "risk_score": float(score)
                }
                
                with col_inputs:
                    st.subheader("📊 Current Live Conditions")
                    st.success(f"📍 Location Confirmed: **{res_data.get('resolved_name')}**" if not is_test_mode else "📍 Location Confirmed: Manali (SIMULATED FORCING)")
                    
                    # 🗺️ OPEN ROUTES SERVICE (ORS) API INTEGRATION
                    dest_lat = float(res_data.get("latitude", 0.0))
                    dest_lon = float(res_data.get("longitude", 0.0))
                    
                    src_lat = float(telemetry.get("source_lat", 28.6139)) 
                    src_lon = float(telemetry.get("source_lon", 77.2090))
                    
                    adjusted_dist = 0.0
                    adjusted_dur = 0.0
                    route_coordinates = [] # 🌟 NEW: This will hold our winding road map data!
                    
                    if not ORS_API_KEY:
                        st.warning("⚠️ ERROR: ORS_API_KEY is not loading from secrets!")
                    elif src_lat == 0.0 or dest_lat == 0.0:
                        st.warning("⚠️ ERROR: Missing GPS coordinates from backend.")
                    else:
                        try:
                            ors_url = f"https://api.openrouteservice.org/v2/directions/driving-car?api_key={ORS_API_KEY}&start={src_lon},{src_lat}&end={dest_lon},{dest_lat}"
                            route_res = requests.get(ors_url, timeout=5)
                            
                            if route_res.status_code == 200:
                                route_data = route_res.json()
                                if "features" in route_data and len(route_data["features"]) > 0:
                                    summary = route_data["features"][0]["properties"]["summary"]
                                    adjusted_dist = round(summary["distance"] / 1000, 1) 
                                    adjusted_dur = round(summary["duration"] / 3600, 1)
                                    
                                    # 🌟 NEW: Extract the exact road geometry!
                                    route_coordinates = route_data["features"][0]["geometry"]["coordinates"]
                            else:
                                st.warning(f"⚠️ ORS API Failed (Code {route_res.status_code}): {route_res.text}")
                        except Exception as e:
                            st.warning(f"⚠️ API Request Crash: {e}")
                            
                    # Fallback failsafe
                    if adjusted_dist == 0.0:
                        raw_dist = float(res_data.get('route_distance_km', 0))
                        raw_dur = float(res_data.get('route_duration_hrs', 0))
                        dist_mult = 1.8 if "Mountain" in res_data.get("destination_type", "") else 1.2
                        adjusted_dist = round(raw_dist * dist_mult, 1)
                        adjusted_dur = round(raw_dur * dist_mult, 1)

                    st.markdown(f"🛣️ **Route Planner:** {res_data.get('source_name', 'Origin')} ➔ {res_data.get('resolved_name')}")
                    st.markdown(f"📏 **Driving Distance:** {adjusted_dist} km (Approx {adjusted_dur} hours)")
                    st.markdown(f"#### Terrain Profile: **{res_data.get('destination_type')}**")
                    st.caption(res_data.get("destination_description"))
                    st.write("")
                        
                    m_r1_c1, m_r1_col2 = st.columns(2)
                    with m_r1_c1:
                        st.metric(label="⛰️ Altitude Height", value=f"{normalized_features.get('elevation', 0):,.0f} m")
                    with m_r1_col2:
                        t_max = normalized_features.get("temp_max", 0.0)
                        t_min = normalized_features.get("temp_min", 0.0)
                        sub_col1, sub_col2 = st.columns(2)
                        with sub_col1:
                            st.metric(label="🔴 High Temp", value=f"{t_max:.0f}°C")
                        with sub_col2:
                            st.metric(label="🔵 Low Temp", value=f"{t_min:.0f}°C")

                    st.write("")
                    m_r2_c1, m_r2_col2 = st.columns(2)
                    
                    # 🌟 THE RAINFALL FIX IS APPLIED HERE!
                    with m_r2_c1:
                        rain_val = normalized_features.get('rain', 0.0)
                        rain_status, travel_risk = translate_rainfall(rain_val)
                        
                        st.metric(
                            label="🌧️ Predicted Rainfall", 
                            value=f"{rain_val:.2f} mm",
                            delta=f"{rain_status} ({travel_risk})",
                            delta_color="off"
                        )
                    with m_r2_col2:
                        st.metric(label="💨 Estimated Wind", value=f"{normalized_features.get('wind_speed', 0.0):.1f} km/h")

                with col_advisory:
                    try:
                        mid_lat = (src_lat + dest_lat) / 2
                        mid_lon = (src_lon + dest_lon) / 2

                        # 📍 1. Draw the Origin and Destination markers
                        map_data = pd.DataFrame({
                            "lat": [src_lat, dest_lat],
                            "lon": [src_lon, dest_lon],
                            "color": [[231, 76, 60, 200], [46, 204, 113, 200]],
                            "name": ["Origin", "Destination"]
                        })

                        scatter_layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=map_data,
                            get_position="[lon, lat]",
                            get_fill_color="color",
                            get_radius=8000,
                            radius_min_pixels=8,
                            radius_max_pixels=25,
                            pickable=True
                        )

                        # 🗺️ 2. THE WOW FACTOR: Draw the winding roads!
                        if route_coordinates:
                            # 🛡️ FIX 1: Pass the data as a pure Python dictionary list instead of a DataFrame
                            route_layer = pdk.Layer(
                                "PathLayer",
                                data=[{"path": route_coordinates}], 
                                get_path="path",
                                get_color=[52, 152, 219, 255], 
                                width_scale=20,
                                width_min_pixels=5,
                                get_width=5
                            )
                        else:
                            # Failsafe Fallback
                            route_layer = pdk.Layer(
                                "LineLayer",
                                data=[{"start": [src_lon, src_lat], "end": [dest_lon, dest_lat]}],
                                get_source_position="start",
                                get_target_position="end",
                                get_color=[52, 152, 219, 180],
                                get_width=5,
                            )

                        # 3. Render the 3D Canvas
                        # Lowered the pitch from 45 to 30 so city names are easier to read
                        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=6.5, pitch=30) 
                        
                        r = pdk.Deck(
                            layers=[route_layer, scatter_layer], 
                            initial_view_state=view_state, 
                            map_style="road", # 🌟 CHANGED: Bright, clean, street-level map!
                            tooltip={"text": "{name}"}
                        )
                        st.pydeck_chart(r)

                    except Exception as e:
                        st.warning(f"⚠️ Map rendering error: {e}. Fallback loaded.")
                        st.map(pd.DataFrame({"latitude": [dest_lat], "longitude": [dest_lon]}), width='stretch')
                        
                weather_weight = float(normalized_features['rain'] * 1.5 + normalized_features['wind_speed'] * 0.5)
                terrain_weight = float(telemetry.get('elevation_penalty', 0) * 2.0 + telemetry.get('transport_complexity_score', 0) * 0.2)
                crowd_weight = float(telemetry.get('crowd_baseline', 45) + telemetry.get('festival_boost', 0))
                total_weight = weather_weight + terrain_weight + crowd_weight
                if total_weight == 0: total_weight = 1.0

                w_pct = round((weather_weight / total_weight) * 100, 1)
                t_pct = round((terrain_weight / total_weight) * 100, 1)
                c_pct = round((crowd_weight / total_weight) * 100, 1)

                generated_narrative = generate_semantic_narrative(normalized_features, tier)

                st.markdown("---")
                st.write("")

                # 🌟 NEW: Dynamic Interactive Speedometer Gauge (5 TIERS)
                score_val = float(score)

                # Badge label + color logic (5 Categories)
                if score_val <= 20:
                    badge_label, badge_color = "Minimal Risk", "#0f5132"
                    badge_bg = "#d1e7dd"
                elif score_val <= 40:
                    badge_label, badge_color = "Low Risk", "#3f6212"
                    badge_bg = "#d9f99d"
                elif score_val <= 60:
                    badge_label, badge_color = "Moderate Risk", "#854d0e"
                    badge_bg = "#fef08a"
                elif score_val <= 80:
                    badge_label, badge_color = "Elevated Risk", "#9a3412"
                    badge_bg = "#ffedd5"
                else:
                    badge_label, badge_color = "Hazardous", "#842029"
                    badge_bg = "#f8d7da"

                # Build the Plotly Gauge with 5 Steps
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_val,
                    number={
                        "suffix": " / 100",
                        "font": {"size": 44, "color": "#1a1a1a"},
                    },
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "#aaaaaa",
                            "tickfont": {"size": 11, "color": "#888"},
                        },
                        "bar": {"color": "rgba(0,0,0,0.75)", "thickness": 0.18},
                        "bgcolor": "white",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 20],  "color": "#d1e7dd"},  # 🟢 Minimal
                            {"range": [20, 40], "color": "#d9f99d"},  # 🟡 Low
                            {"range": [40, 60], "color": "#fef08a"},  # 🟠 Moderate
                            {"range": [60, 80], "color": "#ffedd5"},  # 🔴 Elevated
                            {"range": [80, 100],"color": "#f8d7da"},  # 🚨 Hazardous
                        ],
                        "threshold": {
                            "line": {"color": "rgba(0,0,0,0.3)", "width": 2},
                            "thickness": 0.75,
                            "value": score_val,
                        },
                    },
                ))

                fig.update_layout(
                    height=240,
                    margin=dict(l=24, r=24, t=16, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter, sans-serif"},
                )

                st.plotly_chart(fig, use_container_width=True)

                # Centered badge + label
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(
                        f"""
                        <div style="text-align:center; margin-top:-8px;">
                            <span style="
                                background:{badge_bg}; color:{badge_color};
                                padding:3px 14px; border-radius:20px;
                                font-size:13px; font-weight:600;
                            ">{badge_label}</span>
                            <p style="font-size:11px; color:#999; margin-top:10px;">
                                🤖 Powered by AI Risk Models &nbsp;·&nbsp; v{res_data.get('model_version', '3.0')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.write("")
                
                tier_clean = str(tier).lower()
                if "minimal" in tier_clean or "low" in tier_clean:
                    st.success(generated_narrative)  
                elif "moderate" in tier_clean or "elevated" in tier_clean:
                    st.warning(generated_narrative)  
                else:
                    st.error(generated_narrative)    
                
                
                
                st.write("---")
                st.markdown("#### 📡 Visualized Risk Distribution Share Graph")
                live_shares = {
                    "🌧️ Live Weather Conditions": w_pct,
                    "⛰️ Local Mountain & Terrain Factors": t_pct,
                    "🧑‍🤝‍🧑 Tourist Traffic & Crowd Baseline": c_pct
                }
                share_df = pd.DataFrame(list(live_shares.items()), columns=["Risk Driver Group", "Influence Share (%)"])
                st.bar_chart(share_df.set_index("Risk Driver Group"), horizontal=True)

                if st.session_state.get("log_this_run", False):
                    
                    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    target_date_str = travel_date.strftime("%Y-%m-%d")
                    current_time_str = datetime.now().strftime("%I:%M:%S %p")
                    saved_query = st.session_state.get("saved_final_query", "Unknown")
                    
                    telemetry = st.session_state.get("latest_telemetry", {})
                    
                    # 1. CALCULATE THE REAL TRUTH using the function
                    math_score = get_mathematical_ground_truth(telemetry)
                    
                    # 2. LOGGING PAYLOAD
                    sheet_row_payload = [
                        current_timestamp,
                        saved_query, 
                        res_data.get("resolved_name", "N/A"),
                        float(dest_lat or 0.0),
                        float(dest_lon or 0.0),
                        round(float(score or 0.0), 2),
                        math_score, 
                        tier,
                        res_data.get("destination_type", "General"),
                        res_data.get("destination_description", "N/A"),
                        res_data.get("model_version", "v1.0_ML"),
                        target_date_str,
                        json.dumps(telemetry)
                    ]
                    write_cloud_prediction_log(sheet_row_payload)

                    # 3. Update session history
                    history_log = {
                        "Time Checked": current_time_str,
                        "📅 Planned Travel Date": target_date_str,
                        "Destination Location": "Manali (STRESS_TEST)" if is_test_mode else res_data.get("resolved_name"),
                        "Risk Score Index": float(score),
                        "Safety Status Category": tier
                    }
                    st.session_state.prediction_history.append(history_log)
                    if len(st.session_state.prediction_history) > MAX_HISTORY:
                        st.session_state.prediction_history.pop(0)
                        
                    st.session_state["log_this_run"] = False

        st.write("---")
        st.header("🕒 Recent Checks This Session")
        if st.session_state.prediction_history:
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(hist_df.sort_index(ascending=False), width='stretch')
        else:
            st.info("ℹ️ No routes checked yet this session. Enter a destination above to see your recent search log history.")

    with tab_budget:
        shared_loc = st.session_state.get("shared_query", "Goa")
        shared_date = st.session_state.get("shared_date", datetime.now().strftime("%Y-%m-%d"))
        render_budget_tab(shared_loc, shared_date, write_budget_cloud_log)

elif app_view == "📊 Travel Data Analytics":
    st.title("📊 Travel Network Analytics & System Monitor")
    st.markdown("### Management Control Room for Travel Metrics & Trend Analysis")
    st.write("---")

    st.sidebar.markdown("### 📊 Control Center Status")
    if BACKEND_ONLINE:
        st.sidebar.success("🟢 Analytics Core Online")
    else:
        st.sidebar.error("🔴 Analytics Core Offline")

    tab_risk_analytics, tab_budget_analytics = st.tabs(["🛡️ Route & Risk Trends", "💰 Financial & Budget Analytics"])

    with tab_risk_analytics:
        st.header("⚡ Live Route Risk Summary & User Traffic")
        
        try:
            db_df = fetch_cloud_prediction_logs()
        except Exception:
            db_df = None
            
        if db_df is None or db_df.empty:
            if "prediction_history" in st.session_state and st.session_state.prediction_history:
                db_df = pd.DataFrame(st.session_state.prediction_history)
                st.info("💡 Displaying local session data (Google Sheets database is empty or syncing).")
            else:
                db_df = pd.DataFrame()

        if not db_df.empty:
            score_col = next((c for c in ['predicted_hazard_score', 'Risk Score Index', 'score', 'Risk Score'] if c in db_df.columns), None)
            loc_col = next((c for c in ['resolved_name', 'Destination Location', 'location', 'Location'] if c in db_df.columns), None)
            cat_col = next((c for c in ['risk_category', 'Safety Status Category', 'category'] if c in db_df.columns), None)
            
            if loc_col:
                st.write("") 
                city_list = ["All Destinations"] + sorted(db_df[loc_col].astype(str).unique().tolist())
                selected_city = st.selectbox("🎯 Filter Analytics by Destination:", options=city_list)
                
                if selected_city != "All Destinations":
                    db_df = db_df[db_df[loc_col] == selected_city]
            
            if db_df.empty:
                st.warning(f"No data available for {selected_city}.")
            else:
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                with kpi_col1:
                    st.metric(label="🔢 Total Safety Reports", value=f"{len(db_df)}")
                with kpi_col2:
                    if score_col:
                        avg_risk = pd.to_numeric(db_df[score_col], errors='coerce').mean()
                        st.metric(label="🎚️ Average Risk Score", value=f"{avg_risk:.1f} / 100")
                    else:
                        st.metric(label="🎚️ Average Risk Score", value="N/A")
                with kpi_col3:
                    if loc_col and selected_city == "All Destinations":
                        most_searched = db_df[loc_col].mode()[0] if not db_df.empty else "N/A"
                        st.metric(label="📍 Top Destination", value=most_searched)
                    else:
                        st.metric(label="📍 Selected Filter", value=selected_city if loc_col else "N/A")

                st.write("---")
                g_col1, g_col2 = st.columns([3, 2])
                with g_col1:
                    st.markdown("#### 🕒 Safety Risk Variations Over Time")
                    if score_col:
                        st.line_chart(pd.to_numeric(db_df[score_col], errors='coerce').reset_index(drop=True))
                
                with g_col2:
                    st.markdown("#### 🗂️ Risk Category Share")
                    if cat_col:
                        st.bar_chart(db_df[cat_col].value_counts())

                st.write("---")
                st.markdown("#### 🗄️ Search Logs Archive")
                st.dataframe(db_df, width='stretch')


                # 🌟 AI TRAINING DATA ARCHIVE (WITH FILTER) 🌟
                st.write("---")
                st.markdown("#### 📂 AI Training Data Archive (Weather & Hazards)")
                st.caption("This is the master historical dataset used to train the XGBoost Risk Engine.")
                try:
                    csv_path = "data/master_feature_table_with_hazards.csv"
                    if os.path.exists(csv_path):
                        training_df = pd.read_csv(csv_path)
                        
                        # 1. Find the location column (usually named 'location' or 'city')
                        loc_col_train = next((c for c in ['location', 'Location', 'city', 'City'] if c in training_df.columns), None)
                        
                        # 2. Add the Filter Dropdown
                        if loc_col_train:
                            train_city_list = ["All Destinations"] + sorted(training_df[loc_col_train].astype(str).unique().tolist())
                            selected_train_city = st.selectbox("🎯 Filter Training Data by Destination:", options=train_city_list, key="train_city_filter")
                            
                            # Filter the dataframe based on the dropdown
                            if selected_train_city != "All Destinations":
                                display_df = training_df[training_df[loc_col_train] == selected_train_city]
                            else:
                                display_df = training_df
                        else:
                            display_df = training_df  # Fallback if no location column exists
                        
                        # 3. Display the filtered dataframe
                        st.dataframe(display_df, width='stretch', height=300)
                        
                        # 4. Download button (downloads the filtered data!)
                        training_csv = display_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Displayed Data (CSV)",
                            data=training_csv,
                            file_name=f"training_data_{selected_train_city if loc_col_train and selected_train_city != 'All Destinations' else 'full'}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("💡 Training data CSV file not found in the 'data/' folder.")
                except Exception as e:
                    st.error(f"⚠️ Could not load training data: {e}")


                st.write("---")
                st.markdown("#### ⚖️ AI Accuracy & Model Truth Comparison")
                
                # 🛡️ 1. STRIP HIDDEN SPACES FROM HEADERS
                db_df.columns = db_df.columns.str.strip()
                
                # 2. FIND THE COLUMNS
                true_col = next((c for c in ['calculated_baseline_risk', 'Calculated Baseline Risk', 'actual_score'] if c in db_df.columns), None)
                
                if not true_col:
                    st.warning(f"⚠️ Could not find the baseline column! Headers found: {db_df.columns.tolist()}")
                elif not score_col:
                    st.warning(f"⚠️ Could not find the AI prediction column! Headers found: {db_df.columns.tolist()}")
                else:
                    import plotly.express as px
                    
                    # 3. EXTRACT AND CLEAN THE DATA
                    db_df[score_col] = pd.to_numeric(db_df[score_col], errors='coerce')
                    db_df[true_col] = pd.to_numeric(db_df[true_col], errors='coerce')
                    
                    compare_df = pd.DataFrame({
                        'AI Predicted Score': db_df[score_col],
                        'Mathematical Truth': db_df[true_col]
                    }).dropna()
                    
                    if not compare_df.empty:
                        mae = (compare_df['AI Predicted Score'] - compare_df['Mathematical Truth']).abs().mean()
                        
                        comp_col1, comp_col2 = st.columns([1, 2])
                        with comp_col1:
                            st.metric(label="🎯 Mean Absolute Error (MAE)", value=f"{mae:.2f} pts", delta="Lower is better", delta_color="inverse")
                            st.caption("Average difference between the AI prediction and the mathematical baseline.")
                            st.metric(label="✅ Validated Trips", value=len(compare_df))
                            
                        with comp_col2:
                            st.markdown("**AI Prediction vs. Baseline Trend**")
                            compare_df['Trip Sequence'] = range(1, len(compare_df) + 1)
                            show_dots = len(compare_df) <= 10 
                            
                            fig = px.line(
                                compare_df, 
                                x='Trip Sequence', 
                                y=['AI Predicted Score', 'Mathematical Truth'], 
                                markers=show_dots 
                            )
                            fig.update_layout(
                                legend_title_text='', 
                                xaxis_title="Recent Searches (Chronological)", 
                                yaxis_title="Risk Score",
                                yaxis_range=[min(compare_df['AI Predicted Score'].min(), 10), compare_df['AI Predicted Score'].max() + 5]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("💡 Data found, but predictions or baselines are blank (NaN). Run a new search to generate paired data!")

        # This catches the empty database state
        else:
            st.info("💡 No risk searches recorded yet. Go to the Risk Checker to generate data!")
    with tab_budget_analytics:
        st.header("💸 AI Financial Forecasting Insights")
        
        budget_df = fetch_budget_cloud_logs()

        if budget_df is None or budget_df.empty:
            st.warning("⚠ Could not connect to Google Sheets, or the 'budget_forecasts' tab is empty.")
            st.info("Run a budget calculation in the main app to start generating financial analytics!")
        else:
            import plotly.express as px
            
            cost_col = 'Estimated Total' if 'Estimated Total' in budget_df.columns else budget_df.columns[-3]
            stress_col = 'Stress Score (%)' if 'Stress Score (%)' in budget_df.columns else budget_df.columns[-2]
            style_col = 'Style' if 'Style' in budget_df.columns else budget_df.columns[5]
            transport_col = 'Transport' if 'Transport' in budget_df.columns else budget_df.columns[6]
            
            loc_col_budget = 'Location' if 'Location' in budget_df.columns else budget_df.columns[1]

            if loc_col_budget:
                st.write("") 
                budget_city_list = ["All Destinations"] + sorted(budget_df[loc_col_budget].astype(str).unique().tolist())
                
                selected_budget_city = st.selectbox("🎯 Filter Finances by Destination:", options=budget_city_list, key="budget_city_filter")
                
                if selected_budget_city != "All Destinations":
                    budget_df = budget_df[budget_df[loc_col_budget] == selected_budget_city]

            if budget_df.empty:
                st.warning(f"No financial data available for {selected_budget_city}.")
            else:
                budget_df[cost_col] = pd.to_numeric(budget_df[cost_col], errors='coerce')
                budget_df[stress_col] = pd.to_numeric(budget_df[stress_col], errors='coerce')
                
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    st.metric("Total Forecasts Run", len(budget_df))
                with b_col2:
                    avg_stress = budget_df[stress_col].mean()
                    st.metric("Avg Financial Stress", f"{avg_stress:.1f}%" if pd.notnull(avg_stress) else "N/A")
                with b_col3:
                    avg_cost = budget_df[cost_col].mean()
                    st.metric("Avg Estimated Trip Cost", f"₹{avg_cost:,.0f}" if pd.notnull(avg_cost) else "N/A")

                st.write("---")
                
                # 📈 UPGRADED TRIP COST CHART
                st.markdown("#### 📈 Trip Cost Estimates Over Time")
                
                # 1. Grab the real timestamp instead of a fake sequence
                time_col = 'Timestamp' if 'Timestamp' in budget_df.columns else budget_df.columns[0]
                
                # Format it nicely (e.g., "Jun 22, 14:30") so it fits on the axis
                try:
                    budget_df['Date Run'] = pd.to_datetime(budget_df[time_col]).dt.strftime('%b %d, %H:%M')
                except Exception:
                    budget_df['Date Run'] = budget_df[time_col] # Fallback just in case
                
                # 2. Add rich context for the hover tooltip!
                hover_details = [col for col in ['Location', 'Travel Date', 'Days', 'People', 'Style'] if col in budget_df.columns]
                
                fig_cost = px.line(
                    budget_df, 
                    x='Date Run', 
                    y=cost_col,
                    markers=True,
                    line_shape="spline",
                    color_discrete_sequence=["#2ecc71"],
                    hover_data=hover_details # Injects the trip details into the mouse hover
                )
                
                # 3. Clean up the UI
                fig_cost.update_layout(
                    xaxis_title="When Forecast Was Run", 
                    yaxis_title="Estimated Cost (INR)",
                    hovermode="x unified", # Creates a beautiful, grouped pop-up box
                    xaxis_tickangle=-45 # Tilts the date text so it doesn't overlap on crowded charts
                )
                st.plotly_chart(fig_cost, use_container_width=True)

                st.write("---")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown("#### 🎒 Preferred Travel Styles")
                    style_counts = budget_df[style_col].value_counts().reset_index()
                    style_counts.columns = [style_col, 'Count']
                    fig_style = px.pie(style_counts, values='Count', names=style_col, hole=0.4)
                    st.plotly_chart(fig_style, use_container_width=True)
                    
                with chart_col2:
                    st.markdown("#### 🚆 Preferred Transport Modes")
                    transport_counts = budget_df[transport_col].value_counts().reset_index()
                    transport_counts.columns = [transport_col, 'Count']
                    fig_trans = px.pie(transport_counts, values='Count', names=transport_col, hole=0.4)
                    st.plotly_chart(fig_trans, use_container_width=True)

                st.write("---")
                st.markdown("#### 💾 Complete Budget Forecast Logs")
                st.dataframe(budget_df, width='stretch')
                
                csv_budget = budget_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Budget Data (CSV)",
                    data=csv_budget,
                    file_name=f"budget_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )