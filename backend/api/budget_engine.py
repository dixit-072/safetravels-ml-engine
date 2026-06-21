import csv
import os
from datetime import datetime

# =====================================================================
# DYNAMIC DATABASE LOADER (Reads data/cities.csv)
# =====================================================================
CITY_DB = {}

def load_city_database():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    csv_path = os.path.join(project_root, "data", "cities.csv")
    
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                city_name = row["City"].strip().title()
                peak_months = [int(m) for m in row["Peak_Months"].split(",") if m.strip()]
                
                CITY_DB[city_name] = {
                    "Tier": row["Tier"].strip(),
                    "Peak_Months": peak_months,
                    "Base_Flight_Cost": float(row["Base_Flight_Cost"]),
                    "Has_Airport": int(row.get("Has_Airport", 1)), 
                    "Has_Train": int(row.get("Has_Train", 1))      
                }
    else:
        print(f"⚠️ Warning: cities.csv not found at {csv_path}! Relying on fallback data.")

load_city_database()

# =====================================================================
# 1. FINANCIAL BASELINE MATRIX
# =====================================================================
COST_MATRIX = {
    "Tier_1": {"description": "Metro", "base_hotel": 3500, "base_food": 1200, "base_commute": 600, "base_activity": 1500},
    "Tier_2": {"description": "Standard", "base_hotel": 2000, "base_food": 800, "base_commute": 400, "base_activity": 1000},
    "Tier_3": {"description": "Budget", "base_hotel": 1000, "base_food": 500, "base_commute": 200, "base_activity": 400},
    "Tier_Mountain": {"description": "High-Altitude", "base_hotel": 2500, "base_food": 900, "base_commute": 800, "base_activity": 1200}
}

TIER_1_CITIES = ["Goa", "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune"]
TIER_2_CITIES = ["Jaipur", "Shimla", "Manali", "Udaipur", "Agra", "Kochi", "Rishikesh", "Varanasi", "Darjeeling", "Chandigarh"]

# =====================================================================
# 2. HELPER FUNCTIONS 
# =====================================================================
def get_dynamic_tier(city_name: str, elevation: float) -> str:
    clean_city = city_name.strip().title()
    if clean_city in CITY_DB:
        return CITY_DB[clean_city]["Tier"]
        
    if clean_city in TIER_1_CITIES: return "Tier_1"
    elif clean_city in TIER_2_CITIES: return "Tier_2"
    elif elevation > 2000.0: return "Tier_Mountain"
    else: return "Tier_3"

def get_weekend_premium(target_date_str: str) -> float:
    try:
        date_obj = datetime.strptime(target_date_str, "%Y-%m-%d")
        if date_obj.weekday() in [4, 5]: return 1.15
    except Exception: pass
    return 1.00

def calculate_transport_cost(distance_km: float, transport_mode: str, is_mountain: int) -> float:
    RATES_PER_KM = {"Train (Sleeper/3AC)": 2.0, "Bus (Non-AC)": 1.5, "Bus (AC Volvo)": 2.5, "Personal Car / Taxi": 10.0, "Flight": 6.5}
    base_rate = RATES_PER_KM.get(transport_mode, 2.5)
    
    if is_mountain == 1 and transport_mode in ["Bus (Non-AC)", "Bus (AC Volvo)", "Personal Car / Taxi"]:
        base_rate = base_rate * 1.4 
        
    estimated_fare = distance_km * base_rate
    
    if transport_mode == "Flight": estimated_fare += 2500.0 
    if transport_mode == "Personal Car / Taxi" and distance_km > 100: estimated_fare += (distance_km * 1.5) 
        
    return round(estimated_fare, 2)

def calculate_rigid_math_score(telemetry: dict) -> float:
    base_score = 10.0 
    rain = telemetry.get("rain", 0.0)
    temp = telemetry.get("temp_max", 25.0)
    elevation = telemetry.get("elevation", 0.0)
    distance = telemetry.get("route_distance_km", 0.0)
    
    if rain > 30.0: base_score += 40.0
    elif rain > 10.0: base_score += 20.0
        
    if temp > 40.0 or temp < 0.0: base_score += 25.0
    elif temp > 35.0: base_score += 10.0
        
    if elevation > 3000.0: base_score += 30.0
    elif elevation > 2000.0: base_score += 15.0
        
    if distance > 600.0: base_score += 15.0
        
    return round(min(base_score, 100.0), 1)

ACTIVITY_DB = {
    "River Rafting / Adventure Sports": {"cost": 1500, "type": "per_person", "weather_sensitive": True},
    "Local Sightseeing Tour (Cab)": {"cost": 2500, "type": "group", "weather_sensitive": False},
    "Fine Dining / Special Meal": {"cost": 1200, "type": "per_person", "weather_sensitive": False},
    "Museums / Monument Entry": {"cost": 500, "type": "per_person", "weather_sensitive": False},
    "Spa / Wellness Session": {"cost": 2000, "type": "per_person", "weather_sensitive": False}
}   

# =====================================================================
# 3. MASTER BUDGET CALCULATOR (THE BRAIN MERGE)
# =====================================================================
def calculate_dynamic_budget(user_inputs: dict, ml_telemetry: dict) -> dict:
    raw_city = user_inputs.get("location_query", "Unknown")
    city_name = raw_city.strip().title() 
    
    num_stays = user_inputs.get("num_stays", 2) 
    pax = user_inputs.get("num_people", 2)
    style = user_inputs.get("travel_style", "Standard")
    transport = user_inputs.get("transport_mode", "Bus (AC Volvo)")
    max_budget = user_inputs.get("max_budget", 30000.0)
    target_date = user_inputs.get("target_date", "")
    is_round_trip = user_inputs.get("is_round_trip", True) 
    selected_activities = user_inputs.get("selected_activities", [])

    elevation = ml_telemetry.get("elevation", 0.0)
    rain = ml_telemetry.get("rain", 0.0)
    temp_max = ml_telemetry.get("temp_max", 25.0)
    risk_score = ml_telemetry.get("predicted_hazard_score", 0.0)
    festival_boost = ml_telemetry.get("festival_boost", 0.0)
    distance_km = ml_telemetry.get("route_distance_km", 0.0)
    is_mountain = ml_telemetry.get("mountain_flag", 0)
    
    rigid_math_score = calculate_rigid_math_score(ml_telemetry)

    applied_taxes = [] 
    city_data = CITY_DB.get(city_name, {"Tier": "Tier_3", "Peak_Months": [], "Base_Flight_Cost": 5000})
    tier_key = city_data["Tier"]
    base_rates = COST_MATRIX[tier_key].copy() 

    target_month = 1
    if target_date:
        try:
            target_month = datetime.strptime(target_date, "%Y-%m-%d").month
        except Exception:
            pass 

    season_factor = 1.40 if target_month in city_data["Peak_Months"] else 0.80
    if season_factor > 1.0:
        applied_taxes.append("📈 Peak Season Pricing (+40%)")
    elif season_factor < 1.0:
        applied_taxes.append("📉 Off-Season Discount (-20%)")
        
    weekend_mult = get_weekend_premium(target_date)
    if weekend_mult > 1.0:
        applied_taxes.append("📅 Weekend Premium (+15%)")

    hotel_style_mult = 0.6 if style == "Backpacker" else 2.5 if style == "Luxury" else 1.0
    food_style_mult = 0.7 if style == "Backpacker" else 2.0 if style == "Luxury" else 1.0
    transport_style_mult = 0.8 if style == "Backpacker" else 1.2 if style == "Luxury" else 1.0
    commute_style_mult = 0.5 if style == "Backpacker" else 1.5 if style == "Luxury" else 1.0
    
    festival_mult = 1.0 + (festival_boost / 10.0) 
    final_hotel_mult = hotel_style_mult * weekend_mult * festival_mult

    base_rates["base_hotel"] *= season_factor

    if style == "Backpacker":
        hotel_cost = base_rates["base_hotel"] * final_hotel_mult * pax * num_stays 
    else:
        rooms_needed = (pax + 1) // 2 
        hotel_cost = base_rates["base_hotel"] * final_hotel_mult * rooms_needed * num_stays 
    
    active_days = num_stays + 1 if num_stays > 0 else 1 
    food_cost = base_rates["base_food"] * food_style_mult * pax * active_days
    commute_cost = base_rates["base_commute"] * commute_style_mult * pax * active_days
    
    activity_cost = 0
    if selected_activities:
        for act in selected_activities:
            act_data = ACTIVITY_DB.get(act)
            if not act_data: continue
                
            if risk_score > 80 and act_data.get("weather_sensitive"):
                applied_taxes.append(f"⚠️ AI Safety Override: '{act}' removed due to weather risk.")
                continue 
                
            if act_data["type"] == "per_person":
                activity_cost += (act_data["cost"] * pax)
            elif act_data["type"] == "group":
                activity_cost += act_data["cost"] 
    else:
        activity_cost = 200 * pax 

    if "Fine Dining / Special Meal" in selected_activities:
        standard_dinner_cost = (base_rates["base_food"] * food_style_mult) * 0.40 
        food_cost = max(0, food_cost - (standard_dinner_cost * pax))

    if "Local Sightseeing Tour (Cab)" in selected_activities:
        commute_cost *= 0.20

    if distance_km < 25.0 and distance_km > 0.1: 
        transport_cost = 0.0
    else:
        calc_dist = distance_km if distance_km > 0 else 500.0
        rt_mult = 2 if is_round_trip else 1

        if "Bus" in transport:
            base_ticket = 1500 if (calc_dist > 400 and is_mountain) else 1000 if calc_dist > 400 else 600   
            if "Non-AC" in transport:
                base_ticket *= 0.70
            transport_cost = base_ticket * rt_mult * pax * transport_style_mult 
            
        elif "Train" in transport:
            base_ticket = 800 if calc_dist > 400 else 400
            transport_cost = base_ticket * rt_mult * pax * transport_style_mult

        elif transport == "Flight":
            one_way_flight = city_data["Base_Flight_Cost"] * season_factor
            transport_cost = one_way_flight * rt_mult * pax * transport_style_mult

        elif transport == "Personal Car / Taxi":
            one_way_transport = calculate_transport_cost(distance_km, transport, is_mountain)
            transport_cost = one_way_transport * rt_mult
            commute_cost *= 0.3
            
        else:
            one_way_transport = calculate_transport_cost(distance_km, transport, is_mountain)
            transport_cost = one_way_transport * rt_mult * pax

    if rain > 10.0:
        commute_cost *= 1.30
        applied_taxes.append("🌧️ Heavy Rain Surge: Commute +30%")
    if temp_max > 35.0:
        hotel_cost *= 1.15
        applied_taxes.append("🔥 Heatwave Surge: AC Hotel +15%")

    subtotal = hotel_cost + food_cost + commute_cost + activity_cost + transport_cost

    base_emergency = 500 * pax
    if risk_score > 85: risk_mult = 2.5
    elif risk_score > 65: risk_mult = 1.5
    elif risk_score > 45: risk_mult = 1.0
    else: risk_mult = 0.5
    
    emergency_buffer = base_emergency * risk_mult
    grand_total = subtotal + emergency_buffer

    stress_score = (grand_total / max_budget) * 100 if max_budget > 0 else 100.0
    stress_score = round(min(stress_score, 200.0), 1)

    if grand_total <= max_budget:
        status = f"✅ Under Budget by ₹{max_budget - grand_total:,.0f}"
        if stress_score < 60:
            budget_summary = f"Highly Comfortable. You are utilizing {stress_score}% of your allocated funds."
        else:
            budget_summary = f"Balanced. You are utilizing {stress_score}% of your funds. Stick to the itinerary."
    else:
        status = f"⚠️ Over Budget by ₹{grand_total - max_budget:,.0f}"
        budget_summary = f"Critical Financial Stress. You are operating at {stress_score}% of your maximum limit. Downgrades recommended."

    return {
        "estimated_total": round(grand_total, 2),
        "budget_status": status,
        "financial_stress_score": stress_score,
        "budget_summary": budget_summary,
        "applied_taxes": applied_taxes,
        "math_hazard_score": rigid_math_score,        
        "ml_hazard_score": round(risk_score, 1),     
        "breakdown": {
            "accommodation": round(hotel_cost, 2),
            "food": round(food_cost, 2),
            "local_commute": round(commute_cost, 2),
            "transport": round(transport_cost, 2),
            "activities": round(activity_cost, 2),
            "emergency_buffer": round(emergency_buffer, 2)
        }
    }