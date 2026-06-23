import random
import logging

# 🌟 Updated to the Scientific Meteorologist Thresholds!
THRESHOLDS = {
    "rain": {"moderate": 7.6, "heavy": 50.0},
    "wind": {"high": 35.0},
    "elevation": {"mountain": 1000.0},
    "temperature": {"cold": 10.0, "hot": 35.0}
}

SEMANTIC_LIBRARY = {
    "weather": {
        "no_rain": [
            "Skies are clear with no expected rainfall.",
            "Dry conditions along the route.",
            "Zero precipitation detected on the radar."
        ],
        "light_rain": [
            "Expect a little light rain ({rain:.1f} mm) along the way.",
            "Weather is mostly clear with minor damp patches ({rain:.1f} mm).",
            "Baseline conditions look good with light drizzle ({rain:.1f} mm)."
        ],
        "moderate_rain": [
            "Steady rain ({rain:.1f} mm) is falling. Roads will be slick.",
            "Moderate rainfall ({rain:.1f} mm) is affecting visibility.",
            "Expect running surface water from active rain ({rain:.1f} mm)."
        ],
        "heavy_rain": [
            "⚠️ Heavy downpours ({rain:.1f} mm) are causing major pooling on the roads.",
            "⚠️ Intense rainfall ({rain:.1f} mm) is actively flooding low drainage spots.",
            "⚠️ Torrential cloudbursts ({rain:.1f} mm) have dropped road visibility to near-zero."
        ]
    },
    "temperature": {
        "cold": [
            "It is cold ({temp_max:.1f}°C)—watch out for sudden patches of frost or black ice.",
            "Temperatures are near freezing ({temp_max:.1f}°C). Drive cautiously over bridges."
        ],
        "hot": [
            "High heat ({temp_max:.1f}°C) is putting extra stress on car engines and tires.",
            "Extreme heat alert ({temp_max:.1f}°C). Check your coolant levels."
        ],
        "normal": [
            "Temperatures are comfortable ({temp_max:.1f}°C).",
            "Thermal conditions are standard ({temp_max:.1f}°C)."
        ]
    },
    "terrain": {
        "mountain": [
            "You are crossing a high mountain pass ({elevation:.0f} m) with steep blind curves.",
            "This high-altitude zone ({elevation:.0f} m) is prone to rockfalls and valley fog."
        ],
        "plain": [
            "The route lies across flat, easy plains territory.",
            "Terrain profile is completely flat and stable."
        ]
    },
    "transport": {
        "high_wind": [
            "Strong crosswinds ({wind_speed:.1f} km/h) are actively buffeting high-profile vehicles.",
            "Gusty wind warnings ({wind_speed:.1f} km/h) require a firm grip on the steering wheel."
        ],
        "normal": [
            "Winds are calm ({wind_speed:.1f} km/h) and well within safe limits.",
            "Air movement along the corridor is a normal {wind_speed:.1f} km/h."
        ]
    },
    "interpretation": {
        "Minimal": [
            "The route is entirely clear and safe (Hazard Index: {risk_score:.0f}/100).",
            "All sensors show optimal, stress-free travel tracks (Hazard Index: {risk_score:.0f}/100)."
        ],
        "Low": [
            "Minor local weather shifts are present, but it's a routine drive (Hazard Index: {risk_score:.0f}/100).",
            "Conditions are comfortable with minimal layout friction (Hazard Index: {risk_score:.0f}/100)."
        ],
        "Moderate": [
            "Active elements mean you need to stay alert (Hazard Index: {risk_score:.0f}/100).",
            "Friction checks suggest slowing down around corners (Hazard Index: {risk_score:.0f}/100)."
        ],
        "Elevated": [
            "Compounding risk factors are driving up stress levels across the track (Hazard Index: {risk_score:.0f}/100).",
            "Conditions are demanding. Traffic delays and lane restrictions are likely (Hazard Index: {risk_score:.0f}/100)."
        ],
        "Critical": [
            "🚨 CRITICAL WARNING: Multiple structural hazards have maxed out safe limits (Hazard Index: {risk_score:.0f}/100).",
            "🚨 SAFETY ENVELOPE COLLAPSED: Severe combined danger vectors across the route (Hazard Index: {risk_score:.0f}/100)."
        ]
    },
    "advice": {
        "Minimal": ["Have a great trip!", "Proceed normal speed as planned."],
        "Low": ["Check local news maps before leaving.", "Keep an eye out for standard city traffic updates."],
        "Moderate": ["Drop your speed, extend your braking distance, and avoid night driving.", "Take curves gently."],
        "Elevated": ["Pack an emergency kit, secure all cargo, and double-check your brakes.", "Delay trip if possible."],
        "Critical": ["Do not drive. Postpone your trip immediately until the highway authority clears the area."]
    }
}

def generate_semantic_narrative(features: dict, risk_tier: str) -> str:
    """Compiles the telemetry into clean, snappy, short bullet points for everyday users."""
    try:
        safe_features = {
            "rain": float(features.get("rain", 0.0)),
            "elevation": float(features.get("elevation", 0.0)),
            "wind_speed": float(features.get("wind_speed", 0.0)),
            "temp_max": float(features.get("temp_max", 20.0)),
            "risk_score": float(features.get("risk_score", 0.0)),
            "resolved_name": str(features.get("resolved_name", "Selected Corridor"))
        }

        tier_clean = str(risk_tier).lower()
        if "minimal" in tier_clean: mapped_tier = "Minimal"
        elif "low" in tier_clean: mapped_tier = "Low"
        elif "moderate" in tier_clean: mapped_tier = "Moderate"
        elif "elevated" in tier_clean or "severe" in tier_clean: mapped_tier = "Elevated"
        else: mapped_tier = "Critical"

        # Router Selections
        if safe_features["rain"] >= THRESHOLDS["rain"]["heavy"]: weather_txt = random.choice(SEMANTIC_LIBRARY["weather"]["heavy_rain"])
        elif safe_features["rain"] >= THRESHOLDS["rain"]["moderate"]: weather_txt = random.choice(SEMANTIC_LIBRARY["weather"]["moderate_rain"])
        elif safe_features["rain"] > 0.1: weather_txt = random.choice(SEMANTIC_LIBRARY["weather"]["light_rain"])
        else: weather_txt = random.choice(SEMANTIC_LIBRARY["weather"]["no_rain"])

        if safe_features["temp_max"] <= THRESHOLDS["temperature"]["cold"]: temp_txt = random.choice(SEMANTIC_LIBRARY["temperature"]["cold"])
        elif safe_features["temp_max"] >= THRESHOLDS["temperature"]["hot"]: temp_txt = random.choice(SEMANTIC_LIBRARY["temperature"]["hot"])
        else: temp_txt = random.choice(SEMANTIC_LIBRARY["temperature"]["normal"])

        if safe_features["elevation"] >= THRESHOLDS["elevation"]["mountain"]:
            terrain_txt = random.choice(SEMANTIC_LIBRARY["terrain"]["mountain"])
            is_mountain = True
        else:
            terrain_txt = random.choice(SEMANTIC_LIBRARY["terrain"]["plain"])
            is_mountain = False

        if safe_features["wind_speed"] >= THRESHOLDS["wind"]["high"]: transport_txt = random.choice(SEMANTIC_LIBRARY["transport"]["high_wind"])
        else: transport_txt = random.choice(SEMANTIC_LIBRARY["transport"]["normal"])

        interp_txt = random.choice(SEMANTIC_LIBRARY["interpretation"][mapped_tier])
        advice_txt = random.choice(SEMANTIC_LIBRARY["advice"][mapped_tier])

        # Executive Summary Custom String
        if mapped_tier in ["Minimal", "Low"]: exec_summary = "Drive relaxed. Current parameters show a highly stable, smooth trip ahead."
        elif mapped_tier in ["Moderate", "Elevated"]: exec_summary = "Drive defensively. Live weather elements are creating minor track slickness."
        else: exec_summary = "🚨 Essential travel only. Serious weather limits are reducing road safety profiles."

        # Compile and parse strings
        formatted_analysis = f"• {weather_txt}\n• {temp_txt}\n• {terrain_txt}\n• {transport_txt}\n• {interp_txt}".format(**safe_features)
        formatted_advice = advice_txt.format(**safe_features)

        # Risk Factor Breakdown
        drivers = []
        if safe_features["rain"] > 0: drivers.append(f"   * 🌧️ Rain Level: {safe_features['rain']:.1f} mm")
        if is_mountain: drivers.append(f"   * ⛰️ Altitude: {safe_features['elevation']:.0f} m")
        if safe_features["wind_speed"] > 0: drivers.append(f"   * 💨 Wind Speed: {safe_features['wind_speed']:.1f} km/h")
        risk_drivers_output = "\n".join(drivers) if drivers else "   * None detected"

        # Final Clean Layout Assembly
        final_markdown_block = (
            f"### 🚗 AI Safety Profile: **{mapped_tier} Risk**\n\n"
            f"**1. Quick Summary:**\n_{exec_summary}_\n\n"
            f"**2. Live Conditions Log:**\n{formatted_analysis}\n\n"
            f"**3. Main Drivers:**\n{risk_drivers_output}\n\n"
            f"**4. What you should do:**\n👉 **{formatted_advice}**"
        )
        return final_markdown_block

    except (KeyError, ValueError) as err:
        logging.error(f"❌ Key rendering break: {err}")
        return "### 📊 Live route metrics logged inside system files."

def generate_combined_summary(risk_tier: str, stress_score: float, emergency_buffer: float) -> str:
    """
    Generates a Combined Intelligence narrative based on weather risk and financial stress.
    """
    if stress_score <= 100 and ("Low" in risk_tier or "Minimal" in risk_tier):
        return f"**🟢 GREEN LIGHT:** Highly optimal trip! You have a safe **{risk_tier}** weather profile paired with a healthy **{stress_score}%** financial stress score. You are good to go!"
        
    elif stress_score > 100 and ("Critical" in risk_tier or "Elevated" in risk_tier):
        return f"**🔴 RED LIGHT:** High Risk Trip! You are facing compounding hazards. The severe **{risk_tier}** weather profile increases the likelihood of delays, which will strain a budget that is already **{stress_score - 100:.1f}% over capacity**."
        
    else:
        return f"**🟡 YELLOW LIGHT:** Mixed conditions. Balance your **{risk_tier}** safety profile with your **{stress_score}%** budget utilization. Ensure your emergency buffer (₹{emergency_buffer:,.0f}) is liquid and accessible."