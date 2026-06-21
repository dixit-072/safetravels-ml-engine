import streamlit as st
import plotly.express as px
from datetime import datetime  
from summary import generate_combined_summary
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.budget_engine import CITY_DB
from backend.api.budget_engine import calculate_dynamic_budget

def render_budget_tab(location_query: str, target_date: str, log_function=None):
    st.header("💰 Intelligent Budget Forecaster")
    st.caption(f"Generating dynamic financial forecast for **{location_query}** starting **{target_date}**.")
    st.markdown("---")

    # ==========================================================
    # SECTION 1: THE INPUT ENGINE (Booking Card Layout)
    # ==========================================================
    with st.container(border=True):
        st.markdown("#### 🎛️ Trip Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            num_stays = st.number_input("🌙 Nights", min_value=0, max_value=14, value=2)
        
        with col2:
            num_people = st.number_input("👥 Travelers", min_value=1, max_value=20, value=2)
            
        with col3:
            travel_style = st.selectbox("🎒 Travel Style", ["Backpacker", "Standard", "Luxury"], index=1)
            
        with col4:
            if travel_style == "Backpacker":
                dynamic_transports = ["Train (Sleeper/3AC)", "Bus (Non-AC)", "Bus (AC Volvo)"]
                default_index = 0
            elif travel_style == "Luxury":
                dynamic_transports = ["Flight", "Personal Car / Taxi"]
                default_index = 0
            else:
                dynamic_transports = ["Flight", "Train (Sleeper/3AC)", "Bus (AC Volvo)", "Personal Car / Taxi"]
                default_index = 2

            city_name_clean = location_query.strip().title()
            city_data = CITY_DB.get(city_name_clean, {"Has_Airport": 1}) 
            if city_data.get("Has_Airport", 1) == 0:
                if "Flight" in dynamic_transports:
                    dynamic_transports.remove("Flight")
                    default_index = 0
            
            transport_mode = st.selectbox("🚆 Transport", dynamic_transports, index=min(default_index, len(dynamic_transports)-1))
            if city_data.get("Has_Airport", 1) == 0:
                st.caption(f"*(No airport in {location_query})*")

        st.markdown("---") 
        pref_col1, pref_col2, pref_col3 = st.columns([2, 1, 1])
        
        with pref_col1:
            selected_activities = st.multiselect(
                "🧗 Add Experiences", 
                ["River Rafting / Adventure Sports", "Local Sightseeing Tour (Cab)", "Fine Dining / Special Meal", "Museums / Monument Entry", "Spa / Wellness Session"],
                placeholder="Select planned activities..."
            )
            
        with pref_col2:
            max_budget = st.number_input("💸 Max Budget (INR)", min_value=5000, max_value=500000, value=30000, step=1000)
            
        with pref_col3:
            st.write("") 
            is_round_trip = st.toggle("🔄 Round Trip", value=True)

        st.write("")
        trigger_budget = st.button("✨ Generate AI Financial Forecast", type="primary", use_container_width=True)

    # ==========================================================
    # SECTION 2: RESULTS ENGINE (Dashboard Layout)
    # ==========================================================
    if trigger_budget:
        user_inputs = {
            "location_query": location_query,
            "target_date": target_date,
            "num_stays": num_stays,
            "num_people": num_people,
            "travel_style": travel_style,
            "transport_mode": transport_mode,
            "is_round_trip": is_round_trip, 
            "max_budget": float(max_budget),
            "selected_activities": selected_activities
        }

        telemetry = st.session_state.get("latest_telemetry", {})

        with st.spinner("🤖 AI is calculating dynamic terrain costs and checking weather taxes..."):
            try:
                data = calculate_dynamic_budget(user_inputs, telemetry)
                bd = data["breakdown"]
                stress_score = data["financial_stress_score"]

                st.markdown("<br>", unsafe_allow_html=True)

                # 🗂️ CARD 1: EXECUTIVE SUMMARY
                with st.container(border=True):
                    st.subheader("📋 Executive Summary")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Forecasted Cost", f"₹{data['estimated_total']:,.0f}")
                    c2.metric("Financial Stress Score", f"{stress_score}%")
                    
                    if stress_score <= 100:
                        c3.markdown("**Budget Status:**<br>🟢 Healthy", unsafe_allow_html=True)
                        st.success(data["budget_summary"])
                    else:
                        c3.markdown("**Budget Status:**<br>🔴 Over Budget", unsafe_allow_html=True)
                        st.error(data["budget_summary"])

                # 🗂️ CARD 2: DETAILED BREAKDOWN
                with st.container(border=True):
                    st.subheader("📊 Detailed Expense Breakdown")
                    
                    if data.get("applied_taxes"):
                        st.info("💡 **Pricing Insights:** " + " | ".join(data["applied_taxes"]))

                    res_col1, res_col2 = st.columns([1, 1.2])
                    
                    with res_col1:
                        st.markdown("#### 🛡️ Core Travel Costs")
                        c1, c2 = st.columns(2)
                        c1.write("🏨 Accommodation")
                        c2.write(f"**₹{bd['accommodation']:,.0f}**")
                        
                        c1, c2 = st.columns(2)
                        c1.write("🍔 Food & Dining")
                        c2.write(f"**₹{bd['food']:,.0f}**")
                        
                        c1, c2 = st.columns(2)
                        c1.write("🚕 Transport")
                        c2.write(f"**₹{bd['transport'] + bd['local_commute']:,.0f}**")
                        st.caption(f"• *Transit:* ₹{bd['transport']:,.0f} | *Local:* ₹{bd['local_commute']:,.0f}")
                        
                        st.markdown("---")
                        
                        st.markdown("#### 🌟 Upgrades & Safety Net")
                        c1, c2 = st.columns(2)
                        c1.write("🎟️ Activities")
                        c2.write(f"**₹{bd['activities']:,.0f}**")
                        
                        c1, c2 = st.columns(2)
                        c1.write("🏥 Emergency Buffer")
                        c2.write(f"**₹{bd['emergency_buffer']:,.0f}**")

                    with res_col2:
                        st.markdown("**Cost Distribution**")
                        labels = list(bd.keys())
                        values = list(bd.values())
                        clean_labels = [l.replace('_', ' ').title() for l in labels]
                        
                        fig = px.pie(
                            names=clean_labels, 
                            values=values, 
                            hole=0.45,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_traces(textposition='inside', textinfo='label+percent')
                        fig.update_layout(
                            margin=dict(t=10, b=10, l=0, r=0), 
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # 🗂️ CARD 3: ROUTE INTELLIGENCE
                with st.container(border=True):
                    st.subheader("🧠 Route Intelligence")
                    
                    has_risk_data = "prediction_history" in st.session_state and len(st.session_state.prediction_history) > 0
                    
                    if has_risk_data:
                        latest_risk = st.session_state.prediction_history[-1]
                        risk_tier = latest_risk["Safety Status Category"]
                        
                        st.info(f"🛡️ **Safety Profile:** Based on your recent check for **{location_query}**, the route holds a **{risk_tier}** risk profile.")
                        
                        combined_text = generate_combined_summary(risk_tier, stress_score, bd['emergency_buffer'])
                        if "GREEN" in combined_text:
                            st.success(combined_text)
                        elif "RED" in combined_text:
                            st.error(combined_text)
                        else:
                            st.warning(combined_text)
                    else:
                        st.warning("⚠️ Run the 'Route Safety Profile' in the previous tab to unlock the Weather and Combined intelligence summaries!")

                # ==========================================================
                # SECTION 3: CLOUD LOGGING
                # ==========================================================
                if log_function:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_data = [
                        timestamp, location_query, target_date, num_stays,  
                        num_people, travel_style, transport_mode, 
                        float(max_budget), float(data["estimated_total"]), 
                        float(stress_score), data["budget_status"]
                    ]
                    log_function(log_data)

            except Exception as e:
                st.error(f"🚨 Calculation Error: {e}")