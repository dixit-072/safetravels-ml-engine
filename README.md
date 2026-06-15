# 🚗 SafeTravels AI: Live Route Risk Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)](https://scikit-learn.org/)

SafeTravels AI is an end-to-end Machine Learning pipeline and interactive transit tracking engine designed to compute real-time environmental risk scores for travelers. By fusing live, localized meteorological telemetry with geographical terrain arrays, this engine handles microclimate shifts and dynamically outputs a deterministic hazard classification profile (0-100%).

### 🚀 Live Production Dashboard
> **[Experience the Live Interactive Dashboard Here](https://dixit-072-safetravels-ml-engine-frontendapp-streamlit-lzmyho.streamlit.app/)**

---

## 🧠 System Architecture Matrix

The platform is engineered using a decoupled microservice framework, isolating the underlying compute/inference nodes from the client-facing display suite to optimize uptime, limit resource consumption, and ensure highly available API routing.

```text
       [ Client Dashboard Interface ] (Streamlit Cloud Instance)
                     │
                     ▼ (Secure HTTP POST Request JSON Payload)
         [ API Router Gateway Hub ] (FastAPI Endpoint Core)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 [ Open-Meteo API Network ]  [ ML Inference Engine Node ]
 (Total Daily Rain Forecast)  (Scikit-Learn Classifier State)
```

1. **The Machine Learning Core (FastAPI Backend):** Deserializes pre-trained classification weights and matching dataframe structural schemas from binary memory dumps (`.pkl`), ingests pipeline requests, and serves low-latency probabilistic evaluations.
2. **The Global Telemetry Bridge:** Integrates with the **Open-Meteo Global Array** via geographic tracking coordinates to capture comprehensive daily atmospheric shifts rather than hyper-localized instant cloudburst snapshots.
3. **The User UI Node (Streamlit Frontend):** Evaluates backend payload signals and maps raw numbers to intuitive hazard gauges, custom alerts, and real-time safe travel recommendations.

---

## 🚀 Core Engineering Features

* **Daily Accumulated Telemetry Filters:** Structured to fetch the entire day's integrated `precipitation_sum` loop instead of standard current rain metrics, ensuring mountain monsoons and localized cloudburst developments are fully represented inside the predictive model.
* **Geographical Microclimate Scaling:** Dynamically parses distinct regional terrain thresholds (e.g., *High-Altitude Mountain Pass* vs. *Semi-Arid Plains*), injecting mathematical elevation penalties directly into the tracking tensor arrays.
* **Fault-Tolerant Parameter Sanitization:** Hardened using robust mathematical fallback pipelines. If the external satellite tracking grid transmits null, unaligned, or missing telemetry variables, the backend instantly intercepts the error code and defaults values cleanly to prevent UI failure.
* **Deterministic Tracking Initialization:** Seeds NumPy statistical generators dynamically based on combined string combinations of user destinations and queried calendar dates, ensuring absolute evaluation reproducibility across multiple sessions.

---

## 📊 End-to-End Execution Sample Data

When a user executes a route calculation for a destination like **Shimla**, the system handles parsing automatically:

| Log Parameter | Incoming Telemetry Value | Backend Handling & Transformation |
| :--- | :--- | :--- |
| **Topography Type** | ⛰️ High-Altitude Mountain Pass | Evaluates localized elevation thresholds (2,100m–2,300m) |
| **Rain Metrics** | `12.44 mm` (Daily Forecast Sum) | Safely mapped to input features to simulate wet road adhesion loss |
| **Wind Velocity** | `12.6 km/h` | Extracted directly via live current API data telemetry frames |
| **Thermal Profile**| `5.0 °C` | Passed to analyze potential black-ice risk indices |
| **Pipeline Risk** | **95.0% Risk Probability Score** | **Output Status: Critical Hazard 🚨 (Deploy Emergency Advisories)** |

---

## 🛠️ Technology Stack Specifications

* **Machine Learning & Analysis Suite:** `Scikit-Learn`, `Pandas`, `NumPy`, `Pickle`
* **Backend Application Routing:** `FastAPI`, `Uvicorn`, `Pydantic`, `Requests`
* **Interactive Frontend Layout:** `Streamlit`
* **Cloud Architecture Hosts:** `Render Cluster` (Backend API Engine), `Streamlit Community Cloud` (UI Web Hub)

---

## 💻 Local Installation & Repository Deployment

Follow these commands to deploy and configure the complete architecture environment locally on your desktop machine:

### 1. Repository Setup & Clone
```bash
git clone [https://github.com/Your-Username/Your-Repository-Name.git](https://github.com/Your-Username/Your-Repository-Name.git)
cd Your-Repository-Name
```

### 2. Environment Activation
```bash
python -m venv venv
# On Windows Operating Systems use:
venv\Scripts\activate
# On macOS / Linux Operating Systems use:
source venv/bin/activate
```

### 3. Dependency Compilation
```bash
pip install -r requirements.txt
```

### 4. Booting the FastAPI Inference Server
```bash
uvicorn routes:app --reload --port 8000
```

### 5. Launching the Streamlit Visual Dashboard (In a Parallel Terminal Instance)
```bash
streamlit run app_streamlit.py
```

---

## 📈 Future Enhancement Map
* **Dynamic Global Geocoding:** Migrating from hardcoded topographical dictionaries to a live OpenStreetMap Geocoding API to dynamically process altitude and coordinates for any city globally.
* **Managed Database Tracking Layers:** Transitioning local session state tracking to a persistent cloud database to capture historical route risk trends over time.

---
*Developed by **Dixit Sharma** — Data Analyst & Machine Learning Engineer. Let's connect on [LinkedIn](https://www.linkedin.com/in/dixit-data-analyst)!*