# 🚗 SafeTravels AI — Route Risk Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)](https://scikit-learn.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen)](https://dixit-072-safetravels-ml-engine-frontendapp-streamlit-lzmyho.streamlit.app/)

**SafeTravels AI** is a full-stack machine learning project that predicts real-time travel risk scores based on live weather data and regional terrain. A user enters a destination, and the system fetches live weather from the Open-Meteo API, processes it through a trained ML model, and returns a hazard probability (0–100%) with travel recommendations.

> 🔗 **[Try the Live App Here](https://dixit-072-safetravels-ml-engine-frontendapp-streamlit-lzmyho.streamlit.app/)**

> ⚠️ **Note on Cold Start:** The backend is hosted on Render's free tier. If the app hasn't been used for 15+ minutes, the first prediction may take 30–45 seconds to load while the server wakes up. Subsequent requests are fast.

---

## 📸 Screenshots

> *(Add a screenshot of the Streamlit dashboard here)*

---

## 🧠 How It Works

The project uses a **decoupled architecture** — the ML backend and the frontend run as separate services:

```
[ Streamlit Frontend ]  (Streamlit Community Cloud)
         │
         ▼  HTTP POST (JSON payload)
[ FastAPI Backend ]     (Render Free Tier)
         │
    ┌────┴────┐
    ▼          ▼
[ Open-Meteo  ] [ ML Model ]
  Weather API    (Scikit-Learn .pkl)
```

**Step-by-step flow:**
1. User enters a destination city in the Streamlit app
2. Backend fetches **live daily weather** (precipitation, wind, temperature) from Open-Meteo API
3. The terrain type for the city (e.g., mountain pass vs plains) is looked up and used as a feature
4. The trained Scikit-Learn classifier returns a **risk probability score**
5. Streamlit displays the score with a gauge, color-coded alert, and travel recommendation

---

## 📊 Sample Prediction Output

For a destination like **Shimla** on a rainy day:

| Feature | Value | Notes |
|---|---|---|
| Terrain Type | ⛰️ High-Altitude Mountain Pass | Elevation ~2,200m |
| Daily Rainfall | 12.44 mm | Fetched from Open-Meteo |
| Wind Speed | 12.6 km/h | Live current data |
| Temperature | 5.0°C | Black-ice risk range |
| **Risk Score** | **95% 🚨** | **Critical — avoid travel** |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML / Data | Scikit-Learn, Pandas, NumPy, Pickle |
| Backend API | FastAPI, Uvicorn, Pydantic, Requests |
| Frontend | Streamlit |
| Weather Data | Open-Meteo API (free, no key needed) |
| Hosting | Render (backend), Streamlit Community Cloud (frontend) |

---

## 📁 Project Structure

```
safetravels-ml-engine/
├── backend/              # FastAPI app — ML inference + weather fetching
├── frontend/             # Streamlit app — UI and API calls
├── notebooks/            # EDA, feature engineering, model training
├── models/               # Saved .pkl model and scaler files
├── data/                 # Training dataset
├── src/                  # Shared utility functions
├── analysis/             # Exploratory analysis scripts
├── store_user_query/     # Logs user queries (for future analytics)
├── tests/                # Unit tests
├── requirements.txt
├── run_pipeline.bat      # One-click local setup (Windows)
└── README.md
```

---

## 💻 Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/dixit-072/safetravels-ml-engine.git
cd safetravels-ml-engine
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```bash
cd backend
uvicorn routes:app --reload --port 8000
```

Backend will be running at: `http://localhost:8000`

### 5. Start the Streamlit frontend (in a new terminal)

```bash
cd frontend
streamlit run app_streamlit.py
```

App will open at: `http://localhost:8501`

---

## ✅ Key Engineering Decisions

- **Daily precipitation instead of current rain** — mountain cloudbursts build up over hours. Using `precipitation_sum` (total daily rainfall) gives a more accurate risk signal than a snapshot of current rain.
- **Terrain-based elevation penalties** — flat plains and mountain passes behave differently. The model uses terrain type as a categorical feature to account for this.
- **Deterministic predictions for same inputs** — NumPy seeds are set based on city name + date, so the same query always returns the same result, useful for testing and reproducibility.
- **Null-safe weather parsing** — if Open-Meteo returns missing fields, the backend falls back to safe defaults instead of crashing the UI.

---

## 🗺️ Planned Improvements

- **Live geocoding** — replace the hardcoded city-terrain dictionary with OpenStreetMap/Nominatim API to support any global city
- **Historical risk tracking** — store past queries in a database to show risk trends over time
- **Backend migration into Streamlit** — eliminate the separate FastAPI server to remove the cold start problem entirely (free fix)

---

## 👤 About

Built by **Dixit Sharma** — Data Analyst & ML Engineer based in Himachal Pradesh, India.


