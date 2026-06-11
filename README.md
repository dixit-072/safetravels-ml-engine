# 🚗 SafeTravels: Smart Route Risk Advisor & Predictive ML Engine
> **Enterprise-Grade Decoupled Route Hazard Inference Microservice & Real-Time Analytics Dashboard**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.30%2B-FF4B4B?logo=streamlit)
![MySQL](https://img.shields.io/badge/MySQL-Relational-4479A1?logo=mysql)
![XGBoost](https://img.shields.io/badge/ML--Model-XGBoost%20%7C%20RandomForest-F60?logo=scikit-learn)

SafeTravels is a production-grade machine learning application designed to assess, predict, and monitor environmental, logistical, and structural travel route risks across major tourist corridors in India. Built with a decoupled, microservice-first architecture, the system coordinates real-time satellite meteorology telemetry, geographic spatial profiles, regional infrastructure constraints, and calendar-driven tourist crowd accumulations to provide instant route hazard inference scores.

---

## 🏗️ System Architecture & Separation of Concerns

The application is structured into completely independent operational layers, enforcing industry-standard software engineering boundaries:

1. **Analytical Pipeline Data Tier (`src/`)**: Offline background engineering pipeline that harvests years of historical telemetry, processes holiday calendars, compiles regional transport complexities, and transforms data matrices.
2. **ML Modeling & Tournament Engine (`src/`)**: Trains candidate algorithms (Linear Regression, Random Forest, XGBoost), executes 5-fold cross-validation to enforce structural alignment, audits chronological out-of-sample data-leakage, and locks champion model binaries.
3. **Core Backend Microservice ASGI Layer (`backend/`)**: High-performance FastAPI engine that initializes and exposes endpoints (`/health` and `/predict`), processes Pydantic validation gates, and orchestrates live third-party spatial-weather API telemetry.
4. **Relational Storage Core Layer (`store_user_query/`)**: Intercepts inbound API predictions via an asynchronous execution gate to store transactional payload rows inside local MySQL instances alongside flattening CSV transaction ledger fallbacks.
5. **Interactive Portal View Tier (`frontend/`)**: High-fidelity Streamlit user interface mapping real-time connection status trackers, parameter simulation overrides, line trend variations, and scatter model variance charts ($R^2$ metrics).

---

## 📂 Repository Layout Directory Structure

```text
safetravels-ml-engine/
│
├── .gitignore                   # Excludes environments (.env, venv/), large caches, and binaries
├── README.md                    # Project documentation homepage
├── requirements.txt             # Unified Python third-party dependencies directory index
├── run_pipeline.bat             # Automated Windows multi-stage orchestration control hub
│
├── analysis/                    # Pre-computed regional risk profiles and UI analytical arrays
│   ├── city_risk_share_profiles.csv
│   └── risk_attribution_dashboard.csv
│
├── backend/                     # Live inference serving network microservice engine
│   ├── main.py                  # FastAPI ASGI framework boot controller script
│   └── api/
│       ├── routes.py            # API request distribution routers and endpoint drivers
│       ├── schemas.py           # Pydantic rigorous type-checking validation models
│       └── services.py          # On-demand geocoding coordinate & telemetry loaders
│
├── data/                        # Clean analytical target lookups and feature configurations
│   ├── holidify.csv             # Primary baseline source geographic metrics sheet
│   ├── location_registry.csv    # Fixed 10-city target master baseline profiling profiles
│   └── master_feature_table_with_hazards.csv  # Final integrated ML modeling training array
│
├── frontend/                    # Responsive multi-view dashboard layout UI layer
│   └── app_streamlit.py         # Polished interactive reporting system portal
│
├── models/                      # Frozen predictive champion binaries
│   ├── model_feature_schema.pkl # Structural column constraint sorting arrays
│   └── travel_risk_model.pkl    # Serialized ML model algorithmic weight vectors
│
├── src/                         # Decoupled batch processing data science modules
│   ├── pipeline_data_prep.py    # Historical data harvester and matrix builder
│   └── pipeline_train_model.py  # Model tournament validator and chronological stress auditor
│
└── tests/                       # Automated software integration test scripts
    └── test_api.py              # Pytest backend endpoint verification and Pydantic validation suite