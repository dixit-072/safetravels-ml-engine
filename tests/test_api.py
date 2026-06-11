import os
import pytest
import requests
from dotenv import load_dotenv  # ✅ Added environment loading agent

# Ingest configuration mappings from your hidden local register file
load_dotenv()

# Securely extract backend network destination with clean standard backup fallback
FASTAPI_HOST = os.getenv("DB_HOST", "127.0.0.1")  # Reuses host configuration safely
BASE_URL = f"http://{FASTAPI_HOST}:8000"

def test_backend_health_endpoint():
    """Verifies the core FastAPI ASGI service layer is online."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend server is currently offline. Start main.py to test.")

def test_prediction_schema_rejection():
    """Asserts that invalid payload envelopes are strictly blocked by Pydantic gates."""
    invalid_payload = {"location_query": "Shimla", "target_date": "not-a-date"}
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_payload, timeout=2)
        assert response.status_code == 422  # Unprocessable Entity (Pydantic validation failure)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend server is currently offline.")