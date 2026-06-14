import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

# Import Streamlit safely so backend tasks don't crash if executed manually
try:
    import streamlit as st
except ImportError:
    st = None

# Ingest configuration mappings from your hidden local register file
load_dotenv()

# ============================================
# GOOGLE SHEETS CONFIGURATION (CLOUD LAYER)
# ============================================
# Read from Streamlit secrets if available, fallback to environment variables
if st and hasattr(st, "secrets") and "SPREADSHEET_NAME" in st.secrets:
    SPREADSHEET_NAME = st.secrets.get("SPREADSHEET_NAME")
    WORKSHEET_NAME = st.secrets.get("GOOGLE_SHEET_TAB")
else:
    SPREADSHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "SafeTravels_Cloud_Logs")
    WORKSHEET_NAME = os.getenv("GOOGLE_SHEET_TAB", "prediction_responses")

GOOGLE_CREDS_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google_creds.json")

# Backup CSV path (Kept intact exactly as you designed it)
BACKUP_DIR = Path("data")
BACKUP_DIR.mkdir(exist_ok=True)
BACKUP_FILE = BACKUP_DIR / "predict_api_responses_backup.csv"


def _normalize_private_key(raw_value: str) -> str:
    """Normalize a Streamlit/ENV private key string into valid PEM format."""
    if not raw_value:
        return raw_value

    pem_text = raw_value.strip()
    pem_text = pem_text.replace("\r\n", "\n").replace("\\n", "\n")
    pem_text = pem_text.replace(" \n", "\n").replace("\n ", "\n")

    if "-----BEGIN PRIVATE KEY-----" not in pem_text:
        pem_text = "-----BEGIN PRIVATE KEY-----\n" + pem_text
    if "-----END PRIVATE KEY-----" not in pem_text:
        pem_text = pem_text + "\n-----END PRIVATE KEY-----"

    pem_text = pem_text.replace("-----BEGIN PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----\n")
    pem_text = pem_text.replace("-----END PRIVATE KEY-----", "\n-----END PRIVATE KEY-----")

    lines = [line.strip() for line in pem_text.splitlines() if line.strip()]
    normalized_lines = []
    for line in lines:
        if line.startswith("-----"):
            normalized_lines.append(line)
        else:
            normalized_lines.append("".join(ch for ch in line if ch.isalnum() or ch in "+/="))

    normalized = "\n".join(normalized_lines)
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def _load_service_account_json(raw_json: str):
    if not raw_json:
        return None
    candidate = raw_json.strip().replace("\r\n", "\n").replace("\\n", "\n")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _get_service_account_info():
    raw_json = None
    if st and hasattr(st, "secrets") and st.secrets:
        raw_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or st.secrets.get("SERVICE_ACCOUNT_JSON")
    raw_json = raw_json or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("SERVICE_ACCOUNT_JSON")

    creds = _load_service_account_json(raw_json)
    if isinstance(creds, dict):
        if "private_key" in creds:
            creds["private_key"] = _normalize_private_key(creds["private_key"])
        return creds

    def secret(key, fallback=None):
        if st and hasattr(st, "secrets") and st.secrets:
            return st.secrets.get(key, os.getenv(key, fallback))
        return os.getenv(key, fallback)

    private_key = secret("GCP_PRIVATE_KEY") or secret("GOOGLE_PRIVATE_KEY")
    if not private_key:
        return None

    return {
        "type": secret("GCP_TYPE", "service_account"),
        "project_id": secret("GCP_PROJECT_ID"),
        "private_key_id": secret("GCP_PRIVATE_KEY_ID"),
        "private_key": _normalize_private_key(private_key),
        "client_email": secret("GCP_CLIENT_EMAIL"),
        "client_id": secret("GCP_CLIENT_ID"),
        "auth_uri": secret("GCP_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": secret("GCP_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": secret("GCP_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
        "client_x509_cert_url": secret("GCP_CLIENT_X509_CERT_URL"),
        "universe_domain": secret("GCP_UNIVERSE_DOMAIN")
    }


class GoogleSheetsDatabase:
    """Google Sheets Cloud Data Writer - Production-Ready Replacement for MySQL"""
    
    def __init__(self):
        self.client = None
        self.sheet = None
        self.connect()

    def connect(self):
        """Establishes a secure connection handshake with the Google Sheets Cloud API."""
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 🟢 1. LOCAL MACHINE FALLBACK: Check if running locally on your laptop
        if os.path.exists(GOOGLE_CREDS_FILE):
            try:
                creds = Credentials.from_service_account_file(GOOGLE_CREDS_FILE, scopes=scopes)
                self.client = gspread.authorize(creds)
                self.sheet = self.client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
                print(f"✓ Connected via Local JSON - Spreadsheet: '{SPREADSHEET_NAME}'")
                return
            except Exception as e:
                print(f"✗ Local JSON Handshake Failed: {e}")

        # 🟢 2. LIVE PRODUCTION ENGINE: Pull direct flat variables on Streamlit Cloud
        creds_dict = _get_service_account_info()
        if creds_dict:
            try:
                creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
                self.client = gspread.authorize(creds)
                self.sheet = self.client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
                print(f"✓ Connected via Streamlit Cloud Secrets - Spreadsheet: '{SPREADSHEET_NAME}'")
                return
            except Exception as e:
                print(f"✗ Streamlit Cloud Secrets Handshake Failed: {e}")

        print("✗ Connection Failed: No valid credentials sources discovered.")
        self.client = None
        self.sheet = None

    def insert_prediction(self, response_data: Dict[str, Any], location_query: str):
        """Insert prediction response row directly into the Google Sheet columns (A to M)"""
        if not self.sheet:
            print("⚠ Skipping cloud storage insert - no connection established")
            return False
            
        try:
            # Compile row elements exactly matching your spreadsheet headers
            row_to_append = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),                                  # A: timestamp
                location_query,                                                                # B: location_query
                response_data.get('resolved_name', 'N/A'),                                    # C: resolved_name
                float(response_data.get('latitude', 0.0) or 0.0),                             # D: latitude
                float(response_data.get('longitude', 0.0) or 0.0),                            # E: longitude
                float(response_data.get('predicted_hazard_score', 0.0) or 0.0),                # F: predicted_hazard_score
                response_data.get('risk_category', 'Unassigned'),                             # G: risk_category
                response_data.get('destination_type', 'General'),                             # H: destination_type
                response_data.get('destination_description', 'N/A'),                          # I: destination_description
                response_data.get('model_version', 'XGBoost.v2.6'),                           # J: model_version
                response_data.get('forecast_date', datetime.now().strftime("%Y-%m-%d")),      # K: forecast_date
                json.dumps(response_data.get('processed_features', {})),                       # L: processed_features
                "SUCCESS"                                                                      # M: status flag
            ]
            
            # Stream row to Google Sheets cloud matrix
            self.sheet.append_row(row_to_append)
            print(f"✓ Prediction stored in Cloud Sheet - {response_data.get('resolved_name')}")
            return True
        except Exception as e:
            print(f"✗ Google Cloud insert failed: {e}")
            return False


def backup_to_csv(response_data: Dict[str, Any], location_query: str):
    """Store API response as backup CSV (Your original code preserved)"""
    try:
        flat_data = {
            'timestamp': datetime.now().isoformat(),
            'location_query': location_query,
            'resolved_name': response_data.get('resolved_name'),
            'latitude': response_data.get('latitude'),
            'longitude': response_data.get('longitude'),
            'predicted_hazard_score': response_data.get('predicted_hazard_score'),
            'risk_category': response_data.get('risk_category'),
            'destination_type': response_data.get('destination_type'),
            'destination_description': response_data.get('destination_description'),
            'model_version': response_data.get('model_version'),
            'forecast_date': response_data.get('forecast_date'),
            'status': response_data.get('status'),
            'full_response_json': json.dumps(response_data)
        }
        
        df = pd.DataFrame([flat_data])
        
        if BACKUP_FILE.exists():
            existing_df = pd.read_csv(BACKUP_FILE)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(BACKUP_FILE, index=False)
        print(f"✓ Response backed up to CSV - {BACKUP_FILE}")
        return True
    except Exception as e:
        print(f"✗ CSV backup failed: {e}")
        return False


def entry_store(response_data: Dict[str, Any], location_query: str):
    """Main entry point to store API predictions to both Google Cloud Sheets and local CSV backup"""
    print("\n" + "="*50)
    print("📊 STORING PREDICTION DATA (CLOUD ENGINE WORKFLOW)")
    print("="*50)
    
    # Step 1: Store to CSV backup (Your trusted fallback remains alive!)
    print("📁 Step 1: Saving to CSV backup...")
    backup_to_csv(response_data, location_query)
    
    # Step 2: Store to Google Cloud Sheets 
    print("☁️ Step 2: Saving to Google Cloud Sheets Server...")
    db = GoogleSheetsDatabase()
    db.insert_prediction(response_data, location_query)
    
    print("="*50 + "\n")