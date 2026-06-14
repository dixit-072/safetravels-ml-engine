import os
import json
import base64
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

try:
    import streamlit as st
except ImportError:
    st = None

load_dotenv()

def _get_service_account_info():
    """Decodes a clean Base64 string directly in memory."""
    try:
        # Pull from Streamlit secrets, or fallback to ENV
        if st and hasattr(st, "secrets") and "GCP_CREDS_B64" in st.secrets:
            b64_token = st.secrets["GCP_CREDS_B64"]
        else:
            b64_token = os.getenv("GCP_CREDS_B64", "")

        if not b64_token:
            return None
        
        # Clean and decode
        clean_token = str(b64_token).strip().replace("\n", "").replace(" ", "").replace('"', '').replace("'", "")
        decoded_json = base64.b64decode(clean_token).decode("utf-8")
        return json.loads(decoded_json)
    except Exception as e:
        print(f"✗ Decoding Failed: {e}")
        return None

class GoogleSheetsDatabase:
    def __init__(self):
        self.client = None
        self.sheet = None
        self.connect()

    def connect(self):
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds_dict = _get_service_account_info()
        if creds_dict:
            try:
                creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
                self.client = gspread.authorize(creds)
                # Use secrets for spreadsheet/worksheet names
                s_name = st.secrets.get("SPREADSHEET_NAME", "SafeTravels_Cloud_Logs") if st else "SafeTravels_Cloud_Logs"
                w_name = st.secrets.get("GOOGLE_SHEET_TAB", "prediction_responses") if st else "prediction_responses"
                self.sheet = self.client.open(s_name).worksheet(w_name)
            except Exception as e:
                print(f"✗ Connection Failed: {e}")

    def insert_prediction(self, data, query):
        if not self.sheet: return False
        try:
            row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query, data.get('resolved_name', 'N/A'), 
                   float(data.get('latitude', 0)), float(data.get('longitude', 0)), 
                   float(data.get('predicted_hazard_score', 0)), data.get('risk_category', 'Unassigned'),
                   data.get('destination_type', 'General'), data.get('destination_description', 'N/A'),
                   data.get('model_version', 'v2.6'), data.get('forecast_date', datetime.now().strftime("%Y-%m-%d")),
                   json.dumps(data.get('processed_features', {})), "SUCCESS"]
            self.sheet.append_row(row)
            return True
        except Exception: return False