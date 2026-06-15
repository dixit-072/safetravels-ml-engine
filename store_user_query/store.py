import os
import json
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

def get_creds():
    """Smart loader: checks Streamlit Secrets -> .env -> google_creds.json."""
    
    # 1. Cloud Mode (Streamlit Secrets)
    if hasattr(st, "secrets") and "GCP_PROJECT_ID" in st.secrets:
        raw_key = st.secrets["GCP_PRIVATE_KEY"]
        # Fix formatting for the cloud
        if not raw_key.startswith("-----BEGIN PRIVATE KEY-----"):
            raw_key = "-----BEGIN PRIVATE KEY-----\n" + raw_key
        
        return {
            "type": st.secrets["GCP_TYPE"],
            "project_id": st.secrets["GCP_PROJECT_ID"],
            "private_key_id": st.secrets["GCP_PRIVATE_KEY_ID"],
            "private_key": raw_key.replace("\\n", "\n"),
            "client_email": st.secrets["GCP_CLIENT_EMAIL"],
            "client_id": st.secrets["GCP_CLIENT_ID"],
            "auth_uri": st.secrets["GCP_AUTH_URI"],
            "token_uri": st.secrets["GCP_TOKEN_URI"],
            "auth_provider_x509_cert_url": st.secrets["GCP_AUTH_PROVIDER_X509_CERT_URL"],
            "client_x509_cert_url": st.secrets["GCP_CLIENT_X509_CERT_URL"]
        }

    # 2. Local Mode (.env)
    if os.getenv("GCP_PROJECT_ID"):
        return {
            "type": os.getenv("GCP_TYPE"),
            "project_id": os.getenv("GCP_PROJECT_ID"),
            "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GCP_PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("GCP_CLIENT_EMAIL"),
            "client_id": os.getenv("GCP_CLIENT_ID"),
            "auth_uri": os.getenv("GCP_AUTH_URI"),
            "token_uri": os.getenv("GCP_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("GCP_CLIENT_X509_CERT_URL")
        }

    # 3. File fallback
    if os.path.exists("google_creds.json"):
        with open("google_creds.json") as f:
            return json.load(f)
            
    return None

class GoogleSheetsDatabase:
    def __init__(self):
        self.client = None
        self.sheet = None
        self.connect()

    def connect(self):
        creds_dict = get_creds()
        if creds_dict:
            try:
                scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
                creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
                self.client = gspread.authorize(creds)
                
                s_name = st.secrets.get("SPREADSHEET_NAME", "SafeTravels_Cloud_Logs") if st else "SafeTravels_Cloud_Logs"
                w_name = st.secrets.get("GOOGLE_SHEET_TAB", "prediction_responses") if st else "prediction_responses"
                
                self.sheet = self.client.open(s_name).worksheet(w_name)
            except Exception as e:
                print(f"✗ Connection failed: {e}")

    def insert_prediction(self, data, query):
        if not self.sheet: return False
        try:
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query, data.get('resolved_name', 'N/A'), 
                float(data.get('latitude', 0)), float(data.get('longitude', 0)), 
                float(data.get('predicted_hazard_score', 0)), data.get('risk_category', 'Unassigned'),
                data.get('destination_type', 'General'), data.get('destination_description', 'N/A'),
                data.get('model_version', 'v2.6'), data.get('forecast_date', datetime.now().strftime("%Y-%m-%d")),
                json.dumps(data.get('processed_features', {})), "SUCCESS"
            ]
            self.sheet.append_row(row)
            return True
        except Exception: return False