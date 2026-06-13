import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
import json

# Ingest configuration mappings from your hidden local register file
load_dotenv()

# ============================================
# GOOGLE SHEETS CONFIGURATION (CLOUD LAYER)
# ============================================
GOOGLE_CREDS_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google_creds.json")
SPREADSHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "SafeTravels_Cloud_Logs")
WORKSHEET_NAME = os.getenv("GOOGLE_SHEET_TAB", "prediction_responses")

# Backup CSV path (Kept intact exactly as you designed it)
BACKUP_DIR = Path("data")
BACKUP_DIR.mkdir(exist_ok=True)
BACKUP_FILE = BACKUP_DIR / "predict_api_responses_backup.csv"


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
        try:
            creds = Credentials.from_service_account_file(GOOGLE_CREDS_FILE, scopes=scopes)
            self.client = gspread.authorize(creds)
            # Open the targeted spreadsheet matrices
            self.sheet = self.client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
            print(f"✓ Connected to Google Cloud Engine - Spreadsheet: '{SPREADSHEET_NAME}'")
        except Exception as e:
            print(f"✗ Google Cloud Connection Failed: {e}")
            print("💡 Make sure your 'google_creds.json' file is present and shared with your sheet email editor address!")
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
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),                  # A: timestamp
                location_query,                                               # B: location_query
                response_data.get('resolved_name', 'N/A'),                    # C: resolved_name
                float(response_data.get('latitude', 0.0) or 0.0),             # D: latitude
                float(response_data.get('longitude', 0.0) or 0.0),            # E: longitude
                float(response_data.get('predicted_hazard_score', 0.0) or 0.0),# F: predicted_hazard_score
                response_data.get('risk_category', 'Unassigned'),             # G: risk_category
                response_data.get('destination_type', 'General'),             # H: destination_type
                response_data.get('destination_description', 'N/A'),          # I: destination_description
                response_data.get('model_version', 'XGBoost.v2.6'),           # J: model_version
                response_data.get('forecast_date', datetime.now().strftime("%Y-%m-%d")), # K: forecast_date
                json.dumps(response_data.get('processed_features', {})),       # L: processed_features
                "SUCCESS"                                                      # M: status flag
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