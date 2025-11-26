import os
import json
import logging
import streamlit as st
from google.cloud import vision
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)


class GCPAuth:
    def __init__(self):
        self.vision_client = None
        self.translate_client = None

    def initialize_clients(self):
        """Initialize GCP clients for Render deployment"""
        try:
            # 1️⃣ Render Environment Variable (Primary)
            if "GCP_SERVICE_ACCOUNT_JSON" in os.environ:
                creds_json = os.environ["GCP_SERVICE_ACCOUNT_JSON"]
                creds_dict = json.loads(creds_json)
                gcp_credentials = service_account.Credentials.from_service_account_info(
                    creds_dict)
                return self._create_clients(gcp_credentials, "Render environment variable")

            # 2️⃣ Streamlit secrets fallback (for local development)
            elif hasattr(st, 'secrets') and "gcp_service_account" in st.secrets:
                gcp_credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                return self._create_clients(gcp_credentials, "Streamlit secrets")

            else:
                st.error("""
                🔐 GCP Credentials Required!
                
                Please set up your Google Cloud service account:
                1. Go to Google Cloud Console → IAM & Admin → Service Accounts
                2. Create a service account with Vision API & Translate API access
                3. Generate and download JSON key file
                4. In Render dashboard, add environment variable:
                   - Key: GCP_SERVICE_ACCOUNT_JSON
                   - Value: Paste entire JSON content
                """)
                return False

        except Exception as e:
            logging.error(f"❌ Failed to initialize GCP clients: {e}")
            st.error(f"🚨 GCP authentication failed: {e}")
            return False

    def _create_clients(self, credentials, method):
        self.vision_client = vision.ImageAnnotatorClient(
            credentials=credentials)
        self.translate_client = translate.Client(credentials=credentials)
        logging.info(f"✅ GCP clients initialized using {method}")
        return True
