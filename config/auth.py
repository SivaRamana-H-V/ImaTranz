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
        """Initialize GCP clients with detailed error handling"""
        try:
            # Method 1: Render Environment Variable
            if "GCP_SERVICE_ACCOUNT_JSON" in os.environ:
                logging.info("Found GCP_SERVICE_ACCOUNT_JSON in environment")
                creds_json = os.environ["GCP_SERVICE_ACCOUNT_JSON"]
                try:
                    creds_dict = json.loads(creds_json)
                    gcp_credentials = service_account.Credentials.from_service_account_info(
                        creds_dict)
                    return self._create_clients(gcp_credentials, "Render environment variable")
                except json.JSONDecodeError as e:
                    logging.error(
                        f"Invalid JSON in GCP_SERVICE_ACCOUNT_JSON: {e}")
                    return False

            # Method 2: Streamlit Secrets (for local development)
            elif hasattr(st, 'secrets') and "gcp_service_account" in st.secrets:
                logging.info("Found gcp_service_account in Streamlit secrets")
                gcp_credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                return self._create_clients(gcp_credentials, "Streamlit secrets")

            else:
                logging.error(
                    "No GCP credentials found in environment variables or secrets")
                st.error("""
                🔐 **GCP Credentials Not Found**
                
                Please add your GCP service account credentials:
                
                1. **In Render Dashboard:**
                   - Go to Environment tab
                   - Add: GCP_SERVICE_ACCOUNT_JSON = [your-json-here]
                
                2. **Required APIs:**
                   - Vision API
                   - Cloud Translation API
                """)
                return False

        except Exception as e:
            logging.error(f"❌ GCP initialization failed: {e}")
            st.error(f"GCP initialization failed: {str(e)}")
            return False

    def _create_clients(self, credentials, method):
        try:
            self.vision_client = vision.ImageAnnotatorClient(
                credentials=credentials)
            self.translate_client = translate.Client(credentials=credentials)
            logging.info(f"✅ GCP clients initialized using {method}")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to create GCP clients: {e}")
            return False
