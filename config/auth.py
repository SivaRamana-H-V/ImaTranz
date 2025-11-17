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
        """Initialize GCP clients with fallback strategies"""
        try:
            # 1️⃣ Streamlit Cloud method
            gcp_credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            return self._create_clients(gcp_credentials, "Streamlit secrets")
            
        except Exception as e:
            logging.warning(f"Streamlit secrets failed: {e}")
            try:
                # 2️⃣ Environment variable fallback
                creds_dict = json.loads(os.environ["GOOGLE_CREDENTIALS"])
                gcp_credentials = service_account.Credentials.from_service_account_info(creds_dict)
                return self._create_clients(gcp_credentials, "environment variable")
                
            except Exception as e2:
                logging.error(f"❌ Failed to initialize GCP clients: {e2}")
                raise Exception("GCP authentication failed")

    def _create_clients(self, credentials, method):
        self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        self.translate_client = translate.Client(credentials=credentials)
        logging.info(f"✅ GCP clients initialized using {method}")
        return True