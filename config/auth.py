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
                creds_dict = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
                gcp_credentials = service_account.Credentials.from_service_account_info(
                    creds_dict)
                return self._create_clients(gcp_credentials, "Render environment variable")

            # 2️⃣ Streamlit secrets fallback
            elif hasattr(st, 'secrets') and "gcp_service_account" in st.secrets:
                gcp_credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                return self._create_clients(gcp_credentials, "Streamlit secrets")

            else:
                raise Exception(
                    "No GCP credentials found. Please set GCP_SERVICE_ACCOUNT_JSON environment variable in Render.")

        except Exception as e:
            logging.error(f"❌ Failed to initialize GCP clients: {e}")
            st.error(
                "🚨 GCP authentication failed. Please check your service account credentials.")
            raise Exception("GCP authentication failed")

    def _create_clients(self, credentials, method):
        self.vision_client = vision.ImageAnnotatorClient(
            credentials=credentials)
        self.translate_client = translate.Client(credentials=credentials)
        logging.info(f"✅ GCP clients initialized using {method}")
        return True
