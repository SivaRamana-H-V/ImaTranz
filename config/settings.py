import os
import json
import logging
import streamlit as st
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_IMAGES = 3
BOX_EXPANSION_INPAINT = 12
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

TEXT_COLOR = 'white'
FILL_COLOR = (0, 0, 0, 255)
INITIAL_FONT_SIZE = 50
FONT_PATH = "assets/fonts/POPPINS-MEDIUM.TTF"
DISTANCE_THRESHOLD = 400
Y_DIFF_THRESHOLD = 2
MIN_FONT_SIZE = 12

class AppConfig:
    PAGE_TITLE = "Amazon Image Translator (GCP)"
    PAGE_ICON = "🚀"
    LAYOUT = "wide"

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        layout="wide",
        page_title="Amazon Image Translator",
        page_icon="🌐"
    )

def load_gcp_credentials():
    """Load GCP credentials from Streamlit secrets or environment."""
    try:
        # Method 1: Streamlit Cloud (secrets.toml)
        creds_dict = st.secrets["gcp_service_account"]
        logger.info("✅ Loaded GCP credentials from Streamlit secrets")
        return service_account.Credentials.from_service_account_info(creds_dict)
    except (KeyError, FileNotFoundError):
        try:
            # Method 2: Render/Cloud Run (environment variable)
            creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
            if creds_json:
                creds_dict = json.loads(creds_json)
                logger.info("✅ Loaded GCP credentials from environment variable")
                return service_account.Credentials.from_service_account_info(creds_dict)
        except Exception as e:
            logger.error(f"❌ Failed to load credentials from environment: {e}")
    
    return None