import os
import json
import logging
from google.cloud import vision
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCPAuth:
    def __init__(self):
        self.vision_client = None
        self.translate_client = None
        self.initialization_error = None  # Track errors

    def initialize_clients(self):
        """Initialize GCP services - returns True/False with error tracking"""
        try:
            # 1. Check for the variable (Cloud Run / Secret Manager)
            raw_data = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")

            if not raw_data:
                self.initialization_error = "GCP_SERVICE_ACCOUNT_JSON environment variable not found"
                logger.error(self.initialization_error)
                return False

            # Check if it's a file path (Local) or actual JSON (Cloud)
            if raw_data.strip().startswith('{'):
                # It's actual JSON text - use .from_service_account_info()
                creds_dict = json.loads(raw_data)
                gcp_credentials = service_account.Credentials.from_service_account_info(
                    creds_dict)
            else:
                # It's a file path - use .from_service_account_file()
                gcp_credentials = service_account.Credentials.from_service_account_file(
                    raw_data)

            self.vision_client = vision.ImageAnnotatorClient(
                credentials=gcp_credentials)
            self.translate_client = translate.Client(
                credentials=gcp_credentials)
            logger.info("✅ GCP clients initialized successfully")
            return True

        except json.JSONDecodeError as e:
            self.initialization_error = f"Invalid JSON in GCP_SERVICE_ACCOUNT_JSON: {str(e)}"
            logger.error(self.initialization_error)
            return False
        except FileNotFoundError as e:
            self.initialization_error = f"Credentials file not found: {str(e)}"
            logger.error(self.initialization_error)
            return False
        except Exception as e:
            self.initialization_error = f"GCP initialization error: {type(e).__name__}: {str(e)}"
            logger.error(self.initialization_error)
            return False
