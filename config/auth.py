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
            # 1. Attempt to load from environment variable (Local/Specific SA)
            raw_data = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
            gcp_credentials = None

            if raw_data:
                # Check if it's a file path (Local) or actual JSON (Cloud/Secrets)
                if raw_data.strip().startswith('{'):
                    # It's actual JSON text
                    creds_dict = json.loads(raw_data)
                    gcp_credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    logger.info("✅ Loaded credentials from JSON string")
                else:
                    # It's a file path
                    gcp_credentials = service_account.Credentials.from_service_account_file(raw_data)
                    logger.info("✅ Loaded credentials from file path")
            
            else:
                # 2. Fallback to Application Default Credentials (ADC) - Best for Cloud Run
                import google.auth
                gcp_credentials, project_id = google.auth.default()
                logger.info(f"✅ Loaded Application Default Credentials (Project: {project_id})")

            # Initialize clients with the resolved credentials
            self.vision_client = vision.ImageAnnotatorClient(credentials=gcp_credentials)
            self.translate_client = translate.Client(credentials=gcp_credentials)
            
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
