import os
import json
import logging
from google.cloud import vision
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)


class GCPAuth:
    def __init__(self):
        self.vision_client = None
        self.translate_client = None

    def initialize_clients(self):
        try:
            # 1. Try Cloud Run / Render (Environment Variable String)
            if "GCP_SERVICE_ACCOUNT_JSON" in os.environ:
                creds_json = os.environ["GCP_SERVICE_ACCOUNT_JSON"]
                creds_dict = json.loads(creds_json)
                # Use .from_service_account_info for RAW TEXT
                gcp_credentials = service_account.Credentials.from_service_account_info(
                    creds_dict)
                return self._create_clients(gcp_credentials, "Cloud Environment")

            # 2. Try Local Development (File Path)
            elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                # Use .from_service_account_file for FILE PATHS
                gcp_credentials = service_account.Credentials.from_service_account_file(
                    cred_path)
                return self._create_clients(gcp_credentials, "Local File")

            else:
                print("No credentials found. Please check setup.")
                return False

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            return False

    def _parse_json_credentials(self, creds_json):
        """Parse JSON credentials handling different formats"""

        # Method 1: Direct parse
        try:
            return json.loads(creds_json)
        except json.JSONDecodeError:
            pass

        # Method 2: Handle escaped JSON
        try:
            cleaned = creds_json.replace('\\n', '\n').replace('\\"', '"')
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Method 3: Base64 encoded
        try:
            import base64
            decoded = base64.b64decode(creds_json).decode('utf-8')
            return json.loads(decoded)
        except:
            pass

        logging.error("All JSON parsing methods failed")
        return None

    def _create_clients(self, credentials, method):
        """Create GCP clients with the provided credentials"""
        try:
            self.vision_client = vision.ImageAnnotatorClient(
                credentials=credentials
            )
            self.translate_client = translate.Client(
                credentials=credentials
            )
            logging.info(f"✅ GCP clients initialized using {method}")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to create GCP clients: {e}")
            return False

    def _print_help(self):
        """Print helpful error message"""
        print("\n" + "="*50)
        print("🔐 GCP CREDENTIALS NOT FOUND")
        print("="*50)
        print("\nFor LOCAL DEVELOPMENT:")
        print("Your credentials file should be at:")
        print(f"   {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
        print("\nCheck if this file exists:")
        cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if cred_path:
            if os.path.exists(cred_path):
                print(f"✅ File exists: {cred_path}")
            else:
                print(f"❌ File NOT found: {cred_path}")
                print("\nJSON files in current directory:")
                for f in os.listdir('.'):
                    if f.endswith('.json'):
                        print(f"   - {f}")
        print("="*50 + "\n")
