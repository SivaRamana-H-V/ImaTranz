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
        """Initialize GCP clients for both local development and Cloud Run"""
        try:
            # METHOD 1: Environment variable with JSON string (Cloud Run / GitHub Secrets)
            if "GCP_SERVICE_ACCOUNT_JSON" in os.environ:
                logging.info("Found GCP_SERVICE_ACCOUNT_JSON in environment")
                creds_json = os.environ["GCP_SERVICE_ACCOUNT_JSON"]
                creds_dict = self._parse_json_credentials(creds_json)

                if creds_dict:
                    credentials = service_account.Credentials.from_service_account_info(
                        creds_dict)
                    return self._create_clients(credentials, "Environment variable JSON")

            # METHOD 2: File path environment variable (Local development)
            elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
                logging.info(
                    f"Found GOOGLE_APPLICATION_CREDENTIALS: {cred_path}")

                if os.path.exists(cred_path):
                    credentials = service_account.Credentials.from_service_account_file(
                        cred_path)
                    return self._create_clients(credentials, "Credentials file")
                else:
                    logging.error(f"Credentials file not found: {cred_path}")

            # METHOD 3: Look for common local credential files
            else:
                # Check for common local credential files
                local_paths = [
                    "gcp-key.json",
                    "credentials.json",
                    "service-account.json",
                    "E:\JD Sister Projects\Tranze\pivotal-robot-369519-6d5bdc432664.json",
                    os.path.expanduser(
                        "~/.config/gcloud/application_default_credentials.json")
                ]

                for path in local_paths:
                    if os.path.exists(path):
                        logging.info(f"Found local credentials: {path}")
                        credentials = service_account.Credentials.from_service_account_file(
                            path)
                        return self._create_clients(credentials, f"Local file: {path}")

            # METHOD 4: Mounted secret (Cloud Run specific)
            if os.path.exists('/secrets/gcp-credentials.json'):
                logging.info(
                    "Found mounted secret at /secrets/gcp-credentials.json")
                credentials = service_account.Credentials.from_service_account_file(
                    '/secrets/gcp-credentials.json'
                )
                return self._create_clients(credentials, "Mounted secret")

            # METHOD 5: Application Default Credentials (GCP services)
            logging.info("Attempting Application Default Credentials")
            self.vision_client = vision.ImageAnnotatorClient()
            self.translate_client = translate.Client()
            logging.info("✅ GCP clients initialized using ADC")
            return True

        except Exception as e:
            logging.error(f"❌ GCP initialization failed: {e}")
            self._print_help()
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
