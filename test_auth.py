import os
from firebase_admin import auth, credentials
import firebase_admin
from dotenv import load_dotenv

load_dotenv()

def test_config():
    print("--- AgriMicro IQ ML Auth Diagnostic ---")
    
    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        print("✅ Environment: FIREBASE_SERVICE_ACCOUNT_JSON found.")
    else:
        print("❌ Environment: FIREBASE_SERVICE_ACCOUNT_JSON NOT found.")
        
    sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "firebaseServiceAccount.json")
    if os.path.exists(sa_path):
        print(f"✅ File: {sa_path} found.")
    else:
        print(f"❌ File: {sa_path} NOT found.")

    try:
        if not firebase_admin._apps:
            if sa_json:
                import json
                cred = credentials.Certificate(json.loads(sa_json))
                firebase_admin.initialize_app(cred)
            elif os.path.exists(sa_path):
                cred = credentials.Certificate(sa_path)
                firebase_admin.initialize_app(cred)
            else:
                print("❌ Cannot initialize: No credentials.")
                return
        
        print("✅ Firebase Admin: Initialized successfully.")
        print("✅ Verification readiness: OK.")
        
    except Exception as e:
        print(f"❌ Firebase Error: {e}")

if __name__ == "__main__":
    test_config()
