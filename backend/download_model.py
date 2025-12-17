import os
import requests
import shutil
from pathlib import Path

FILE_ID = "https://drive.google.com/file/d/1ULeOcT7t4oy5T-d9Shr7JStkqsqGG1F8/view?usp=drive_link"
DESTINATION_ROOT = Path("hybrid_personality_system.pkl")
DESTINATION_SERVICES = Path(__file__).parent / "services" / "hybrid_personality_system.pkl"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: 
                f.write(chunk)

def main():
    print(f"📥 Starting model download from Google Drive...")
    
    if not DESTINATION_ROOT.exists():
        print("   Downloading to root directory...")
        try:
            download_file_from_google_drive(FILE_ID, DESTINATION_ROOT)
            print(f"   ✅ Downloaded to: {DESTINATION_ROOT}")
        except Exception as e:
            print(f"   ❌ Failed to download: {e}")
            exit(1)
    else:
        print(f"   ✅ Found in root: {DESTINATION_ROOT}")

    DESTINATION_SERVICES.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION_SERVICES.exists():
        print("   Copying to backend/services/...")
        shutil.copy(DESTINATION_ROOT, DESTINATION_SERVICES)
        print(f"   ✅ Copied to: {DESTINATION_SERVICES}")
    else:
        print(f"   ✅ Found in services: {DESTINATION_SERVICES}")

    print("🚀 Model setup complete.")

if __name__ == "__main__":
    main()