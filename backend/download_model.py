import os
import shutil
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("❌ Error: 'gdown' library is missing. Please add 'gdown' to requirements.txt")
    sys.exit(1)

FILE_ID = "1ULeOcT7t4oy5T-d9Shr7JStkqsqGG1F8"
DESTINATION_ROOT = Path("hybrid_personality_system.pkl")
DESTINATION_SERVICES = Path(__file__).parent / "services" / "hybrid_personality_system.pkl"

def main():
    print(f"📥 Starting model download from Google Drive (ID: {FILE_ID})...")
    
    if DESTINATION_ROOT.exists():
        os.remove(DESTINATION_ROOT)
    
    try:
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, str(DESTINATION_ROOT), quiet=False)
        
        if not DESTINATION_ROOT.exists():
            print("❌ Error: Download failed (file not found).")
            sys.exit(1)
            
        file_size = DESTINATION_ROOT.stat().st_size
        print(f"   ✅ Downloaded: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 100 * 1024:
             print("❌ Error: File is too small. It might be an HTML error page.")
             sys.exit(1)

    except Exception as e:
        print(f"❌ Failed to download: {e}")
        sys.exit(1)

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