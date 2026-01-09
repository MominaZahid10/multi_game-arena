import os
import requests
import shutil
import sys
from pathlib import Path

MODEL_URL = "https://huggingface.co/mominazahid/multi-game-arena-model/resolve/main/hybrid_personality_system.pkl?download=true"

DESTINATION_ROOT = Path("hybrid_personality_system.pkl")
DESTINATION_SERVICES = Path(__file__).parent / "services" / "hybrid_personality_system.pkl"

def main():
    print(f"📥 Starting model download from Hugging Face...")
    print(f"   URL: {MODEL_URL}")
    
    if DESTINATION_ROOT.exists():
        os.remove(DESTINATION_ROOT)
    
    try:
        with requests.get(MODEL_URL, stream=True) as response:
            response.raise_for_status()
            with open(DESTINATION_ROOT, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        if not DESTINATION_ROOT.exists():
            print("❌ Error: Download failed (file not found).")
            sys.exit(1)
            
        file_size = DESTINATION_ROOT.stat().st_size
        print(f"   ✅ Downloaded: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 100 * 1024:
             print("❌ Error: File is too small. Check your URL.")
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
