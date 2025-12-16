
import os
import requests
from pathlib import Path

MODEL_URL = "https://drive.google.com/file/d/1cKsLODbuZXHapxAVjPTSaTRnlJklJ6G_/view?usp=drive_link"  
MODEL_PATH = Path(__file__).parent / "services" / "hybrid_personality_system.pkl"

def download_model():
    """Download model if not present"""
    if MODEL_PATH.exists():
        print(f"✅ Model already exists at {MODEL_PATH}")
        return
    
    print(f"📥 Downloading model from cloud storage...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # For Google Drive direct download
    file_id = MODEL_URL.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(MODEL_PATH, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                progress = (downloaded / total_size) * 100
                print(f"\rProgress: {progress:.1f}%", end='')
    
    print(f"\n✅ Model downloaded successfully to {MODEL_PATH}")

if __name__ == "__main__":
    download_model()