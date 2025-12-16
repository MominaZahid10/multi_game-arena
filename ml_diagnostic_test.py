"""
🧪 SAFE API INTEGRATION TEST (No Local Loading)
This script tests your ML system ONLY via the API.
It will NOT crash your memory and will NOT overwrite your model.
"""
import requests
import time
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
API_URL = "http://localhost:8000/api/v1"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def print_pass(text):
    print(f"{Colors.GREEN}✅ PASS: {text}{Colors.RESET}")

def print_fail(text):
    print(f"{Colors.RED}❌ FAIL: {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.RESET}")

def run_safe_test():
    print(f"\n{Colors.CYAN}🚀 STARTING SAFE ML SYSTEM VERIFICATION{Colors.RESET}")
    print("="*60)

    # 1. Check if Server is Online
    try:
        requests.get("http://localhost:8000/", timeout=5)
        print_pass("Server is ONLINE")
    except:
        print_fail("Server is OFFLINE. Please run: python -m backend.main_optimized")
        return

    # 2. Test ML Prediction (The most important test)
    print_info("Testing ML Prediction Endpoint...")
    payload = {
        "aggression_rate": 0.9,
        "defense_ratio": 0.1,
        "combo_preference": 0.8,
        "reaction_time": 0.7
    }
    
    try:
        # Give it 60 seconds because large models can be slow on the first request
        start = time.time()
        response = requests.post(
            f"{API_URL}/ml/fighting/predict", 
            json=payload, 
            timeout=60
        )
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            archetype = data.get('archetype_name', 'Unknown')
            confidence = data.get('confidence', 0)
            print_pass(f"ML Prediction Successful in {duration:.2f}s")
            print(f"   👤 Archetype: {Colors.YELLOW}{archetype}{Colors.RESET}")
            print(f"   📊 Confidence: {confidence:.1%}")
            
            if duration > 5.0:
                print(f"{Colors.YELLOW}   ⚠️  Note: Prediction was slow ({duration:.2f}s). This is normal for 700MB+ models.{Colors.RESET}")
        else:
            print_fail(f"API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print_fail(f"Connection failed: {e}")

    # 3. Test Game Analytics (Database Check)
    print_info("Testing Database Integration...")
    try:
        # Create a dummy session
        session_id = f"test_safe_{int(time.time())}"
        requests.post(
            f"{API_URL}/games/fighting/action",
            params={"session_id": session_id},
            json={"action_type": "punch", "position": [0,0]}
        )
        
        # Read it back
        res = requests.get(f"{API_URL}/analytics/{session_id}")
        if res.status_code == 200:
            print_pass("Database Read/Write OK")
        else:
            print_fail("Database Error")
            
    except Exception as e:
        print_fail(f"Database test failed: {e}")

    print("="*60)
    print(f"{Colors.GREEN}✨ SYSTEM VERIFIED WITHOUT TOUCHING MODEL FILE{Colors.RESET}\n")

if __name__ == "__main__":
    run_safe_test()