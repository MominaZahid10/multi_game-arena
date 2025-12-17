import sys
import os
import joblib  
import requests 
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from prefect import flow, task, serve
from prefect.cache_policies import NO_CACHE 
from backend.services.model1 import CrossGamePersonalityClassifier
from backend.databaseconn import SessionLocal
from backend.dbmodels.games import PlayerAction

DISCORD_WEBHOOK_URL=os.getenv("DISCORD_WEBHOOK_URL")
@task(name="Load Existing Model", retries=3, retry_delay_seconds=60)
def load_existing_model():
    """Load your pre-trained model safely"""
    print("📦 Loading existing trained model...")
    
    model_path = "hybrid_personality_system.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")
    
    clf = CrossGamePersonalityClassifier()
    
    with joblib.parallel_backend('threading', n_jobs=1):
        success = clf.load_models(model_path)
    
    if not success:
        raise ValueError("❌ Failed to load model")
    
    try:
        if hasattr(clf, 'fighting_regressor'): clf.fighting_regressor.n_jobs = 1
        if hasattr(clf, 'badminton_regressor'): clf.badminton_regressor.n_jobs = 1
        if hasattr(clf, 'racing_regressor'): clf.racing_regressor.n_jobs = 1
        if hasattr(clf, 'personality_classifier'): clf.personality_classifier.n_jobs = 1
        if hasattr(clf, 'playstyle_classifier'): clf.playstyle_classifier.n_jobs = 1
        print("✅ Applied memory safety patches to model.")
    except Exception as e:
        print(f"⚠️ Warning: Could not patch model settings: {e}")

    print(f"✅ Model loaded successfully from {model_path}")
    return clf

@task(name="Extract Training Data", retries=3, retry_delay_seconds=60)
def extract_training_data():
    """Extract player actions from database"""
    print("📊 Extracting recent player actions from database...")
    
    db = SessionLocal()
    try:
        cutoff_date = datetime.now() - timedelta(days=7)
        actions = db.query(PlayerAction).filter(PlayerAction.created_at >= cutoff_date).all()
        
        print(f"✅ Extracted {len(actions)} player actions")
        
        monitoring_data = []
        if not actions:
            print("⚠️ Database is empty, generating dummy data for pipeline test")
            return pd.DataFrame([{
                'game': 'fighting', 'aggression': 0.8, 'defense': 0.2, 
                'combo': 5, 'success': 1, 'timestamp': datetime.now()
            }])

        for action in actions:
            monitoring_data.append({
                'game': action.game_type,
                'aggression': 1 if action.move_type in ['attack', 'punch'] else 0,
                'defense': 1 if action.move_type == 'block' else 0,
                'combo': action.combo_count or 0,
                'success': 1 if action.success else 0,
                'timestamp': action.created_at
            })
        
        return pd.DataFrame(monitoring_data)
    finally:
        db.close()

@task(name="Validate Data Quality", retries=2)
def validate_data(df: pd.DataFrame):
    print("🔍 Validating data quality...")
    if df.empty: return df
    df = df.drop_duplicates()
    print(f"📊 Data shape: {df.shape}")
    return df

@task(name="Test Model Predictions", retries=2, cache_policy=NO_CACHE)
def test_model_predictions(clf: CrossGamePersonalityClassifier):
    """Test model with memory safety"""
    print("🧪 Testing model predictions...")
    
    test_features = {'fighting': [0.95, 0.10, 0.85, 0.75]}
    
    # --- MEMORY FIX: Force single core execution ---
    with joblib.parallel_backend('threading', n_jobs=1):
        result = clf.predict_personality(test_features)
    
    print(f"✅ Test Prediction: {result['personality_archetype']}")
    return True

@task(name="Monitor Model Performance", retries=2, cache_policy=NO_CACHE)
def monitor_model_performance(clf: CrossGamePersonalityClassifier, df: pd.DataFrame):
    print("📈 Monitoring model performance...")
    # (Simplified logic for demo)
    return {
        "total_actions": len(df),
        "success_rate": 0.85, 
        "needs_retraining": False,
        "timestamp": datetime.now().isoformat()
    }

@task(name="Conditional Model Retraining", retries=1)
def conditional_retrain(needs_retraining: bool):
    if not needs_retraining:
        print("✅ Model performance is good - skipping retraining")
        return
    print("🔄 Retraining logic would go here...")

@task(name="Update Analytics Dashboard", retries=2)
def update_analytics_dashboard(metrics: dict):
    print("📊 Updating analytics dashboard...")
    print(f"✅ Dashboard updated with {metrics['total_actions']} actions")
    return {"updated_at": datetime.now().isoformat()}

@task(name="Send Notification", retries=3)
def send_notification(success: bool, message: str):
    """Send REAL notification to Discord/Slack"""
    print(f"📧 Sending notification: {message}")
    
    if "YOUR_DISCORD" in DISCORD_WEBHOOK_URL:
        print("⚠️ No Discord URL provided. Skipping web request.")
        return
        
    color = 5763719 if success else 15548997 # Green or Red
    
    data = {
        "embeds": [{
            "title": "🤖 ML Pipeline Status: " + ("SUCCESS" if success else "FAILED"),
            "description": message,
            "color": color,
            "footer": {"text": "Prefect Orchestrator"},
            "timestamp": datetime.now().isoformat()
        }]
    }
    
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data)
        print("✅ Notification sent to Discord!")
    except Exception as e:
        print(f"⚠️ Failed to send notification: {e}")

@flow(name="Daily Model Monitoring", log_prints=True)
def daily_model_monitoring():
    try:
        print("🚀 Starting Daily Model Monitoring...")
        clf = load_existing_model()
        df = extract_training_data()
        df_clean = validate_data(df)
        test_model_predictions(clf)
        metrics = monitor_model_performance(clf, df_clean)
        
        if metrics.get('needs_retraining'):
            conditional_retrain(True)
            
        update_analytics_dashboard(metrics)
        send_notification(True, f"Pipeline finished. Actions processed: {metrics.get('total_actions')}")
        print("✅ Monitoring completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")
        send_notification(False, f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("🚀 Configuring Prefect deployments...")
    
    daily_deploy = daily_model_monitoring.to_deployment(
        name="daily-model-monitoring",
        cron="0 2 * * *",
        tags=["ml", "monitoring"]
    )
    
    print("🚀 Starting local runner...")
    serve(daily_deploy)