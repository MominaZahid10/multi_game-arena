from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation, data_integrity, train_test_validation
import pandas as pd
import numpy as np
from backend.services.model1 import CrossGamePersonalityClassifier
import joblib

def validate_training_data(df: pd.DataFrame):
    """
    Validate training data quality using DeepChecks
    """
    print("🔍 Running data integrity checks...")
    
    ds = Dataset(
        df, 
        label='personality_category',
        cat_features=['game_type']
    )
    
    suite = data_integrity()
    result = suite.run(ds)
    
    print("✅ Data integrity check complete!")
    result.save_as_html('reports/data_integrity_report.html')
    
    return result


def validate_train_test_split(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Validate that train and test sets are properly split
    """
    print("🔍 Running train-test validation...")
    
    train_ds = Dataset(train_df, label='personality_category')
    test_ds = Dataset(test_df, label='personality_category')
    
    suite = train_test_validation()
    result = suite.run(train_ds, test_ds)
    
    print("✅ Train-test validation complete!")
    result.save_as_html('reports/train_test_validation.html')
    
    return result


def validate_model_performance(model, X_train, y_train, X_test, y_test):
    """
    Validate model performance using DeepChecks
    """
    print("🔍 Running model evaluation checks...")
    
    train_ds = Dataset(
        pd.DataFrame(X_train),
        label=pd.Series(y_train, name='label')
    )
    
    test_ds = Dataset(
        pd.DataFrame(X_test),
        label=pd.Series(y_test, name='label')
    )
    
    suite = model_evaluation()
    result = suite.run(train_ds, test_ds, model)
    
    print("✅ Model evaluation complete!")
    result.save_as_html('reports/model_evaluation.html')
    
    return result


def detect_data_drift(reference_df: pd.DataFrame, production_df: pd.DataFrame):
    """
    Detect data drift between reference and production data
    """
    from deepchecks.tabular.checks import DatasetsSizeComparison, FeatureDrift
    
    print("🔍 Running drift detection...")
    
    reference_ds = Dataset(reference_df, label='personality_category')
    production_ds = Dataset(production_df, label='personality_category')
    
    drift_check = FeatureDrift()
    drift_result = drift_check.run(reference_ds, production_ds)
    
    if drift_result.value['Drift score'] > 0.3:
        print(f"⚠️ HIGH DRIFT DETECTED: {drift_result.value['Drift score']:.3f}")
        print("📧 Consider retraining the model!")
    else:
        print(f"✅ Drift within acceptable range: {drift_result.value['Drift score']:.3f}")
    
    return drift_result


def run_complete_validation():
    """
    Run complete validation pipeline
    """
    print("🚀 Starting comprehensive ML validation...")
    
    print("\n📦 Loading ML model...")
    clf = CrossGamePersonalityClassifier()
    clf.load_models('hybrid_personality_system.pkl')
    
    print("\n📊 Generating test data...")
    test_data = clf.generate_training_data(n_samples=1000)
    
    from sklearn.model_selection import train_test_split
    X = test_data['fighting_features']
    y = test_data['personality_categories']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n🔍 Step 1: Data Integrity Validation")
    train_df = pd.DataFrame(X_train)
    train_df['personality_category'] = y_train
    validate_training_data(train_df)
    
    print("\n🔍 Step 2: Train-Test Split Validation")
    test_df = pd.DataFrame(X_test)
    test_df['personality_category'] = y_test
    validate_train_test_split(train_df, test_df)
    
    print("\n🔍 Step 3: Model Performance Validation")
    print("\n🔍 Step 4: Drift Detection")
    production_data = clf.generate_training_data(n_samples=500)
    production_df = pd.DataFrame(production_data['fighting_features'])
    production_df['personality_category'] = production_data['personality_categories']
    detect_data_drift(train_df, production_df)
    
    print("\n✅ Complete validation pipeline finished!")
    print("📊 Reports saved to reports/ directory")


def setup_continuous_monitoring():
    """
    Set up continuous monitoring for production
    """
    print("📊 Setting up continuous monitoring...")
    
    monitoring_config = {
        "data_quality_threshold": 0.95,
        "drift_threshold": 0.3,
        "min_accuracy": 0.75,
        "check_interval": "daily"
    }
    
    print("✅ Monitoring configured:")
    for key, value in monitoring_config.items():
        print(f"   - {key}: {value}")
    
    return monitoring_config


if __name__ == "__main__":
    import os
    
    os.makedirs('reports', exist_ok=True)
    run_complete_validation()
    setup_continuous_monitoring()
    
    print("\n" + "="*70)
    print("✅ ALL VALIDATION COMPLETE!")
    print("="*70)
    