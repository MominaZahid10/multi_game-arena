
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CrossGamePersonalityClassifier:
   
    def __init__(self):
        self.fighting_regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))
        self.badminton_regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))
        self.racing_regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))
        self.meta_regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1))
        
        self.personality_classifier = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, random_state=42, n_jobs=1)),
            ('et', ExtraTreesClassifier(n_estimators=300, max_depth=18, min_samples_split=3, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=10, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=150, learning_rate=0.1, random_state=42)),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
            ('lda', LinearDiscriminantAnalysis()),
            ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance'))
        ], voting='soft')
        
        self.playstyle_classifier = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42, n_jobs=1)),
            ('et', ExtraTreesClassifier(n_estimators=250, max_depth=15, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=180, learning_rate=0.1, random_state=42)),
            ('svm', SVC(kernel='rbf', C=5, probability=True, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=800, random_state=42))
        ], voting='soft')
        
        self.scaler_fighting = RobustScaler()  
        self.scaler_badminton = RobustScaler()
        self.scaler_racing = RobustScaler()
        self.scaler_combined = StandardScaler()
        
        self.is_trained = False
        
        self.feature_names = {
            'fighting': ['aggression_rate', 'defense_ratio', 'combo_preference', 'reaction_time'],
            'badminton': ['shot_variety', 'power_control', 'court_positioning', 'rally_patience'],
            'racing': ['speed_preference', 'precision_level', 'overtaking_aggression', 'consistency']
        }
        
        self.personality_traits = [
            'aggression_level', 'risk_tolerance', 'analytical_thinking', 
            'patience_level', 'precision_focus', 'competitive_drive', 'strategic_thinking'
        ]
        
        self.personality_archetypes = [
            "🔥 Aggressive Dominator",     
            "🧠 Strategic Analyst",     
            "⚡ Risk-Taking Maverick",    
            "🛡️ Defensive Tactician",    
            "🎯 Precision Master",        
            "🌪️ Chaos Creator",          
            "📊 Data-Driven Player",      
            "🏆 Victory Seeker"            
        ]
        
        self.playstyle_categories = [
            "🥊 Combat Veteran",           
            "🏸 Court Strategist",        
            "🏎️ Speed Demon",           
            "🎮 Multi-Game Master",       
            "🧩 Adaptive Learner",        
            "🔄 Pattern Seeker"           
        ]
    
    def generate_training_data(self, n_samples=10000):
        print(f" Generating {n_samples} samples with EXTREME personality separation...")
        
        archetypes = {
            'aggressive_dominator': {
                'category_id': 0,
                'playstyle_id': 0,
                'weight': 0.14,
                'traits': {
                    'aggression_level': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0),    
                    'risk_tolerance': lambda: np.clip(np.random.normal(0.90, 0.05), 0.8, 1.0),
                    'patience_level': lambda: np.clip(np.random.normal(0.15, 0.05), 0.05, 0.25),    
                    'precision_focus': lambda: np.clip(np.random.normal(0.20, 0.05), 0.1, 0.3),    
                    'competitive_drive': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0), 
                    'analytical_thinking': lambda: np.clip(np.random.normal(0.25, 0.08), 0.1, 0.4), 
                    'strategic_thinking': lambda: np.clip(np.random.normal(0.25, 0.08), 0.1, 0.4)    
                }
            },
            'strategic_analyst': {
                'category_id': 1,
                'playstyle_id': 1,
                'weight': 0.16,
                'traits': {
                    'aggression_level': lambda: np.clip(np.random.normal(0.12, 0.04), 0.05, 0.20),  
                    'risk_tolerance': lambda: np.clip(np.random.normal(0.20, 0.05), 0.1, 0.3),      
                    'patience_level': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0),     
                    'precision_focus': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0),    
                    'competitive_drive': lambda: np.clip(np.random.normal(0.60, 0.10), 0.4, 0.8),
                    'analytical_thinking': lambda: np.clip(np.random.normal(0.98, 0.02), 0.9, 1.0),  
                    'strategic_thinking': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0)    
                }
            },
            'risk_taking_maverick': {
                'category_id': 2,
                'playstyle_id': 2,
                'weight': 0.13,
                'traits': {
                    'aggression_level': lambda: np.clip(np.random.normal(0.80, 0.08), 0.65, 0.95),
                    'risk_tolerance': lambda: np.clip(np.random.normal(0.98, 0.02), 0.9, 1.0),       
                    'patience_level': lambda: np.clip(np.random.normal(0.12, 0.05), 0.05, 0.2),      
                    'precision_focus': lambda: np.clip(np.random.normal(0.15, 0.05), 0.05, 0.25),    
                    'competitive_drive': lambda: np.clip(np.random.normal(0.90, 0.06), 0.75, 1.0),
                    'analytical_thinking': lambda: np.clip(np.random.normal(0.30, 0.10), 0.1, 0.5),
                    'strategic_thinking': lambda: np.clip(np.random.normal(0.25, 0.08), 0.1, 0.4)
                }
            },
            'defensive_tactician': {
                'category_id': 3,
                'playstyle_id': 1,
                'weight': 0.13,
                'traits': {
                    'aggression_level': lambda: np.clip(np.random.normal(0.10, 0.04), 0.05, 0.18),   
                    'risk_tolerance': lambda: np.clip(np.random.normal(0.12, 0.04), 0.05, 0.2),      
                    'patience_level': lambda: np.clip(np.random.normal(0.98, 0.02), 0.9, 1.0),       
                    'precision_focus': lambda: np.clip(np.random.normal(0.85, 0.08), 0.7, 1.0),
                    'competitive_drive': lambda: np.clip(np.random.normal(0.70, 0.10), 0.5, 0.9),
                    'analytical_thinking': lambda: np.clip(np.random.normal(0.90, 0.06), 0.75, 1.0),
                    'strategic_thinking': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0)    
                }
            },
            'precision_master': {
                'category_id': 4,
                'playstyle_id': 4,
                'weight': 0.13,
                'traits': {
                    'aggression_level': lambda: np.clip(np.random.normal(0.45, 0.08), 0.35, 0.60),
                    'risk_tolerance': lambda: np.clip(np.random.normal(0.10, 0.04), 0.05, 0.18),    
                    'patience_level': lambda: np.clip(np.random.normal(0.85, 0.08), 0.7, 1.0),
                    'precision_focus': lambda: np.clip(np.random.normal(0.98, 0.02), 0.9, 1.0),      
                    'competitive_drive': lambda: np.clip(np.random.normal(0.75, 0.08), 0.6, 0.9),
                    'analytical_thinking': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0),  
                    'strategic_thinking': lambda: np.clip(np.random.normal(0.80, 0.08), 0.65, 0.95)
                }
            },
            'chaos_creator': {
                'category_id': 5,
                'playstyle_id': 0,
                'weight': 0.13,
                'traits': {
                    'aggression_level': lambda: np.clip(np.random.normal(0.90, 0.06), 0.75, 1.0),    
                    'risk_tolerance': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0),     
                    'patience_level': lambda: np.clip(np.random.normal(0.08, 0.03), 0.02, 0.15),   
                    'precision_focus': lambda: np.clip(np.random.normal(0.08, 0.03), 0.02, 0.15),   
                    'competitive_drive': lambda: np.clip(np.random.normal(0.85, 0.08), 0.7, 1.0),
                    'analytical_thinking': lambda: np.clip(np.random.normal(0.15, 0.05), 0.05, 0.25),
                    'strategic_thinking': lambda: np.clip(np.random.normal(0.15, 0.05), 0.05, 0.25)  
                }
            },
            'data_driven_player': {
                'category_id': 6,
                'playstyle_id': 5,
                'weight': 0.12,
                'traits': {
                    'aggression_level': lambda: np.clip(np.random.normal(0.30, 0.06), 0.2, 0.4),
                    'risk_tolerance': lambda: np.clip(np.random.normal(0.15, 0.04), 0.08, 0.25),     
                    'patience_level': lambda: np.clip(np.random.normal(0.92, 0.05), 0.8, 1.0),       
                    'precision_focus': lambda: np.clip(np.random.normal(0.95, 0.04), 0.85, 1.0),     
                    'competitive_drive': lambda: np.clip(np.random.normal(0.65, 0.08), 0.5, 0.8),
                    'analytical_thinking': lambda: np.clip(np.random.normal(0.98, 0.02), 0.9, 1.0),   
                    'strategic_thinking': lambda: np.clip(np.random.normal(0.90, 0.05), 0.8, 1.0)
                }
            },
            'victory_seeker': {  
                'category_id': 7,
                'playstyle_id': 3,
                'weight': 0.04, 
                'traits': {
                    'aggression_level': lambda: np.clip(np.random.normal(0.75, 0.08), 0.6, 0.9),
                    'risk_tolerance': lambda: np.clip(np.random.normal(0.70, 0.08), 0.55, 0.85),
                    'patience_level': lambda: np.clip(np.random.normal(0.60, 0.10), 0.4, 0.8),
                    'precision_focus': lambda: np.clip(np.random.normal(0.80, 0.08), 0.65, 0.95),
                    'competitive_drive': lambda: np.clip(np.random.normal(0.98, 0.02), 0.9, 1.0),    
                    'analytical_thinking': lambda: np.clip(np.random.normal(0.70, 0.08), 0.55, 0.85),
                    'strategic_thinking': lambda: np.clip(np.random.normal(0.75, 0.08), 0.6, 0.9)
                }
            }
        }
        
        training_data = {
            'fighting_features': [],
            'badminton_features': [],
            'racing_features': [],
            'personality_scores': [],
            'personality_categories': [],
            'playstyle_categories': []
        }
        
       
        archetype_names = list(archetypes.keys())
        archetype_weights = np.array(
                 [archetypes[name]['weight'] for name in archetype_names],
                 dtype=np.float64
)

        archetype_weights /= archetype_weights.sum()

        
        for i in range(n_samples):
            archetype_name = np.random.choice(archetype_names, p=archetype_weights)
            archetype = archetypes[archetype_name]
      
            personality = {}
            for trait in self.personality_traits:
                if trait in archetype['traits']:
                    personality[trait] = archetype['traits'][trait]()
                else:
                    personality[trait] = np.clip(np.random.uniform(0.4, 0.6), 0.1, 1.0)
            
            fighting_features = [
                 personality['aggression_level'] + np.random.normal(0, 0.02),   # aggression_rate
                 personality['patience_level'] + np.random.normal(0, 0.02),     # defense_ratio (NOT inverted!)
                 personality['risk_tolerance'] + np.random.normal(0, 0.02),     # combo_preference (NOT scaled!)
                 personality['precision_focus'] + np.random.normal(0, 0.015)    # reaction_time (NOT inverted!)
            ]
            
            badminton_features = [
                personality['strategic_thinking'] + np.random.normal(0, 0.02),
                personality['precision_focus'] + np.random.normal(0, 0.02),
                personality['analytical_thinking'] + np.random.normal(0, 0.02),
                personality['patience_level'] + np.random.normal(0, 0.02)
            ]
            
            racing_features = [
                personality['competitive_drive'] + np.random.normal(0, 0.02),
                personality['precision_focus'] + np.random.normal(0, 0.02),
                personality['risk_tolerance'] + np.random.normal(0, 0.02),
                personality['analytical_thinking'] + np.random.normal(0, 0.02)
            ]
            
         
            fighting_features = np.clip(fighting_features, 0.05, 1.0)
            badminton_features = np.clip(badminton_features, 0.05, 1.0)
            racing_features = np.clip(racing_features, 0.05, 1.0)
      
            training_data['fighting_features'].append(fighting_features)
            training_data['badminton_features'].append(badminton_features)
            training_data['racing_features'].append(racing_features)
            training_data['personality_scores'].append([personality[trait] for trait in self.personality_traits])
            training_data['personality_categories'].append(archetype['category_id'])
            training_data['playstyle_categories'].append(archetype['playstyle_id'])
        
      
        for key in training_data:
            training_data[key] = np.array(training_data[key])
        
        print(f"✓ Generated ULTRA-DISTINCT training data:")
        print(f"  - Total samples: {n_samples}")
        print(f"  - Personality categories: {len(np.unique(training_data['personality_categories']))} (8 archetypes)")
        print(f"  - Playstyle categories: {len(np.unique(training_data['playstyle_categories']))} playstyles")
     
        unique, counts = np.unique(training_data['personality_categories'], return_counts=True)
        print(f"\n📊 EXTREME Category Distribution:")
        for cat_id, count in zip(unique, counts):
            percentage = (count / n_samples) * 100
            print(f"  {self.personality_archetypes[cat_id]}: {count} ({percentage:.1f}%)")
        
        return training_data
    
    def _create_ultimate_features(self, X_fighting, X_badminton, X_racing):
        
        all_features = np.hstack([X_fighting, X_badminton, X_racing])
        n_samples = all_features.shape[0]
        
        engineered_features = []
        
        for i in range(n_samples):
            features = all_features[i]
            
            aggr_risk = features[0] * features[8] 
            patience_precision = features[3] * features[5]   
            analytical_strategic = features[2] * features[6]  
            competitive_aggr = features[0] * features[9] if len(features) > 9 else features[0] * features[8]  
            precision_analytical = features[5] * features[2] 
            
            aggr_patience_ratio = features[0] / (features[3] + 0.01)
            risk_precision_ratio = features[8] / (features[5] + 0.01) 
            strategic_aggr_ratio = features[6] / (features[0] + 0.01)
            competitive_patience_ratio = features[9] / (features[3] + 0.01) if len(features) > 9 else features[8] / (features[3] + 0.01)
            analytical_risk_ratio = features[2] / (features[8] + 0.01)
            precision_aggr_ratio = features[5] / (features[0] + 0.01)
            
            aggr_squared = features[0] ** 2
            patience_squared = features[3] ** 2
            risk_squared = features[8] ** 2
            precision_squared = features[5] ** 2
            
            chaos_score = (features[0] + features[8]) - (features[3] + features[5]) 
            control_score = (features[3] + features[5] + features[2]) / 3  
            instinct_score = (features[0] + features[8]) / 2  
            balance_score = 1 - np.var([features[0], features[3], features[5], features[8]])  
            extreme_score = np.max([features[0], features[3], features[5], features[8]]) - np.min([features[0], features[3], features[5], features[8]])
            consistency_score = 1 - np.std([features[0], features[3], features[5], features[8], features[2], features[6]])
            
            fighting_dominance = np.mean([features[0], features[1]]) 
            badminton_dominance = np.mean([features[4], features[5], features[6], features[7]])  
            racing_dominance = np.mean([features[8], features[9], features[10], features[11]]) if len(features) > 11 else np.mean([features[8], features[9]]) 
            sample_engineered = [
                aggr_risk, patience_precision, analytical_strategic, competitive_aggr, precision_analytical,
                aggr_patience_ratio, risk_precision_ratio, strategic_aggr_ratio, competitive_patience_ratio,
                analytical_risk_ratio, precision_aggr_ratio,
                aggr_squared, patience_squared, risk_squared, precision_squared,
                chaos_score, control_score, instinct_score, balance_score, extreme_score, consistency_score,
                fighting_dominance, badminton_dominance, racing_dominance
            ]
            
            engineered_features.append(sample_engineered)
        
        engineered_features = np.array(engineered_features)
        
        ultimate_features = np.hstack([all_features, engineered_features])
        
        # Removed verbose logging - was printing on every prediction
        
        return ultimate_features
    
    def train_models(self, training_data):
        print(f"\n Training Hybrid System...")
        
        X_fighting = training_data['fighting_features']
        X_badminton = training_data['badminton_features']
        X_racing = training_data['racing_features']
        y_scores = training_data['personality_scores']
        y_personality_cats = training_data['personality_categories']
        y_playstyle_cats = training_data['playstyle_categories']
        
        X_fighting_scaled = self.scaler_fighting.fit_transform(X_fighting)
        X_badminton_scaled = self.scaler_badminton.fit_transform(X_badminton)
        X_racing_scaled = self.scaler_racing.fit_transform(X_racing)
        
        print(f"\n Training Regression Models:")
        
        self.fighting_regressor.fit(X_fighting_scaled, y_scores)
        fighting_r2 = self._calculate_r2_score(self.fighting_regressor, X_fighting_scaled, y_scores)
        print(f"  ✓ Fighting Regressor R² Score: {fighting_r2:.3f}")
        
        self.badminton_regressor.fit(X_badminton_scaled, y_scores)
        badminton_r2 = self._calculate_r2_score(self.badminton_regressor, X_badminton_scaled, y_scores)
        print(f"  ✓ Badminton Regressor R² Score: {badminton_r2:.3f}")
        
        self.racing_regressor.fit(X_racing_scaled, y_scores)
        racing_r2 = self._calculate_r2_score(self.racing_regressor, X_racing_scaled, y_scores)
        print(f"  ✓ Racing Regressor R² Score: {racing_r2:.3f}")
        
        fighting_pred = self.fighting_regressor.predict(X_fighting_scaled)
        badminton_pred = self.badminton_regressor.predict(X_badminton_scaled)
        racing_pred = self.racing_regressor.predict(X_racing_scaled)
        meta_features = np.hstack([fighting_pred, badminton_pred, racing_pred])
        
        self.meta_regressor.fit(meta_features, y_scores)
        meta_r2 = self._calculate_r2_score(self.meta_regressor, meta_features, y_scores)
        print(f"  ✓ Meta-Regressor R² Score: {meta_r2:.3f}")
        
        print(f"\n Training  Classification Models:")
        
        ultimate_features = self._create_ultimate_features(X_fighting_scaled, X_badminton_scaled, X_racing_scaled)
        ultimate_features_scaled = self.scaler_combined.fit_transform(ultimate_features)
        
        print(f"   Training Personality Classifier (7-algorithm ensemble)...")
        self.personality_classifier.fit(ultimate_features_scaled, y_personality_cats)
        personality_acc = cross_val_score(self.personality_classifier, ultimate_features_scaled, 
                                        y_personality_cats, cv=10, scoring='accuracy').mean()  
        print(f"  ✓ Personality Category Classifier: {personality_acc:.3f} accuracy")
        
        print(f"   Training Playstyle Classifier (5-algorithm ensemble)...")
        self.playstyle_classifier.fit(ultimate_features_scaled, y_playstyle_cats)
        playstyle_acc = cross_val_score(self.playstyle_classifier, ultimate_features_scaled, 
                                       y_playstyle_cats, cv=10, scoring='accuracy').mean()
        print(f"  ✓ Playstyle Category Classifier: {playstyle_acc:.3f} accuracy")
        
        self.is_trained = True
        
        print(f"\n ULTIMATE Classification Performance Analysis:")
        
        y_pred_personality = cross_val_predict(self.personality_classifier, ultimate_features_scaled, 
                                             y_personality_cats, cv=5)
        
        print(" Per-Category Accuracy:")
        category_accuracies = []
        for i, archetype in enumerate(self.personality_archetypes):
            mask = y_personality_cats == i
            if np.sum(mask) > 0:
                accuracy = np.mean(y_pred_personality[mask] == i)
                category_accuracies.append(accuracy)
                status = "✓ EXCELLENT" if accuracy > 0.9 else "🔧 IMPROVING" if accuracy > 0.8 else "❌ POOR"
                print(f"  {archetype}: {accuracy:.3f} ({status}) - {np.sum(mask)} samples")
        
        min_category_accuracy = np.min(category_accuracies) if category_accuracies else 0
        print(f"\n Weakest Category Accuracy: {min_category_accuracy:.3f}")
        
        performance = {
            'regression_scores': {
                'fighting_r2': fighting_r2,
                'badminton_r2': badminton_r2,
                'racing_r2': racing_r2,
                'unified_r2': meta_r2
            },
            'classification_scores': {
                'personality_accuracy': personality_acc,
                'playstyle_accuracy': playstyle_acc,
                'min_category_accuracy': min_category_accuracy
            },
            'meets_targets': {
                'regression_quality': meta_r2 > 0.85,
                'classification_quality': personality_acc > 0.90 and playstyle_acc > 0.88,
                'ultimate_target': personality_acc > 0.95 and min_category_accuracy > 0.90
            }
        }
        
        print(f"\n ULTIMATE System Performance:")
        print(f"  Regression Quality (>85% R²): {'✓ PASSED' if performance['meets_targets']['regression_quality'] else '❌ NEEDS IMPROVEMENT'}")
        print(f"  Classification Quality (>90%): {'✓ PASSED' if performance['meets_targets']['classification_quality'] else '❌ NEEDS IMPROVEMENT'}")
        print(f"  ULTIMATE Target (>95%): {'🏆 ACHIEVED!' if performance['meets_targets']['ultimate_target'] else '🔥 CLOSE - KEEP PUSHING!'}")
        
        return performance
    
    def predict_personality(self, game_features: Dict[str, List[float]]) -> Dict:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        regression_predictions = self._get_regression_predictions(game_features)
        category_predictions = self._get_ultimate_category_predictions(game_features)
        
        return {
            'personality_scores': regression_predictions['unified_personality'],
            'individual_game_scores': regression_predictions['individual_predictions'],
            'confidence_score': regression_predictions['confidence_score'],
            'personality_archetype': category_predictions['personality_archetype'],
            'playstyle_category': category_predictions['playstyle_category'],
            'category_confidence': category_predictions['category_confidence'],
            'games_analyzed': list(game_features.keys()),
            'cross_game_consistency': regression_predictions['cross_game_consistency'],
            'ultimate_features_used': category_predictions['total_features_used']
        }
    
    def _get_ultimate_category_predictions(self, game_features):
        feature_vector = []
        for game in ['fighting', 'badminton', 'racing']:
            if game in game_features:
                if game == 'fighting':
                    scaled = self.scaler_fighting.transform([game_features[game]])[0]
                elif game == 'badminton':
                    scaled = self.scaler_badminton.transform([game_features[game]])[0]
                else:
                    scaled = self.scaler_racing.transform([game_features[game]])[0]
                feature_vector.extend(scaled)
            else:
                feature_vector.extend([0.0] * 4)  
        
        feature_vector = np.array(feature_vector)
        ultimate_features = self._create_ultimate_features(
            feature_vector[:4].reshape(1, -1),
            feature_vector[4:8].reshape(1, -1),
            feature_vector[8:12].reshape(1, -1)
        )
        
        ultimate_features_scaled = self.scaler_combined.transform(ultimate_features)
        
        personality_cat_id = self.personality_classifier.predict(ultimate_features_scaled)[0]
        playstyle_cat_id = self.playstyle_classifier.predict(ultimate_features_scaled)[0]
        
        personality_proba = np.max(self.personality_classifier.predict_proba(ultimate_features_scaled))
        playstyle_proba = np.max(self.playstyle_classifier.predict_proba(ultimate_features_scaled))
        
        return {
            'personality_archetype': self.personality_archetypes[personality_cat_id],
            'playstyle_category': self.playstyle_categories[playstyle_cat_id],
            'category_confidence': float((personality_proba + playstyle_proba) / 2),
            'total_features_used': ultimate_features_scaled.shape[1]
        }
    
    def _get_regression_predictions(self, game_features):
        predictions = {}
        confidences = {}
        
        if 'fighting' in game_features:
            fighting_scaled = self.scaler_fighting.transform([game_features['fighting']])
            fighting_pred = self.fighting_regressor.predict(fighting_scaled)[0]
            predictions['fighting'] = dict(zip(self.personality_traits, fighting_pred))
            confidences['fighting'] = self._calculate_prediction_confidence(fighting_pred)
        
        if 'badminton' in game_features:
            badminton_scaled = self.scaler_badminton.transform([game_features['badminton']])
            badminton_pred = self.badminton_regressor.predict(badminton_scaled)[0]
            predictions['badminton'] = dict(zip(self.personality_traits, badminton_pred))
            confidences['badminton'] = self._calculate_prediction_confidence(badminton_pred)
        
        if 'racing' in game_features:
            racing_scaled = self.scaler_racing.transform([game_features['racing']])
            racing_pred = self.racing_regressor.predict(racing_scaled)[0]
            predictions['racing'] = dict(zip(self.personality_traits, racing_pred))
            confidences['racing'] = self._calculate_prediction_confidence(racing_pred)
        
        if len(predictions) >= 2:
            meta_input = []
            for game in ['fighting', 'badminton', 'racing']:
                if game in predictions:
                    meta_input.extend([predictions[game][trait] for trait in self.personality_traits])
                else:
                    meta_input.extend([0.5] * len(self.personality_traits))
            
            unified_pred = self.meta_regressor.predict([meta_input])[0]
            unified_personality = dict(zip(self.personality_traits, unified_pred))
            overall_confidence = np.mean(list(confidences.values()))
        else:
            game_name = list(predictions.keys())[0]
            unified_personality = predictions[game_name]
            overall_confidence = confidences[game_name]
        
        return {
            'unified_personality': unified_personality,
            'individual_predictions': predictions,
            'confidence_score': float(overall_confidence),
            'cross_game_consistency': self._calculate_consistency(predictions) if len(predictions) > 1 else 1.0
        }
    
    def _calculate_r2_score(self, model, X, y):
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        return scores.mean()
    
    def _calculate_prediction_confidence(self, prediction):
        extremeness = np.mean([abs(p - 0.5) * 2 for p in prediction])
        return min(1.0, 0.5 + extremeness / 2)
    
    def _calculate_consistency(self, predictions: Dict) -> float:
        if len(predictions) < 2:
            return 1.0
        
        games = list(predictions.keys())
        consistencies = []
        
        for trait in self.personality_traits:
            trait_values = [predictions[game][trait] for game in games]
            trait_consistency = 1.0 - np.var(trait_values)
            consistencies.append(max(0, trait_consistency))
        
        return np.mean(consistencies)
    
    def save_models(self, filepath: str):
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'fighting_regressor': self.fighting_regressor,
            'badminton_regressor': self.badminton_regressor,
            'racing_regressor': self.racing_regressor,
            'meta_regressor': self.meta_regressor,
            'personality_classifier': self.personality_classifier,
            'playstyle_classifier': self.playstyle_classifier,
            'scaler_fighting': self.scaler_fighting,
            'scaler_badminton': self.scaler_badminton,
            'scaler_racing': self.scaler_racing,
            'scaler_combined': self.scaler_combined,
            'feature_names': self.feature_names,
            'personality_traits': self.personality_traits,
            'personality_archetypes': self.personality_archetypes,
            'playstyle_categories': self.playstyle_categories,
            'is_trained': self.is_trained
        }
        
        if True: # Optimized save
            joblib.dump(model_data, filepath, compress=3)
        
        print(f"✓ ULTIMATE hybrid models saved to {filepath}")
    
    def load_models(self, filepath: str):
        try:
            if True: # Optimized load
                model_data = joblib.load(filepath)
            
            self.fighting_regressor = model_data['fighting_regressor']
            self.badminton_regressor = model_data['badminton_regressor']
            self.racing_regressor = model_data['racing_regressor']
            self.meta_regressor = model_data['meta_regressor']
            self.personality_classifier = model_data['personality_classifier']
            self.playstyle_classifier = model_data['playstyle_classifier']
            self.scaler_fighting = model_data['scaler_fighting']
            self.scaler_badminton = model_data['scaler_badminton']
            self.scaler_racing = model_data['scaler_racing']
            self.scaler_combined = model_data['scaler_combined']
            self.feature_names = model_data['feature_names']
            self.personality_traits = model_data['personality_traits']
            self.personality_archetypes = model_data['personality_archetypes']
            self.playstyle_categories = model_data['playstyle_categories']
            self.is_trained = model_data['is_trained']
            
            print(f"✓ ULTIMATE hybrid models loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False


if __name__ == "__main__":
    print(" Testing ULTIMATE Hybrid Personality System...")
    
    ultimate_classifier = CrossGamePersonalityClassifier()
    
    training_data = ultimate_classifier.generate_training_data(n_samples=10000)
    
    performance = ultimate_classifier.train_models(training_data)
    
    test_features = {
        'fighting': [0.9, 0.1, 0.8, 0.2],     
        'badminton': [0.6, 0.8, 0.7, 0.3],    
        'racing': [0.8, 0.7, 0.9, 0.6]       
    }
    
    prediction = ultimate_classifier.predict_personality(test_features)
    
    print(f"\n PREDICTION RESULTS:")
    print(f"Personality Type: {prediction['personality_archetype']}")
    print(f"Playstyle: {prediction['playstyle_category']}")
    print(f"Category Confidence: {prediction['category_confidence']:.1%}")
    print(f"Features Used: {prediction['ultimate_features_used']}")
    
    print(f"\n🔢 PRECISE PERSONALITY SCORES:")
    for trait, value in prediction['personality_scores'].items():
        print(f"  {trait}: {value:.3f}")
    
    print(f"\nOverall Confidence: {prediction['confidence_score']:.3f}")
    print(f"Cross-game Consistency: {prediction['cross_game_consistency']:.3f}")
    
    ultimate_classifier.save_models("hybrid_personality_system.pkl")
    print(f"\n💾 ULTIMATE models saved successfully!")
    
    print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
    print(f"  Regression R²: {performance['regression_scores']['unified_r2']:.3f}")
    print(f"  Personality Classification: {performance['classification_scores']['personality_accuracy']:.3f}")
    print(f"  Playstyle Classification: {performance['classification_scores']['playstyle_accuracy']:.3f}")
    print(f"  Minimum Category Accuracy: {performance['classification_scores']['min_category_accuracy']:.3f}")
    
    if performance['meets_targets']['ultimate_target']:
        print(f"  🏆 ULTIMATE TARGET ACHIEVED! (>95% classification)")
    elif performance['meets_targets']['classification_quality']:
        print(f"  ✓ Good classification quality achieved!")
    else:
        print(f"  🔧 Still improving classification...")
    
    


