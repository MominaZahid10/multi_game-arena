import asyncio
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from backend.dbmodels.personality import BehavioralFeature, UnifiedPersonality
from datetime import datetime
import os
import pickle

try:
  
    import sys
    sys.path.append('.')  
    from backend.services.model1 import CrossGamePersonalityClassifier
    HybridPersonalitySystem = CrossGamePersonalityClassifier  
except ImportError:
    print("❌ CrossGamePersonalityClassifier not found, using fallback")
    HybridPersonalitySystem = None

class MultiGameAnalyzer:
    def __init__(self):
        self.fighting_analyzer = FightingBehaviorAnalyzer()
        self.badminton_analyzer = BadmintonBehaviorAnalyzer()
        self.racing_analyzer = RacingBehaviorAnalyzer()
        
        self.hybrid_system = None
        if CrossGamePersonalityClassifier:
            self.hybrid_system = CrossGamePersonalityClassifier()
            self._load_trained_models()
    
    def _load_trained_models(self):
        """🔥 FIX 2: Force retraining to fix feature mismatch"""
        model_path = "hybrid_personality_system.pkl"
        
        force_retrain = False
        
        if os.path.exists(model_path):
            try:
                # Try to load and validate the model
                success = self.hybrid_system.load_models(model_path)
                if success:
                    # Test the model with sample data to check feature count
                    try:
                        test_features = {
                            'fighting': [0.5, 0.5, 0.5, 0.5],
                            'badminton': [0.5, 0.5, 0.5, 0.5],
                            'racing': [0.5, 0.5, 0.5, 0.5]
                        }
                        # This will fail if feature count is wrong
                        _ = self.hybrid_system.predict_personality(test_features)
                        print("✅ Loaded existing hybrid personality models successfully")
                        return
                    except Exception as feature_error:
                        print(f"🔧 Feature mismatch detected: {feature_error}")
                        print("📚 This usually means the model was trained with different features")
                        force_retrain = True
                else:
                    force_retrain = True
            except Exception as e:
                print(f"❌ Failed to load hybrid models: {e}")
                force_retrain = True
        else:
            print("🆕 No pre-trained hybrid models found.")
            force_retrain = True
        
        if force_retrain:
            print("🔄 Training new hybrid models with correct feature engineering...")
            self._train_new_models()
    
    def _train_new_models(self):
        """Train new hybrid ML models with correct feature engineering"""
        if not self.hybrid_system:
            print("❌ CrossGamePersonalityClassifier not available")
            return
            
        try:
            print("🏗️ Generating fresh training data...")
            # Generate training data using your model's method
            training_data = self.hybrid_system.generate_training_data(n_samples=5000)
            
            print("🤖 Training hybrid models...")
            # Train models using your model's method
            performance = self.hybrid_system.train_models(training_data)
            
            print("💾 Saving trained models...")
            # Save models using your model's method
            self.hybrid_system.save_models("hybrid_personality_system.pkl")
            
            print(f"✅ New hybrid models trained and saved successfully!")
            print(f"  📊 Regression R²: {performance['regression_scores']['unified_r2']:.3f}")
            print(f"  🎯 Classification Accuracy: {performance['classification_scores']['personality_accuracy']:.3f}")
            print(f"  📈 Min Category Accuracy: {performance['classification_scores']['min_category_accuracy']:.3f}")
            
            # Test the newly trained model
            try:
                test_features = {
                    'fighting': [0.8, 0.2, 0.6, 0.3],
                    'badminton': [0.7, 0.8, 0.5, 0.4],
                    'racing': [0.9, 0.7, 0.8, 0.6]
                }
                test_prediction = self.hybrid_system.predict_personality(test_features)
                print(f"🧪 Model test successful - Features used: {test_prediction.get('ultimate_features_used', 'N/A')}")
            except Exception as test_error:
                print(f"⚠️ Model test failed: {test_error}")
            
        except Exception as e:
            print(f"❌ Hybrid model training failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to rule-based analysis
            self.hybrid_system = None
    
    async def analyze_universal_behavior(self, actions: Dict[str, List]):
        """Updated main analysis method with better error handling"""
        
        # Extract behavioral features using existing analyzers
        all_features = []
        game_ml_features = {}
        
        print(f"🔍 Analyzing actions from {len(actions)} games...")
        
        # Fighting game analysis
        if actions.get("fighting"):
            print(f"⚔️ Processing {len(actions['fighting'])} fighting actions...")
            try:
                fighting_features = await self.fighting_analyzer.extract_features(actions["fighting"])
                all_features.extend(fighting_features)
                game_ml_features['fighting'] = self._extract_ml_features_fighting(actions["fighting"])
                print(f"✅ Fighting features: {game_ml_features['fighting']}")
            except Exception as e:
                print(f"❌ Fighting analysis failed: {e}")
                
        # Badminton game analysis  
        if actions.get("badminton"):
            print(f"🏸 Processing {len(actions['badminton'])} badminton actions...")
            try:
                badminton_features = await self.badminton_analyzer.extract_features(actions["badminton"])
                all_features.extend(badminton_features)
                game_ml_features['badminton'] = self._extract_ml_features_badminton(actions["badminton"])
                print(f"✅ Badminton features: {game_ml_features['badminton']}")
            except Exception as e:
                print(f"❌ Badminton analysis failed: {e}")
                
        # Racing game analysis
        if actions.get("racing"):
            print(f"🏎️ Processing {len(actions['racing'])} racing actions...")
            try:
                racing_features = await self.racing_analyzer.extract_features(actions["racing"])
                all_features.extend(racing_features)
                game_ml_features['racing'] = self._extract_ml_features_racing(actions["racing"])
                print(f"✅ Racing features: {game_ml_features['racing']}")
            except Exception as e:
                print(f"❌ Racing analysis failed: {e}")
        
        # Use hybrid ML system for personality prediction if available
        if self.hybrid_system and self.hybrid_system.is_trained and game_ml_features:
            try:
                print(f"🤖 Using hybrid ML prediction with {len(game_ml_features)} games...")
                # Get hybrid prediction using your model's predict_personality method
                hybrid_prediction = self.hybrid_system.predict_personality(game_ml_features)
                
                # Convert hybrid prediction to UnifiedPersonality format
                personality_scores = hybrid_prediction['personality_scores']
                
                unified_profile = UnifiedPersonality(
                    aggression_level=personality_scores['aggression_level'],
                    risk_tolerance=personality_scores['risk_tolerance'],
                    analytical_thinking=personality_scores['analytical_thinking'],
                    patience_level=personality_scores['patience_level'],
                    precision_focus=personality_scores['precision_focus'],
                    competitive_drive=personality_scores['competitive_drive'],
                    strategic_thinking=personality_scores['strategic_thinking'],
                    adaptability=np.mean([personality_scores['strategic_thinking'], 
                                        personality_scores['analytical_thinking']]),
                    confidence_score=hybrid_prediction['confidence_score'],
                    total_actions_analyzed=sum(len(actions[game]) for game in actions),
                    games_played=hybrid_prediction['games_analyzed'],
                    last_updated=datetime.now(),
                    # 🔥 FIX 3: Set the missing fields
                    personality_archetype=hybrid_prediction.get('personality_archetype', '🎮 Multi-Game Player'),
                    playstyle_category=hybrid_prediction.get('playstyle_category', '🎯 Adaptive Gamer'),
                    category_confidence=hybrid_prediction.get('category_confidence', 0.0)
                )
                
                print(f"✅ Hybrid ML personality analysis complete!")
                print(f"  🏷️ Type: {unified_profile.personality_archetype}")
                print(f"  🎮 Style: {unified_profile.playstyle_category}")
                print(f"  📊 Confidence: {unified_profile.confidence_score:.3f}")
                print(f"  🔄 Consistency: {hybrid_prediction.get('cross_game_consistency', 0.0):.3f}")
                
                return unified_profile
                
            except Exception as e:
                print(f"❌ Hybrid ML prediction failed, using rule-based fallback: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to rule-based analysis
        print(f"📋 Using rule-based analysis fallback...")
        personality_synthesizer = PersonalitySynthesizer()
        unified_profile = await personality_synthesizer.synthesize(all_features)
        
        # Add default categories for fallback
        unified_profile.personality_archetype = "🎮 Classic Gamer"
        unified_profile.playstyle_category = "📊 Rule-Based Analysis"
        unified_profile.category_confidence = unified_profile.confidence_score
        
        print(f"✅ Rule-based analysis complete")
        return unified_profile
    
    def _extract_ml_features_fighting(self, actions: List[Any]) -> List[float]:
        """Extract ML features for fighting game (exactly matching your model's expectations)"""
        if not actions:
            return [0.5, 0.5, 0.5, 0.5]  # Default neutral features
        
        # Your model expects: [aggression_rate, defense_ratio, combo_preference, reaction_time]
        attack_actions = sum(1 for a in actions if getattr(a, 'move_type', '') == 'attack')
        aggression_rate = min(1.0, attack_actions / len(actions))
        
        block_actions = sum(1 for a in actions if getattr(a, 'move_type', '') == 'block')
        defense_ratio = min(1.0, block_actions / len(actions))
        
        combo_total = sum(getattr(a, 'combo_count', 0) for a in actions)
        combo_preference = min(1.0, combo_total / (len(actions) * 3))  # Normalize
        
        # Reaction time (inverse of success rate as proxy)
        success_rate = sum(1 for a in actions if getattr(a, 'success', False)) / len(actions)
        reaction_time = max(0.1, min(1.0, 1.0 - success_rate))
        
        return [aggression_rate, defense_ratio, combo_preference, reaction_time]
    
    def _extract_ml_features_badminton(self, actions: List[Any]) -> List[float]:
        """Extract ML features for badminton game (exactly matching your model's expectations)"""
        if not actions:
            return [0.5, 0.5, 0.5, 0.5]
        
        # Your model expects: [shot_variety, power_control, court_positioning, rally_patience]
        shot_types = set(getattr(a, 'shot_type', 'clear') for a in actions)
        shot_variety = min(1.0, len(shot_types) / 5.0)  # Max 5 shot types
        
        # Power control (consistency in power levels)
        power_levels = [getattr(a, 'power_level', 0.5) for a in actions]
        if power_levels:
            power_variance = np.var(power_levels)
            power_control = max(0.1, min(1.0, 1.0 - power_variance))
        else:
            power_control = 0.5
        
        # Court positioning (improved calculation)
        court_positions = [getattr(a, 'court_position', (0.5, 0.5)) for a in actions]
        if court_positions:
            # Calculate positioning variety as a proxy for strategic thinking
            valid_positions = [pos for pos in court_positions if isinstance(pos, (tuple, list)) and len(pos) >= 2]
            if valid_positions:
                position_variance = np.var([pos[0] + pos[1] for pos in valid_positions])
                court_positioning = max(0.1, min(1.0, position_variance * 10))  # Scale variance
            else:
                court_positioning = 0.5
        else:
            court_positioning = 0.5
        
        # Rally patience (rally position average)
        rally_positions = [max(1, getattr(a, 'rally_position', 1)) for a in actions]
        rally_patience = min(1.0, max(0.1, np.mean(rally_positions) / 10.0)) if rally_positions else 0.5
        
        return [shot_variety, power_control, court_positioning, rally_patience]
    
    def _extract_ml_features_racing(self, actions: List[Any]) -> List[float]:
        """Extract ML features for racing game (exactly matching your model's expectations)"""
        if not actions:
            return [0.5, 0.5, 0.5, 0.5]
        
        # Your model expects: [speed_preference, precision_level, overtaking_aggression, consistency]
        speeds = [getattr(a, 'speed', 50) for a in actions if getattr(a, 'speed', 0) > 0]
        if speeds:
            max_reasonable_speed = 120  # Assume max reasonable speed
            speed_preference = min(1.0, max(0.1, np.mean(speeds) / max_reasonable_speed))
        else:
            speed_preference = 0.5
        
        # Precision level (inverse of crashes and consistency)
        crashes = sum(1 for a in actions if getattr(a, 'crash_occurred', False))
        crash_rate = crashes / len(actions)
        precision_level = max(0.1, min(1.0, 1.0 - crash_rate))
        
        # Overtaking aggression
        overtakes = sum(1 for a in actions if getattr(a, 'overtaking_attempt', False))
        overtaking_aggression = min(1.0, overtakes / len(actions))
        
        # Consistency (speed variance)
        if speeds and len(speeds) > 1:
            speed_variance = np.var(speeds)
            max_variance = 400  # Assume max reasonable variance
            consistency = max(0.1, min(1.0, 1.0 - speed_variance / max_variance))
        else:
            consistency = 0.5
        
        return [speed_preference, precision_level, overtaking_aggression, consistency]
    
    def get_cross_game_insights(self):
        """Enhanced cross-game insights with hybrid ML integration"""
        base_insights = {
            "consistency_analysis": "Personality traits analyzed with hybrid ML system",
            "transfer_learning": "Cross-game behavioral patterns detected",
            "adaptation_speed": "Real-time personality synthesis active",
            "dominant_traits": ["competitive_drive", "risk_tolerance", "precision_focus"]
        }
        
        if self.hybrid_system and self.hybrid_system.is_trained:
            base_insights["ml_integration"] = "CrossGamePersonalityClassifier active (Regression + Classification)"
            base_insights["precise_scoring"] = "Continuous personality trait values available"
            base_insights["category_display"] = "8 personality archetypes + 6 playstyle categories"
            base_insights["model_quality"] = "Ultimate ensemble with 36+ engineered features"
        else:
            base_insights["ml_integration"] = "Rule-based fallback analysis active"
            
        return base_insights

    def get_personality_display_data(self, unified_personality: UnifiedPersonality) -> Dict[str, Any]:
        """Get data formatted for impressive frontend display"""
        display_data = {
            "impressive_categories": {
                "personality_type": getattr(unified_personality, 'personality_archetype', '🎮 Multi-Game Player'),
                "playstyle": getattr(unified_personality, 'playstyle_category', '🎯 Adaptive Gamer'),
                "category_confidence": getattr(unified_personality, 'category_confidence', unified_personality.confidence_score)
            },
            "precise_scores": {
                "aggression_level": unified_personality.aggression_level,
                "risk_tolerance": unified_personality.risk_tolerance,
                "analytical_thinking": unified_personality.analytical_thinking,
                "patience_level": unified_personality.patience_level,
                "precision_focus": unified_personality.precision_focus,
                "competitive_drive": unified_personality.competitive_drive,
                "strategic_thinking": unified_personality.strategic_thinking
            },
            "meta_info": {
                "confidence": unified_personality.confidence_score,
                "games_analyzed": len(unified_personality.games_played),
                "total_actions": unified_personality.total_actions_analyzed,
                "analysis_type": "CrossGame ML Classifier" if self.hybrid_system and self.hybrid_system.is_trained else "Rule-based"
            }
        }
        
        return display_data

# Keep your existing analyzers (they're working well)
class FightingBehaviorAnalyzer:
    """Enhanced fighting game behavioral analysis"""
    
    async def extract_features(self, actions: List[Any]) -> List[BehavioralFeature]:
        if not actions:
            return []
        
        features = []
        
        # Aggression analysis
        attack_actions = sum(1 for a in actions if getattr(a, 'move_type', '') == 'attack')
        aggression_score = attack_actions / len(actions) if actions else 0
        features.append(BehavioralFeature(
            name="aggression_level",
            value=min(1.0, aggression_score * 1.2),  # Scale up for fighting context
            confidence=min(1.0, len(actions) / 20),
            game_source="fighting",
            description=f"Based on {attack_actions}/{len(actions)} attack moves"
        ))
        
        # Risk tolerance from combo usage
        combo_actions = sum(getattr(a, 'combo_count', 0) for a in actions)
        avg_combo = combo_actions / len(actions) if actions else 0
        risk_score = min(1.0, avg_combo / 5.0)  # Normalize combo count
        features.append(BehavioralFeature(
            name="risk_tolerance",
            value=risk_score,
            confidence=min(1.0, len(actions) / 15),
            game_source="fighting",
            description=f"Average combo length: {avg_combo:.1f}"
        ))
        
        # Defensive behavior analysis
        block_actions = sum(1 for a in actions if getattr(a, 'move_type', '') == 'block')
        patience_score = block_actions / len(actions) if actions else 0
        features.append(BehavioralFeature(
            name="patience_level",
            value=patience_score,
            confidence=min(1.0, len(actions) / 15),
            game_source="fighting",
            description=f"Defensive moves: {block_actions}/{len(actions)}"
        ))
        
        # Strategic thinking from movement patterns
        move_actions = sum(1 for a in actions if getattr(a, 'move_type', '') == 'move')
        strategic_score = move_actions / len(actions) if actions else 0
        features.append(BehavioralFeature(
            name="strategic_thinking",
            value=strategic_score,
            confidence=min(1.0, len(actions) / 12),
            game_source="fighting",
            description=f"Strategic positioning: {move_actions}/{len(actions)}"
        ))
        
        return features

class BadmintonBehaviorAnalyzer:
    """Enhanced badminton game behavioral analysis"""
    
    async def extract_features(self, actions: List[Any]) -> List[BehavioralFeature]:
        if not actions:
            return []
        
        features = []
        
        # Strategic thinking from shot variety
        shot_types = set(getattr(a, 'shot_type', 'unknown') for a in actions)
        strategy_score = len(shot_types) / 5.0  # Max 5 shot types
        features.append(BehavioralFeature(
            name="strategic_thinking",
            value=min(1.0, strategy_score),
            confidence=min(1.0, len(actions) / 15),
            game_source="badminton",
            description=f"Uses {len(shot_types)} different shot types"
        ))
        
        # Precision from power control
        power_levels = [getattr(a, 'power_level', 0.5) for a in actions]
        power_variance = np.var(power_levels) if power_levels else 0
        precision_score = max(0, 1.0 - power_variance * 2)  # Lower variance = higher precision
        features.append(BehavioralFeature(
            name="precision_focus",
            value=precision_score,
            confidence=min(1.0, len(actions) / 12),
            game_source="badminton",
            description=f"Power control variance: {power_variance:.3f}"
        ))
        
        # Competitive drive from smash frequency
        smash_count = sum(1 for a in actions if getattr(a, 'shot_type', '') == 'smash')
        competitive_score = min(1.0, smash_count / len(actions) * 2)
        features.append(BehavioralFeature(
            name="competitive_drive",
            value=competitive_score,
            confidence=min(1.0, len(actions) / 10),
            game_source="badminton",
            description=f"Aggressive shots: {smash_count}/{len(actions)}"
        ))
        
        # Patience from rally length
        rally_positions = [getattr(a, 'rally_position', 1) for a in actions]
        avg_rally_pos = np.mean(rally_positions) if rally_positions else 1
        patience_score = min(1.0, avg_rally_pos / 10.0)
        features.append(BehavioralFeature(
            name="patience_level",
            value=patience_score,
            confidence=min(1.0, len(actions) / 15),
            game_source="badminton",
            description=f"Average rally position: {avg_rally_pos:.1f}"
        ))
        
        return features

class RacingBehaviorAnalyzer:
    """Enhanced racing game behavioral analysis"""
    
    async def extract_features(self, actions: List[Any]) -> List[BehavioralFeature]:
        if not actions:
            return []
        
        features = []
        
        # Risk tolerance from overtaking
        overtake_attempts = sum(1 for a in actions if getattr(a, 'overtaking_attempt', False))
        risk_score = overtake_attempts / len(actions) if actions else 0
        features.append(BehavioralFeature(
            name="risk_tolerance",
            value=min(1.0, risk_score * 2),  # Scale up for racing context
            confidence=min(1.0, len(actions) / 25),
            game_source="racing",
            description=f"Overtaking attempts: {overtake_attempts}/{len(actions)}"
        ))
        
        # Precision from speed consistency
        speeds = [getattr(a, 'speed', 0) for a in actions if getattr(a, 'speed', 0) > 0]
        if speeds:
            speed_variance = np.var(speeds)
            precision_score = max(0, 1.0 - speed_variance / 1000)  # Normalize variance
            features.append(BehavioralFeature(
                name="precision_focus",
                value=precision_score,
                confidence=min(1.0, len(speeds) / 20),
                game_source="racing",
                description=f"Speed consistency (variance: {speed_variance:.1f})"
            ))
        
        # Analytical thinking from crash avoidance
        crashes = sum(1 for a in actions if getattr(a, 'crash_occurred', False))
        analytical_score = max(0, 1.0 - crashes / len(actions) * 2)
        features.append(BehavioralFeature(
            name="analytical_thinking",
            value=analytical_score,
            confidence=min(1.0, len(actions) / 15),
            game_source="racing",
            description=f"Crash rate: {crashes}/{len(actions)}"
        ))
        
        # Competitive drive from speed preference
        if speeds:
            avg_speed = np.mean(speeds)
            competitive_score = min(1.0, avg_speed / 100.0)  # Normalize to 0-1
            features.append(BehavioralFeature(
                name="competitive_drive",
                value=competitive_score,
                confidence=min(1.0, len(speeds) / 15),
                game_source="racing",
                description=f"Average speed preference: {avg_speed:.1f}"
            ))
        
        return features

class PersonalitySynthesizer:
    """Enhanced personality synthesis with ML fallback"""
    
    async def synthesize(self, features: List[BehavioralFeature]) -> UnifiedPersonality:
        """Main synthesis method - fallback for when ML models aren't available"""
        # Initialize base personality
        personality_traits = {
            "aggression_level": 0.5,
            "risk_tolerance": 0.5,
            "analytical_thinking": 0.5,
            "patience_level": 0.5,
            "precision_focus": 0.5,
            "competitive_drive": 0.5,
            "strategic_thinking": 0.5,
            "adaptability": 0.5
        }
        
        # Group features by trait name
        trait_groups = {}
        for feature in features:
            if feature.name not in trait_groups:
                trait_groups[feature.name] = []
            trait_groups[feature.name].append(feature)
        
        # Synthesize each trait with confidence weighting
        overall_confidence = 0.0
        total_actions = len(features)
        
        for trait_name, trait_features in trait_groups.items():
            if trait_name in personality_traits:
                # Confidence-weighted average
                total_weight = sum(f.confidence for f in trait_features)
                if total_weight > 0:
                    weighted_value = sum(f.value * f.confidence for f in trait_features) / total_weight
                    personality_traits[trait_name] = weighted_value
                    overall_confidence += total_weight / len(trait_features)
        
        # Calculate final confidence
        final_confidence = min(1.0, overall_confidence / len(personality_traits))
        
        return UnifiedPersonality(
            **personality_traits,
            confidence_score=final_confidence,
            total_actions_analyzed=total_actions,
            games_played=list(set(f.game_source for f in features)),
            last_updated=datetime.now()
        )