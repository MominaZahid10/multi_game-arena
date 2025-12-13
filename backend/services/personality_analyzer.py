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
        """Load trained models without aggressive retraining - FIXED VERSION"""
        model_path = "hybrid_personality_system.pkl"
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"📂 Found model file: {file_size / (1024*1024):.1f}MB")
            
            # Check if file is reasonable size (not corrupted)
            if file_size < 1024:  # Less than 1KB means corrupted
                print("⚠️ Model file too small, appears corrupted. Retraining...")
                self._train_new_models()
                return
            
            try:
                # Simply load the model without aggressive testing
                success = self.hybrid_system.load_models(model_path)
                if success:
                    print("✅ Loaded existing hybrid personality models successfully")
                    print("🚀 Ready for ML-powered personality analysis!")
                    return
                else:
                    print("❌ Model loading failed, retraining...")
                    self._train_new_models()
            except Exception as e:
                print(f"❌ Failed to load hybrid models: {e}")
                print("🔄 Falling back to retraining...")
                self._train_new_models()
        else:
            print("🆕 No pre-trained hybrid models found.")
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
            
            # Optional: Test the newly trained model (but don't retrain if it fails)
            try:
                test_features = {
                    'fighting': [0.8, 0.2, 0.6, 0.3],
                    'badminton': [0.7, 0.8, 0.5, 0.4],
                    'racing': [0.9, 0.7, 0.8, 0.6]
                }
                test_prediction = self.hybrid_system.predict_personality(test_features)
                print(f"🧪 Model test successful - Features used: {test_prediction.get('ultimate_features_used', 'N/A')}")
            except Exception as test_error:
                print(f"⚠️ Model test had issues but continuing anyway: {test_error}")
            
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
                    # Set the missing fields
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
        # Initialize the rule-based personality synthesizer
        personality_synthesizer = PersonalitySynthesizer()
        
        # Log that we're using the rule-based fallback
        print(f"🔄 Using rule-based personality analysis with {len(all_features)} features")
        
        # Determine which games were analyzed
        games_analyzed = set()
        for feature in all_features:
            if hasattr(feature, 'game_source') and feature.game_source:
                games_analyzed.add(feature.game_source)
        
        print(f"🎮 Games analyzed: {', '.join(games_analyzed)}")
        
        # Synthesize the personality profile using rule-based approach
        try:
            unified_profile = await personality_synthesizer.synthesize(all_features)
            
            print(f"✅ Rule-based analysis complete")
            print(f"  🏷️ Type: {unified_profile.personality_archetype}")
            print(f"  🎮 Style: {unified_profile.playstyle_category}")
            print(f"  📊 Confidence: {unified_profile.confidence_score:.3f}")
            print(f"  🔄 Category Confidence: {unified_profile.category_confidence:.3f}")
        except Exception as e:
            print(f"❌ Error in rule-based personality synthesis: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a default personality profile as ultimate fallback
            print(f"⚠️ Creating default personality profile as ultimate fallback")
            unified_profile = UnifiedPersonality(
                aggression_level=0.5,
                risk_tolerance=0.5,
                analytical_thinking=0.5,
                patience_level=0.5,
                precision_focus=0.5,
                competitive_drive=0.5,
                strategic_thinking=0.5,
                adaptability=0.5,
                confidence_score=0.3,  # Low confidence since this is a default
                total_actions_analyzed=len(all_features),
                games_played=list(games_analyzed),
                last_updated=datetime.now(),
                personality_archetype="🎮 Balanced Player",
                playstyle_category="🎯 Adaptive Gamer",
                category_confidence=0.3
            )
        
        return unified_profile
    
    def _extract_ml_features_fighting(self, actions: List[Any]) -> List[float]:
        """Extract ML features for fighting game - FIXED to use actual action data"""
        if not actions:
          return [0.5, 0.5, 0.5, 0.5]  # Default neutral features
    
        print(f"🔍 Extracting fighting features from {len(actions)} actions")
    
    # Analyze actual action patterns instead of hard-coded values
        attack_count = 0
        defend_count = 0
        combo_total = 0
        success_count = 0
    
        for action in actions:
        # Handle different action data structures
          action_type = None
          success = False
          combo_count = 0
        
          if isinstance(action, dict):
            action_type = action.get('action_type')
            success = action.get('success', False)
            combo_count = action.get('combo_count', 0)
          else:
            # Handle object attributes
            action_type = getattr(action, 'action_type', None)
            success = getattr(action, 'success', False)
            combo_count = getattr(action, 'combo_count', 0)
        
        # Count attack vs defense actions
          if action_type in ['attack', 'punch', 'kick', 'combo', 'special_move']:
            attack_count += 1
          elif action_type in ['block', 'dodge', 'counter']:
            defend_count += 1
            
          if success:
            success_count += 1
            
          combo_total += combo_count
    
    # Calculate dynamic features
        total_actions = len(actions)
        aggression_rate = attack_count / total_actions if total_actions > 0 else 0.5
        defense_ratio = defend_count / total_actions if total_actions > 0 else 0.5
        combo_preference = combo_total / (total_actions * 3) if total_actions > 0 else 0.5  # Normalize
        reaction_time = max(0.1, min(1.0, success_count / total_actions)) if total_actions > 0 else 0.5
    
        features = [aggression_rate, defense_ratio, combo_preference, reaction_time]
        print(f"📊 Fighting features calculated: {features}")
    
        return features
    
    def _extract_ml_features_badminton(self, actions: List[Any]) -> List[float]:
      """Extract ML features for badminton - FIXED with dynamic analysis"""
      if not actions:
        return [0.5, 0.5, 0.5, 0.5]
    
      print(f"🏸 Extracting badminton features from {len(actions)} actions")
    
      shot_types = set()
      power_levels = []
      court_positions = []
      rally_positions = []
    
      for action in actions:
        # Handle different data structures
        if isinstance(action, dict):
            shot_type = action.get('shot_type')
            power_level = action.get('power_level')
            court_position = action.get('court_position')
            rally_position = action.get('rally_position')
        else:
            shot_type = getattr(action, 'shot_type', None)
            power_level = getattr(action, 'power_level', None)
            court_position = getattr(action, 'court_position', None)
            rally_position = getattr(action, 'rally_position', None)
        
        if shot_type:
            shot_types.add(shot_type)
        if power_level is not None:
            power_levels.append(power_level)
        if court_position:
            court_positions.append(court_position)
        if rally_position:
            rally_positions.append(rally_position)
    
    # Calculate features
      shot_variety = len(shot_types) / 5.0 if shot_types else 0.5  # Max 5 shot types
    
      if power_levels:
        power_variance = np.var(power_levels)
        power_control = max(0.1, min(1.0, 1.0 - power_variance))
      else:
        power_control = 0.5
    
    # Court positioning analysis
      if court_positions:
        # Calculate movement variety as strategic thinking indicator
        if len(court_positions) > 1:
            position_changes = 0
            for i in range(1, len(court_positions)):
                prev_pos = court_positions[i-1]
                curr_pos = court_positions[i]
                if isinstance(prev_pos, (list, tuple)) and isinstance(curr_pos, (list, tuple)):
                    if len(prev_pos) >= 2 and len(curr_pos) >= 2:
                        distance = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
                        if distance > 0.5:  # Significant movement
                            position_changes += 1
            court_positioning = min(1.0, position_changes / len(court_positions))
        else:
            court_positioning = 0.5
      else:
        court_positioning = 0.5
    
      rally_patience = np.mean(rally_positions) / 10.0 if rally_positions else 0.5
      rally_patience = min(1.0, max(0.1, rally_patience))
    
      features = [shot_variety, power_control, court_positioning, rally_patience]
      print(f"🏸 Badminton features calculated: {features}")
    
      return features
    
    def _extract_ml_features_racing(self, actions: List[Any]) -> List[float]:
      """Extract ML features for racing - FIXED with dynamic analysis"""
      if not actions:
        return [0.5, 0.5, 0.5, 0.5]
    
      print(f"🏁 Extracting racing features from {len(actions)} actions")
    
      speeds = []
      crashes = 0
      overtake_attempts = 0
      total_actions = len(actions)
    
      for action in actions:
        # Handle different data structures
        if isinstance(action, dict):
            speed = action.get('speed')
            crash_occurred = action.get('crash_occurred', False)
            overtaking_attempt = action.get('overtaking_attempt', False)
        else:
            speed = getattr(action, 'speed', None)
            crash_occurred = getattr(action, 'crash_occurred', False)
            overtaking_attempt = getattr(action, 'overtaking_attempt', False)
        
        if speed is not None and speed > 0:
            speeds.append(speed)
        if crash_occurred:
            crashes += 1
        if overtaking_attempt:
            overtake_attempts += 1
    
    # Calculate features
      if speeds:
        speed_preference = np.mean(speeds) / 120.0  # Normalize to max reasonable speed
        speed_preference = min(1.0, max(0.1, speed_preference))
        
        # Consistency from speed variance
        if len(speeds) > 1:
            speed_variance = np.var(speeds)
            consistency = max(0.1, min(1.0, 1.0 - speed_variance / 400))
        else:
            consistency = 0.5
      else:
        speed_preference = 0.5
        consistency = 0.5
    
    # Precision from crash avoidance
      crash_rate = crashes / total_actions if total_actions > 0 else 0
      precision_level = max(0.1, min(1.0, 1.0 - crash_rate))
    
    # Overtaking aggression
      overtaking_aggression = overtake_attempts / total_actions if total_actions > 0 else 0
      overtaking_aggression = min(1.0, overtaking_aggression)
    
      features = [speed_preference, precision_level, overtaking_aggression, consistency]
      print(f"🏁 Racing features calculated: {features}")
    
      return features
    
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
    
    def __init__(self):
        # Define personality archetypes with their trait thresholds
        self.personality_archetypes = {
            "🔥 Aggressive Dominator": {
                "primary_traits": ["aggression_level", "competitive_drive"],
                "threshold": 0.7
            },
            "🧠 Strategic Analyst": {
                "primary_traits": ["analytical_thinking", "strategic_thinking"],
                "threshold": 0.7
            },
            "⚡ Risk-Taking Maverick": {
                "primary_traits": ["risk_tolerance", "aggression_level"],
                "threshold": 0.7
            },
            "🛡️ Defensive Tactician": {
                "primary_traits": ["patience_level", "strategic_thinking"],
                "threshold": 0.7
            },
            "🎯 Precision Master": {
                "primary_traits": ["precision_focus", "analytical_thinking"],
                "threshold": 0.7
            },
            "🌪️ Chaos Creator": {
                "primary_traits": ["risk_tolerance", "aggression_level"],
                "threshold": 0.7,
                "negative_traits": ["precision_focus", "analytical_thinking"],
                "negative_threshold": 0.3
            },
            "📊 Data-Driven Player": {
                "primary_traits": ["analytical_thinking", "precision_focus"],
                "threshold": 0.7
            },
            "🏆 Victory Seeker": {
                "primary_traits": ["competitive_drive", "strategic_thinking"],
                "threshold": 0.7
            }
        }
        
        # Define playstyle categories with their game-specific traits
        self.playstyle_categories = {
            "🥊 Combat Veteran": {
                "game_focus": "fighting",
                "traits": ["aggression_level", "risk_tolerance"]
            },
            "🏸 Court Strategist": {
                "game_focus": "badminton",
                "traits": ["strategic_thinking", "precision_focus"]
            },
            "🏎️ Speed Demon": {
                "game_focus": "racing",
                "traits": ["competitive_drive", "risk_tolerance"]
            },
            "🎮 Multi-Game Master": {
                "game_focus": "all",
                "traits": ["adaptability", "strategic_thinking"]
            },
            "🧩 Adaptive Learner": {
                "game_focus": "all",
                "traits": ["analytical_thinking", "adaptability"]
            },
            "🔄 Pattern Seeker": {
                "game_focus": "all",
                "traits": ["analytical_thinking", "precision_focus"]
            }
        }
    
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
        
        # Determine personality archetype
        personality_archetype, archetype_confidence = self._determine_personality_archetype(personality_traits)
        
        # Determine playstyle category
        playstyle_category, playstyle_confidence = self._determine_playstyle_category(
            personality_traits, 
            list(set(f.game_source for f in features))
        )
        
        # Calculate category confidence (average of archetype and playstyle confidence)
        category_confidence = (archetype_confidence + playstyle_confidence) / 2
        
        return UnifiedPersonality(
            **personality_traits,
            confidence_score=final_confidence,
            total_actions_analyzed=total_actions,
            games_played=list(set(f.game_source for f in features)),
            last_updated=datetime.now(),
            personality_archetype=personality_archetype,
            playstyle_category=playstyle_category,
            category_confidence=category_confidence
        )
    
    def _determine_personality_archetype(self, traits: Dict[str, float]) -> tuple:
        """Determine the best matching personality archetype based on traits"""
        best_match = None
        best_score = 0.0
        
        for archetype, criteria in self.personality_archetypes.items():
            # Calculate match score based on primary traits
            primary_traits = criteria["primary_traits"]
            threshold = criteria["threshold"]
            
            # Calculate average of primary traits
            primary_avg = sum(traits[trait] for trait in primary_traits) / len(primary_traits)
            match_score = primary_avg
            
            # Apply negative trait penalties if defined
            if "negative_traits" in criteria and "negative_threshold" in criteria:
                negative_traits = criteria["negative_traits"]
                negative_threshold = criteria["negative_threshold"]
                
                # Check if negative traits are below threshold (as expected)
                negative_avg = sum(traits[trait] for trait in negative_traits) / len(negative_traits)
                if negative_avg > negative_threshold:
                    # Penalize score if negative traits are too high
                    match_score *= (1 - (negative_avg - negative_threshold))
            
            # Update best match if this is better
            if match_score > best_score and primary_avg >= threshold:
                best_score = match_score
                best_match = archetype
        
        # Default if no good match found
        if not best_match:
            return "🎮 Balanced Player", 0.5
        
        return best_match, best_score
    
    def _determine_playstyle_category(self, traits: Dict[str, float], games_played: List[str]) -> tuple:
        """Determine the best matching playstyle category based on traits and games played"""
        best_match = None
        best_score = 0.0
        
        for playstyle, criteria in self.playstyle_categories.items():
            game_focus = criteria["game_focus"]
            style_traits = criteria["traits"]
            
            # Skip if game-specific and not played
            if game_focus != "all" and game_focus not in games_played:
                continue
            
            # Calculate match score based on traits
            trait_avg = sum(traits[trait] for trait in style_traits) / len(style_traits)
            
            # Boost score if this is a game-specific category and that game was played
            game_bonus = 0.2 if game_focus in games_played and game_focus != "all" else 0.0
            match_score = trait_avg + game_bonus
            
            # Update best match if this is better
            if match_score > best_score:
                best_score = match_score
                best_match = playstyle
        
        # Default if no good match or no games played
        if not best_match or not games_played:
            return "🎯 Adaptive Gamer", 0.5
        
        return best_match, best_score