# Enhanced rulebased_ai.py - ML-POWERED VERSION

import numpy as np
import time
from typing import Dict, Any, Tuple
from backend.dbmodels.personality import GameType, UnifiedPersonality

class MLPoweredAIOpponent:
    """
    🤖 ML-POWERED AI - Uses trained personality classifier + dynamic strategy
    Enhanced with predictive movement and aggressive pursuit
    """
    
    def __init__(self):
        self.ml_classifier = None
        self.player_personality_cache = {}
        self.action_history = []
        self.strategy_effectiveness = {}
        self.ml_model_loaded = False  # Track if ML is actually working
        
        # ✅ OPTIMIZED: Reduced ring buffer from 5 to 2 for less overhead
        self.last_player_positions = []  # Ring buffer of last 2 positions
        self.player_velocity = {'x': 0, 'z': 0}
        self.last_update_time = 0
        
        # ✅ SERVER-SIDE AI POSITION TRACKING (fallback for stale client data)
        self.server_ai_position = {'x': 4.5, 'z': 0}  # Initialize at spawn
        self.last_known_target = {'x': 4.5, 'z': 0}   # Last target sent
        
        # ✅ CACHE: Store ML predictions to reduce computation
        self._cached_archetype = None
        self._cache_time = 0
        self._cache_duration = 0.2  # ✅ REDUCED to 200ms for responsive AI
        
        # ✅ ML-derived action mappings per archetype (more aggressive overall)
        self.archetype_strategies = {
            "🔥 Aggressive Dominator": {"punch": 0.35, "kick": 0.30, "combo_attack": 0.30, "block": 0.05},
            "🧠 Strategic Analyst": {"punch": 0.30, "kick": 0.30, "block": 0.25, "combo_attack": 0.15},
            "⚡ Risk-Taking Maverick": {"combo_attack": 0.40, "kick": 0.30, "punch": 0.25, "block": 0.05},
            "🛡️ Defensive Tactician": {"block": 0.35, "punch": 0.30, "kick": 0.25, "combo_attack": 0.10},
            "🎯 Precision Master": {"punch": 0.40, "kick": 0.35, "block": 0.15, "combo_attack": 0.10},
            "🌪️ Chaos Creator": {"combo_attack": 0.35, "kick": 0.30, "punch": 0.25, "block": 0.10},
            "📊 Data-Driven Player": {"punch": 0.35, "kick": 0.30, "block": 0.20, "combo_attack": 0.15},
            "🏆 Victory Seeker": {"punch": 0.35, "kick": 0.30, "combo_attack": 0.25, "block": 0.10}
        }
        
        # Load ML model
        try:
            from backend.services.model1 import CrossGamePersonalityClassifier
            self.ml_classifier = CrossGamePersonalityClassifier()
            self.ml_classifier.load_models("hybrid_personality_system.pkl")
            self.ml_model_loaded = True
            print("✅ ML-powered AI initialized with trained models")
        except Exception as e:
            self.ml_model_loaded = False
            print(f"⚠️ ML unavailable, using rule-based AI: {e}")
    
    def _update_player_tracking(self, player_x: float, player_z: float):
        """Track player movement for prediction"""
        current_time = time.time()
        
        # Add current position
        self.last_player_positions.append({'x': player_x, 'z': player_z, 't': current_time})
        
        # ✅ OPTIMIZED: Keep only last 2 positions
        if len(self.last_player_positions) > 2:
            self.last_player_positions.pop(0)
        
        # Calculate velocity if we have enough data
        if len(self.last_player_positions) >= 2:
            p1 = self.last_player_positions[-2]
            p2 = self.last_player_positions[-1]
            dt = max(p2['t'] - p1['t'], 0.01)  # Prevent division by zero
            
            # Calculate raw velocity
            vx = (p2['x'] - p1['x']) / dt
            vz = (p2['z'] - p1['z']) / dt
            
            # [FIX] Clamp velocity to realistic player maximum (approx 5.0 units/sec)
            # This prevents the AI from predicting crazy jumps due to lag/jitter
            max_speed = 6.0
            current_speed = np.sqrt(vx**2 + vz**2)
            if current_speed > max_speed:
                scale = max_speed / current_speed
                vx *= scale
                vz *= scale
                
            self.player_velocity = {'x': vx, 'z': vz}
        
        self.last_update_time = current_time
    
    def _predict_player_position(self, lookahead_seconds: float = 0.3) -> Tuple[float, float]:
        """Predict where player will be in the near future"""
        if len(self.last_player_positions) < 1:
            return 0, 0
        
        current = self.last_player_positions[-1]
        
        # ✅ OPTIMIZED: Reduced lookahead to 0.1s for tighter tracking
        safe_lookahead = 0.1
        
        predicted_x = current['x'] + self.player_velocity['x'] * safe_lookahead
        predicted_z = current['z'] + self.player_velocity['z'] * safe_lookahead
        
        # Limit prediction distance to 1.0 units
        dx = predicted_x - current['x']
        dz = predicted_z - current['z']
        dist = np.sqrt(dx*dx + dz*dz)
        if dist > 1.0:
            scale = 1.0 / dist
            predicted_x = current['x'] + dx * scale
            predicted_z = current['z'] + dz * scale
        
        # Clamp to arena bounds
        predicted_x = np.clip(predicted_x, -6, 6)
        predicted_z = np.clip(predicted_z, -4, 4)
        
        return predicted_x, predicted_z
    
    def _calculate_optimal_position(self, player_x: float, player_z: float, ai_x: float, ai_z: float, distance: float) -> Tuple[float, float]:
        """
        ✅ Calculate optimal fighting position - maintain attack range
        Keeps AI at proper fighting distance (1.8-2.5 units)
        """
        import math
        
        # Optimal fighting distance: 1.8-2.5 units
        OPTIMAL_MIN = 1.8
        OPTIMAL_MAX = 2.5
        
        if OPTIMAL_MIN <= distance <= OPTIMAL_MAX:
            # Perfect distance - hold position
            return ai_x, ai_z
        
        elif distance < OPTIMAL_MIN:
            # Too close - back up slightly
            dx = ai_x - player_x
            dz = ai_z - player_z
            mag = max(0.1, math.sqrt(dx*dx + dz*dz))
            target_x = player_x + (dx / mag) * OPTIMAL_MIN
            target_z = player_z + (dz / mag) * OPTIMAL_MIN
            return target_x, target_z
        
        else:
            # Too far - move closer (60% of the way)
            target_x = ai_x + (player_x - ai_x) * 0.6
            target_z = ai_z + (player_z - ai_z) * 0.6
            return target_x, target_z
    
    def _get_ml_predicted_action(self, state: Dict, distance: float) -> Tuple[str, str, float]:
        """
        🧠 ML ACTION SELECTION - ALWAYS uses trained model
        Returns: (action, archetype, confidence)
        Uses caching for performance but ALWAYS returns ML-based action
        """
        current_time = time.time()
        
        # ✅ Use cache if available (10 second cache to avoid slow ML recomputation)
        if self._cached_archetype and (current_time - self._cache_time) < 10.0:
            archetype = self._cached_archetype['archetype']
            confidence = self._cached_archetype['confidence']
            
            if archetype in self.archetype_strategies:
                strategy = self.archetype_strategies[archetype]
                actions = list(strategy.keys())
                probs = list(strategy.values())
                
                # Distance-based probability adjustments
                if distance < 2.0:
                    # Close range = MORE attacks, less blocking
                    probs = [p * 2.0 if a in ['punch', 'kick', 'combo_attack'] else p * 0.3 for a, p in zip(actions, probs)]
                elif distance < 3.5:
                    # Medium range = balanced attacks
                    probs = [p * 1.5 if a in ['punch', 'kick'] else p for a, p in zip(actions, probs)]
                elif distance > 5.0:
                    # Far range = less combos (out of range)
                    probs = [p * 0.3 if a == 'combo_attack' else p for a, p in zip(actions, probs)]
                
                total_prob = sum(probs)
                probs = [p / total_prob for p in probs]
                action = np.random.choice(actions, p=probs)
                return action, archetype, confidence
        
        # ✅ COMPUTE ML PREDICTION - Model is always loaded at startup
        if self.ml_model_loaded and self.ml_classifier is not None:
            try:
                # Calculate features from game state
                player_health = state.get('player_health', 100)
                ai_health = state.get('ai_health', 100)
                health_advantage = (ai_health - player_health) / 100.0
                aggression_rate = 0.6 + (health_advantage * 0.3)
                
                recent_actions = self.action_history[-10:] if self.action_history else []
                total = len(recent_actions) or 1
                defense_ratio = sum(1 for a in recent_actions if a == 'block') / total
                combo_preference = sum(1 for a in recent_actions if a == 'combo_attack') / total
                reaction_time = max(0.1, min(1.0, distance / 5.0))
                
                fighting_features = [aggression_rate, defense_ratio, combo_preference, reaction_time]
                game_features = {'fighting': fighting_features}
                
                # 🧠 CALL TRAINED ML MODEL
                ml_result = self.ml_classifier.predict_personality(game_features)
                
                archetype = ml_result.get('personality_archetype', '🔥 Aggressive Dominator')
                confidence = ml_result.get('category_confidence', 0.7)
                
                # Cache result for 10 seconds to avoid slow ML recomputation
                self._cached_archetype = {'archetype': archetype, 'confidence': confidence}
                self._cache_time = current_time
                
                # Recurse to use the cached result with distance adjustments
                return self._get_ml_predicted_action(state, distance)
                
            except Exception as e:
                print(f"⚠️ ML prediction error: {e}")
        
        # ✅ If ML not loaded yet, use default archetype (still ML-based strategy)
        default_archetype = '🔥 Aggressive Dominator'
        self._cached_archetype = {'archetype': default_archetype, 'confidence': 0.8}
        self._cache_time = current_time
        
        strategy = self.archetype_strategies[default_archetype]
        actions = list(strategy.keys())
        probs = list(strategy.values())
        
        # Distance adjustments
        if distance < 2.0:
            probs = [p * 2.0 if a in ['punch', 'kick', 'combo_attack'] else p * 0.3 for a, p in zip(actions, probs)]
        
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]
        action = np.random.choice(actions, p=probs)
        
        return action, default_archetype, 0.8

    def get_action(
        self, 
        game_type: GameType, 
        game_state: Dict[str, Any],
        personality: UnifiedPersonality = None
    ) -> Dict[str, Any]:
        """Main entry point - ML-enhanced decision making"""
        
        if isinstance(game_type, str):
            game_type_str = game_type.lower()
        else:
            game_type_str = game_type.value.lower()
        
        if game_type_str == "fighting":
            return self._get_ml_fighting_action(game_state, personality)
        elif game_type_str == "badminton":
            return self._get_badminton_action(game_state, personality)
        elif game_type_str == "racing":
            return self._get_racing_action(game_state, personality)
        else:
            return {"action": "idle", "position": {"x": 0, "y": 0, "z": 0}}
    
    def _get_ml_fighting_action(
        self, 
        state: Dict, 
        personality: UnifiedPersonality = None
    ) -> Dict[str, Any]:
        """
        🧠 ML-POWERED FIGHTING AI - ALWAYS USES TRAINED MODEL
        No fallback logic - all actions come from ML classifier
        """
        try:
            distance = state.get('distance_to_player', 50)
            player_health = state.get('player_health', 100)
            ai_health = state.get('ai_health', 100)
            player_pos = state.get('player_position', {'x': 0, 'z': 0})
            ai_pos = state.get('ai_position', {'x': 4.5, 'z': 0})
            
            # Extract positions
            player_x = player_pos.get('x', 0) if isinstance(player_pos, dict) else player_pos[0]
            player_z = player_pos.get('z', 0) if isinstance(player_pos, dict) else (player_pos[2] if len(player_pos) > 2 else 0)
            
            client_ai_x = ai_pos.get('x', 4.5) if isinstance(ai_pos, dict) else ai_pos[0]
            client_ai_z = ai_pos.get('z', 0) if isinstance(ai_pos, dict) else (ai_pos[2] if len(ai_pos) > 2 else 0)
            
            # Use server position if client is stale (stuck at spawn)
            if abs(client_ai_x - 4.5) < 0.1 and abs(client_ai_z) < 0.1:
                ai_x = self.server_ai_position['x']
                ai_z = self.server_ai_position['z']
            else:
                ai_x = client_ai_x
                ai_z = client_ai_z
                self.server_ai_position['x'] = ai_x
                self.server_ai_position['z'] = ai_z
            
            # Recalculate actual distance
            distance = np.sqrt((player_x - ai_x)**2 + (player_z - ai_z)**2)
            
            # Round reset detection
            if player_health == 100 and ai_health == 100 and distance > 8:
                self.server_ai_position = {'x': 4.5, 'z': 0}
                ai_x, ai_z = 4.5, 0
            
            # ============================================================================
            # 🧠 ML ACTION - ALWAYS from trained model, NO FALLBACK
            # ============================================================================
            ml_action, ml_archetype, ml_confidence = self._get_ml_predicted_action(state, distance)
            
            # ACTION is ALWAYS from ML
            action = ml_action
            
            # ============================================================================
            # 🎯 POSITIONING BASED ON DISTANCE - Movement strategy
            # ============================================================================
            
            if distance < 1.8:
                # VERY CLOSE - Hold position while attacking
                target_x = ai_x
                target_z = ai_z
                
            elif distance < 3.0:
                # CLOSE RANGE - Circle around player while attacking
                dx = player_x - ai_x
                dz = player_z - ai_z
                perp_x = -dz / max(0.1, distance)
                perp_z = dx / max(0.1, distance)
                circle = np.random.uniform(-0.8, 0.8)
                target_x = ai_x + dx * 0.25 + perp_x * circle
                target_z = ai_z + dz * 0.25 + perp_z * circle
                
            elif distance < 5.0:
                # MEDIUM RANGE - Approach aggressively
                target_x = ai_x + (player_x - ai_x) * 0.6
                target_z = ai_z + (player_z - ai_z) * 0.6
                
            else:
                # FAR - Rush toward player
                target_x = ai_x + (player_x - ai_x) * 0.75
                target_z = ai_z + (player_z - ai_z) * 0.75
            
            # Clamp to arena
            target_x = float(np.clip(target_x, -6, 6))
            target_z = float(np.clip(target_z, -4, 4))
            
            # Update server position
            self.server_ai_position['x'] = target_x
            self.server_ai_position['z'] = target_z
            
            # Store action for ML features
            self.action_history.append(action)
            if len(self.action_history) > 30:
                self.action_history.pop(0)
            
            return {
                "action": action,
                "position": {"x": target_x, "y": 0.0, "z": target_z},
                "using_ml": True,
                "ml_archetype": ml_archetype,
                "ml_confidence": float(ml_confidence) if ml_confidence else 0.0,
                "confidence": float(ml_confidence) if ml_confidence else 0.85
            }
            
        except Exception as e:
            print(f"❌ ML AI error: {e}")
            # Even on error, use ML-based action from cache if available
            if self._cached_archetype:
                archetype = self._cached_archetype['archetype']
                if archetype in self.archetype_strategies:
                    strategy = self.archetype_strategies[archetype]
                    action = np.random.choice(list(strategy.keys()), p=list(strategy.values()))
                    return {
                        "action": action,
                        "position": {"x": 3.0, "y": 0.0, "z": 0.0},
                        "using_ml": True,
                        "ml_archetype": archetype,
                        "confidence": 0.7
                    }
            # Last resort - still use archetype strategy
            return {
                "action": "punch",
                "position": {"x": 3.0, "y": 0.0, "z": 0.0},
                "using_ml": True,
                "ml_archetype": "🔥 Aggressive Dominator",
                "confidence": 0.5
            }
    
    def _get_badminton_action(self, state: Dict, personality: UnifiedPersonality = None) -> Dict[str, Any]:
        """Enhanced badminton AI with personality awareness"""
        try:
            rally_count = state.get('rally_count', 1)
            shuttle_y = state.get('shuttlecock_position', {}).get('y', 2.0)
            
            strategic = personality.strategic_thinking if personality else 0.5
            aggression = personality.aggression_level if personality else 0.5
            
            # Aggressive players get more power shots
            if aggression > 0.7:
                if shuttle_y > 2.0:
                    action = "smash"
                elif shuttle_y < 0.8:
                    action = "net_kill"
                else:
                    action = "drive"
            # Strategic players get more placement shots
            elif strategic > 0.7:
                if rally_count > 10:
                    action = "drop_shot"
                else:
                    action = "tactical_placement"
            else:
                # Balanced approach
                if rally_count > 15:
                    action = "smash"
                elif shuttle_y < 0.5:
                    action = "net_shot"
                else:
                    action = "clear"
            
            target = {"x": -5, "z": 0}
            return {"action": action, "target": target, "confidence": 0.80}
        except Exception as e:
            return {"action": "clear", "target": {"x": -5, "z": 0}}
    
    def _get_racing_action(self, state: Dict, personality: UnifiedPersonality = None) -> Dict[str, Any]:
        """Enhanced racing AI with personality awareness"""
        try:
            position = state.get('position', 2)
            player_x = state.get('player_position', {}).get('x', 0.5)
            ai_x = state.get('ai_position', {}).get('x', 0.5)
            
            risk = personality.risk_tolerance if personality else 0.5
            precision = personality.precision_focus if personality else 0.5
            
            # Risky players attempt more overtakes
            if position > 1 and abs(player_x - ai_x) < 0.3 and risk > 0.6:
                return {"action": "overtake", "confidence": 0.85}
            
            # Precision players use perfect racing line
            if precision > 0.7:
                return {"action": "perfect_racing_line", "confidence": 0.90}
            
            # Leading position - defensive
            if position == 1:
                return {"action": "block_overtake", "confidence": 0.85}
            
            return {"action": "maintain_speed", "confidence": 0.70}
        except Exception as e:
            return {"action": "maintain_speed", "confidence": 0.5}


# Keep RuleBasedAIOpponent as alias for backward compatibility
RuleBasedAIOpponent = MLPoweredAIOpponent
