import random
import time
from typing import Dict, Any, List
from enum import Enum
from backend.dbmodels.personality import GameType, UnifiedPersonality

class Difficulty(Enum):
    EASY = 0.3
    MEDIUM = 0.5
    HARD = 0.7
    ADAPTIVE = 0.6  # Starts medium, adapts

class RuleBasedAIOpponent:
    """Rule-based AI opponent system for all three games"""
    
    def __init__(self):
        self.difficulty = Difficulty.ADAPTIVE
        self.player_pattern_memory = {}
        self.recent_player_actions = []
        self.adaptation_rate = 0.1
        
    def get_action(self, game_type: GameType, game_state: Dict[str, Any], 
                   player_personality: UnifiedPersonality = None) -> Dict[str, Any]:
        """Main AI decision method"""
        
        if game_type == GameType.FIGHTING:
            return self._get_fighting_action(game_state, player_personality)
        elif game_type == GameType.BADMINTON:
            return self._get_badminton_action(game_state, player_personality)
        elif game_type == GameType.RACING:
            return self._get_racing_action(game_state, player_personality)
        
        return {"action": "idle", "reasoning": "Unknown game type"}
    
    def _get_fighting_action(self, game_state: Dict[str, Any], 
                           personality: UnifiedPersonality = None) -> Dict[str, Any]:
        """Fighting game AI logic"""
        
        player_health = game_state.get('player_health', 100)
        ai_health = game_state.get('ai_health', 100)
        player_position = game_state.get('player_position', {'x': 0, 'y': 0})
        distance = game_state.get('distance_to_player', 50)
        
        # Personality-based strategy adjustment
        aggression_factor = 0.5
        if personality:
            aggression_factor = personality.aggression_level
        
        # Decision tree based on game state
        if ai_health < 20:
            # Desperate situation - high risk moves
            actions = ['special_attack', 'combo_attack', 'desperate_rush']
            return {
                'action': random.choice(actions),
                'reasoning': 'Low health - desperate attack',
                'confidence': 0.8
            }
        
        elif distance > 80:
            # Far from player
            if aggression_factor > 0.7:
                return {
                    'action': 'rush_forward',
                    'reasoning': 'Aggressive player detected - rushing',
                    'confidence': 0.7
                }
            else:
                return {
                    'action': 'approach_cautiously',
                    'reasoning': 'Maintaining distance',
                    'confidence': 0.6
                }
        
        elif distance < 30:
            # Close combat
            if player_health < ai_health:
                return {
                    'action': 'aggressive_combo',
                    'reasoning': 'Player weakened - pressing advantage',
                    'confidence': 0.9
                }
            else:
                # React to player patterns
                recent_blocks = sum(1 for a in self.recent_player_actions[-5:] 
                                  if a.get('action_type') == 'block')
                
                if recent_blocks > 2:
                    return {
                        'action': 'grab_throw',
                        'reasoning': 'Player blocking too much',
                        'confidence': 0.8
                    }
                else:
                    return {
                        'action': 'quick_jab',
                        'reasoning': 'Standard close combat',
                        'confidence': 0.6
                    }
        
        # Medium range - default actions
        actions = ['forward_kick', 'punch_combo', 'block', 'sidestep']
        weights = [0.3, 0.3, 0.2, 0.2]
        
        # Adjust weights based on difficulty
        if self.difficulty.value > 0.6:
            weights = [0.4, 0.4, 0.1, 0.1]  # More aggressive
        
        return {
            'action': random.choices(actions, weights=weights)[0],
            'reasoning': 'Medium range combat',
            'confidence': 0.5 + self.difficulty.value * 0.3
        }
    
    def _get_badminton_action(self, game_state: Dict[str, Any], 
                            personality: UnifiedPersonality = None) -> Dict[str, Any]:
        """Badminton game AI logic"""
        
        shuttlecock_pos = game_state.get('shuttlecock_position', {'x': 0.5, 'y': 0.5})
        player_pos = game_state.get('player_position', {'x': 0.3, 'y': 0.8})
        ai_pos = game_state.get('ai_position', {'x': 0.7, 'y': 0.2})
        rally_count = game_state.get('rally_count', 1)
        score_diff = game_state.get('score_difference', 0)  # positive = AI winning
        
        # Personality factors
        strategic_level = 0.5
        patience_level = 0.5
        if personality:
            strategic_level = personality.strategic_thinking
            patience_level = personality.patience_level
        
        # Strategic decision making
        if rally_count < 3:
            # Early rally - be safe or aggressive based on personality
            if strategic_level > 0.7:
                return {
                    'action': 'placement_shot',
                    'target': {'x': 0.8 - player_pos['x'], 'y': 0.9},
                    'power': 0.6,
                    'reasoning': 'Strategic placement - testing player',
                    'confidence': 0.8
                }
            else:
                return {
                    'action': 'clear_shot',
                    'target': {'x': 0.5, 'y': 0.9},
                    'power': 0.7,
                    'reasoning': 'Safe clear shot',
                    'confidence': 0.6
                }
        
        elif rally_count > 8:
            # Long rally - try to end it
            if patience_level < 0.4:
                return {
                    'action': 'smash',
                    'target': {'x': player_pos['x'], 'y': 0.8},
                    'power': 0.9,
                    'reasoning': 'Impatient - ending rally with smash',
                    'confidence': 0.7
                }
            else:
                return {
                    'action': 'drop_shot',
                    'target': {'x': 0.2, 'y': 0.7},
                    'power': 0.3,
                    'reasoning': 'Patient rally - strategic drop',
                    'confidence': 0.8
                }
        
        # Mid-rally decisions
        if shuttlecock_pos['y'] < 0.3:  # Shuttlecock at net
            return {
                'action': 'net_shot',
                'target': {'x': 0.1, 'y': 0.8},
                'power': 0.4,
                'reasoning': 'Shuttlecock at net - tight net shot',
                'confidence': 0.7
            }
        
        elif shuttlecock_pos['y'] > 0.8:  # Back court
            if score_diff < -3:  # AI losing badly
                return {
                    'action': 'aggressive_smash',
                    'target': {'x': player_pos['x'], 'y': 0.9},
                    'power': 1.0,
                    'reasoning': 'Desperate situation - all-out attack',
                    'confidence': 0.6
                }
            else:
                return {
                    'action': 'cross_court_drive',
                    'target': {'x': 1.0 - player_pos['x'], 'y': 0.5},
                    'power': 0.8,
                    'reasoning': 'Back court cross-court drive',
                    'confidence': 0.7
                }
        
        # Default mid-court action
        return {
            'action': 'drive_shot',
            'target': {'x': 0.7, 'y': 0.6},
            'power': 0.7,
            'reasoning': 'Standard drive shot',
            'confidence': 0.6
        }
    
    def _get_racing_action(self, game_state: Dict[str, Any], 
                         personality: UnifiedPersonality = None) -> Dict[str, Any]:
        """Racing game AI logic"""
        
        player_position = game_state.get('player_position', {'x': 0.5, 'y': 0.5})
        ai_position = game_state.get('ai_position', {'x': 0.5, 'y': 0.3})
        track_section = game_state.get('track_section', 'straight')
        speed = game_state.get('ai_speed', 60)
        lap_progress = game_state.get('lap_progress', 0.0)
        position_in_race = game_state.get('position', 2)  # 1 = first place
        
        # Personality factors
        risk_tolerance = 0.5
        precision_focus = 0.5
        if personality:
            risk_tolerance = personality.risk_tolerance
            precision_focus = personality.precision_focus
        
        # Track section specific logic
        if track_section == 'corner':
            if precision_focus > 0.7:
                return {
                    'action': 'perfect_racing_line',
                    'speed_adjustment': -10,
                    'steering': 'optimal_apex',
                    'reasoning': 'High precision - perfect cornering',
                    'confidence': 0.9
                }
            elif risk_tolerance > 0.7:
                return {
                    'action': 'late_braking',
                    'speed_adjustment': 5,
                    'steering': 'aggressive_inside',
                    'reasoning': 'Risk-taker - late braking attempt',
                    'confidence': 0.6
                }
            else:
                return {
                    'action': 'safe_cornering',
                    'speed_adjustment': -5,
                    'steering': 'standard_line',
                    'reasoning': 'Safe cornering approach',
                    'confidence': 0.7
                }
        
        elif track_section == 'straight':
            # Check if player is nearby for overtaking opportunity
            distance_to_player = abs(player_position['y'] - ai_position['y'])
            
            if distance_to_player < 0.1 and position_in_race > 1:
                if risk_tolerance > 0.6:
                    return {
                        'action': 'overtake_attempt',
                        'speed_adjustment': 15,
                        'steering': 'move_left' if player_position['x'] > 0.5 else 'move_right',
                        'reasoning': 'Overtaking opportunity detected',
                        'confidence': 0.7
                    }
                else:
                    return {
                        'action': 'follow_slipstream',
                        'speed_adjustment': 5,
                        'steering': 'follow_player',
                        'reasoning': 'Conservative - using slipstream',
                        'confidence': 0.8
                    }
            else:
                return {
                    'action': 'maintain_speed',
                    'speed_adjustment': 0,
                    'steering': 'center_line',
                    'reasoning': 'Straight section - maintaining position',
                    'confidence': 0.6
                }
        
        # Default action
        return {
            'action': 'adaptive_driving',
            'speed_adjustment': random.randint(-5, 5),
            'steering': 'slight_adjustment',
            'reasoning': 'Adaptive driving behavior',
            'confidence': 0.5
        }
    
    def update_player_pattern(self, player_action: Dict[str, Any]):
        """Learn from player actions for better AI responses"""
        self.recent_player_actions.append(player_action)
        
        # Keep only last 10 actions
        if len(self.recent_player_actions) > 10:
            self.recent_player_actions.pop(0)
        
        # Simple pattern detection
        action_type = player_action.get('action_type')
        if action_type in self.player_pattern_memory:
            self.player_pattern_memory[action_type] += 1
        else:
            self.player_pattern_memory[action_type] = 1
    
    def adapt_difficulty(self, player_success_rate: float):
        """Dynamically adjust AI difficulty based on player performance"""
        if self.difficulty == Difficulty.ADAPTIVE:
            target_success_rate = 0.55  # Aim for slight player advantage
            
            if player_success_rate > 0.7:
                # Player winning too much - increase difficulty
                self.difficulty = Difficulty.HARD
            elif player_success_rate < 0.3:
                # Player losing too much - decrease difficulty
                self.difficulty = Difficulty.EASY
            else:
                # Keep medium difficulty
                self.difficulty = Difficulty.MEDIUM
    
    def get_difficulty_info(self) -> Dict[str, Any]:
        """Get current AI difficulty information"""
        return {
            'current_difficulty': self.difficulty.name,
            'difficulty_value': self.difficulty.value,
            'pattern_memory_size': len(self.player_pattern_memory),
            'recent_actions_tracked': len(self.recent_player_actions)
        }