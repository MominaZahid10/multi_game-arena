from typing import Dict, Any, List
from models.personality import GameType, UnifiedPersonality

class CrossGameStrategySelector:
    """
    DAY 2 DELIVERABLE: Multi-Game AI Strategy Selection
    """
    
    def __init__(self):
        self.strategies = {
            GameType.FIGHTING: {
                "aggressive_pressure": {"difficulty": 0.8, "style": "offensive"},
                "defensive_counter": {"difficulty": 0.6, "style": "defensive"},
                "balanced_adaptive": {"difficulty": 0.7, "style": "balanced"},
                "unpredictable_chaos": {"difficulty": 0.9, "style": "chaotic"}
            },
            GameType.BADMINTON: {
                "baseline_rally": {"difficulty": 0.6, "style": "patient"},
                "net_pressure": {"difficulty": 0.8, "style": "aggressive"},
                "power_game": {"difficulty": 0.7, "style": "offensive"},
                "tactical_placement": {"difficulty": 0.9, "style": "strategic"},
                "endurance_test": {"difficulty": 0.5, "style": "defensive"}
            },
            GameType.RACING: {
                "blocking_defense": {"difficulty": 0.7, "style": "defensive"},
                "speed_challenge": {"difficulty": 0.8, "style": "competitive"},
                "precision_test": {"difficulty": 0.9, "style": "technical"},
                "adaptive_racer": {"difficulty": 0.6, "style": "balanced"}
            }
        }
    
    async def select_action(
        self, 
        game_type: GameType,
        game_state: Dict[str, Any],
        unified_personality: UnifiedPersonality,
        cross_game_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Select optimal AI strategy based on cross-game personality analysis"""
        
        # Analyze player personality for strategy selection
        strategy_name = self._select_strategy_by_personality(game_type, unified_personality)
        strategy_config = self.strategies[game_type][strategy_name]
        
        # Generate game-specific action
        action = await self._generate_game_action(game_type, strategy_name, game_state)
        
        # Create cross-game reasoning
        reasoning = self._build_cross_game_reasoning(
            unified_personality, strategy_name, cross_game_history or []
        )
        
        return {
            "action": action,
            "strategy": strategy_name,
            "confidence": min(1.0, unified_personality.confidence_score + 0.2),
            "reasoning": reasoning,
            "insights": self._generate_personality_insights(unified_personality),
            "notes": [f"Adapting to {game_type.value} based on cross-game analysis"]
        }
    
    def _select_strategy_by_personality(self, game_type: GameType, personality: UnifiedPersonality) -> str:
        """Select strategy based on unified personality traits"""
        
        if game_type == GameType.FIGHTING:
            if personality.aggression_level > 0.7:
                return "aggressive_pressure"
            elif personality.patience_level > 0.7:
                return "defensive_counter"
            elif personality.risk_tolerance > 0.8:
                return "unpredictable_chaos"
            else:
                return "balanced_adaptive"
        
        elif game_type == GameType.BADMINTON:
            if personality.strategic_thinking > 0.8:
                return "tactical_placement"
            elif personality.competitive_drive > 0.7:
                return "power_game"
            elif personality.patience_level > 0.7:
                return "endurance_test"
            elif personality.aggression_level > 0.6:
                return "net_pressure"
            else:
                return "baseline_rally"
        
        elif game_type == GameType.RACING:
            if personality.precision_focus > 0.8:
                return "precision_test"
            elif personality.competitive_drive > 0.8:
                return "speed_challenge"
            elif personality.analytical_thinking > 0.7:
                return "adaptive_racer"
            else:
                return "blocking_defense"
        
        return "balanced_adaptive"
    
    async def _generate_game_action(
        self, game_type: GameType, strategy: str, game_state: Dict[str, Any]
    ) -> str:
        """Generate specific action for the game type and strategy"""
        
        if game_type == GameType.FIGHTING:
            actions = {
                "aggressive_pressure": ["combo_attack", "rush_forward", "power_strike"],
                "defensive_counter": ["block", "counter_attack", "retreat"],
                "balanced_adaptive": ["jab", "move_sideways", "feint"],
                "unpredictable_chaos": ["random_combo", "teleport", "special_move"]
            }
        
        elif game_type == GameType.BADMINTON:
            actions = {
                "tactical_placement": ["drop_shot", "cross_court_clear", "net_shot"],
                "power_game": ["smash", "drive", "power_clear"],
                "endurance_test": ["defensive_clear", "high_clear", "lob"],
                "net_pressure": ["net_kill", "net_drop", "push"],
                "baseline_rally": ["clear", "drop", "drive"]
            }
        
        elif game_type == GameType.RACING:
            actions = {
                "precision_test": ["perfect_racing_line", "late_apex", "smooth_acceleration"],
                "speed_challenge": ["overtake", "slipstream", "full_throttle"],
                "adaptive_racer": ["mirror_player", "defensive_position", "strategic_pace"],
                "blocking_defense": ["block_overtake", "defensive_line", "brake_check"]
            }
        
        else:
            actions = {"default": ["basic_action"]}
        
        import random
        return random.choice(actions.get(strategy, ["default_action"]))
    
    def _build_cross_game_reasoning(
        self, personality: UnifiedPersonality, strategy: str, history: List[Dict]
    ) -> str:
        """Build reasoning explanation using cross-game insights"""
        
        reasoning_parts = []
        
        # Personality-based reasoning
        if personality.aggression_level > 0.7:
            reasoning_parts.append("High aggression detected across games")
        if personality.risk_tolerance > 0.7:
            reasoning_parts.append("Risk-taking behavior observed in previous games")
        if personality.strategic_thinking > 0.8:
            reasoning_parts.append("Strong strategic patterns identified")
        if personality.precision_focus > 0.8:
            reasoning_parts.append("High precision focus from cross-game analysis")
        
        # Strategy selection reasoning
        reasoning_parts.append(f"Selected '{strategy}' strategy based on unified personality profile")
        
        # Cross-game transfer insights
        games_played = personality.games_played
        if len(games_played) > 1:
            reasoning_parts.append(f"Leveraging insights from {', '.join(games_played)} gameplay")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_personality_insights(self, personality: UnifiedPersonality) -> Dict[str, str]:
        """Generate personality insights for the response"""
        
        insights = {}
        
        if personality.aggression_level > 0.7:
            insights["aggression"] = "Highly aggressive playstyle detected"
        elif personality.aggression_level < 0.3:
            insights["aggression"] = "Passive, defensive approach preferred"
        
        if personality.risk_tolerance > 0.7:
            insights["risk"] = "High risk-taker, enjoys challenging moves"
        elif personality.risk_tolerance < 0.3:
            insights["risk"] = "Risk-averse, prefers safe strategies"
        
        if personality.strategic_thinking > 0.8:
            insights["strategy"] = "Excellent strategic planning abilities"
        elif personality.strategic_thinking < 0.4:
            insights["strategy"] = "More reactive than strategic"
        
        if personality.precision_focus > 0.8:
            insights["precision"] = "High attention to detail and accuracy"
        
        return insights
