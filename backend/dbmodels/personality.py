from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
from enum import Enum
from datetime import datetime

class GameType(str, Enum):
    FIGHTING = "fighting"
    BADMINTON = "badminton"
    RACING = "racing"

class UniversalAction(BaseModel):
    game_type: GameType
    action_type: str
    timestamp: float
    success: bool
    context: Dict[str, Any] = {}

class FightingAction(UniversalAction):
    move_type: str = Field(..., description="attack, block, move")
    position: tuple[float, float] = (0.0, 0.0)
    damage_dealt: Optional[float] = None
    combo_count: int = 0

class BadmintonAction(UniversalAction):
    shot_type: str = Field(..., description="clear, drop, smash, net, drive")
    court_position: tuple[float, float] = (0.0, 0.0)
    shuttlecock_target: tuple[float, float] = (0.0, 0.0)
    power_level: float = Field(0.5, ge=0.0, le=1.0)
    rally_position: int = 1

class RacingAction(UniversalAction):
    speed: float = 0.0
    position_on_track: tuple[float, float] = (0.0, 0.0)
    overtaking_attempt: bool = False
    crash_occurred: bool = False

class UnifiedPersonality(BaseModel):
    aggression_level: float = Field(0.5, ge=0.0, le=1.0)
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0)
    analytical_thinking: float = Field(0.5, ge=0.0, le=1.0)
    patience_level: float = Field(0.5, ge=0.0, le=1.0)
    precision_focus: float = Field(0.5, ge=0.0, le=1.0)
    competitive_drive: float = Field(0.5, ge=0.0, le=1.0)
    strategic_thinking: float = Field(0.5, ge=0.0, le=1.0)
    adaptability: float = Field(0.5, ge=0.0, le=1.0)
    
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    total_actions_analyzed: int = 0
    games_played: List[str] = []
    last_updated: Optional[datetime] = None
    
    personality_archetype: Optional[str] = "🎮 Multi-Game Player"
    playstyle_category: Optional[str] = "🎯 Adaptive Gamer" 
    category_confidence: Optional[float] = 0.0

class BehavioralFeature(BaseModel):
    name: str
    value: float
    confidence: float
    game_source: str
    description: Optional[str] = None

class UniversalAnalysisRequest(BaseModel):
    session_id: str
    fighting_actions: List[FightingAction] = []
    badminton_actions: List[BadmintonAction] = []
    racing_actions: List[RacingAction] = []

class GameSpecificAnalysisRequest(BaseModel):
    session_id: str
    actions: List[UniversalAction]

class CrossGameAIRequest(BaseModel):
    session_id: str
    current_game: GameType
    game_state: Dict[str, Union[str, float, bool]]
    cross_game_history: Optional[List[Dict]] = []

class CrossGameAIResponse(BaseModel):
    current_game_action: str
    confidence: float
    strategy: str
    cross_game_reasoning: str
    personality_insights: Dict[str, str] = {}
    adaptation_notes: List[str] = []

class MultiGameState(BaseModel):
    current_game: GameType
    fighting_state: Optional[Dict] = None
    badminton_state: Optional[Dict] = None
    racing_state: Optional[Dict] = None
    unified_personality: UnifiedPersonality
    session_duration: float = 0.0
    games_played: List[GameType] = []