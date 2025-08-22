from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, JSON
from sqlalchemy.sql import func
from databaseconn import Base
import uuid
from datetime import datetime

class GameSession(Base):
    """
    Main game session tracking across all three games
    """
    __tablename__ = "game_sessions"
    
    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, unique=True, index=True, nullable=False)
    player_id = Column(String, index=True, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Session Data
    current_game = Column(String, default="fighting")  # fighting, badminton, racing
    games_played = Column(JSON, default=list)  # List of games played this session
    total_actions = Column(Integer, default=0)
    session_duration = Column(Float, default=0.0)  # in seconds
    voice_commands_used = Column(Integer, default=0)
    
    # Cross-game statistics
    fighting_actions = Column(Integer, default=0)
    badminton_actions = Column(Integer, default=0)
    racing_actions = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())

class PlayerAction(Base):
    """
    Individual player actions across all games
    """
    __tablename__ = "player_actions"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    action_sequence = Column(Integer, default=0)  # Order within session
    
    # Game context
    game_type = Column(String, index=True, nullable=False)  # fighting, badminton, racing
    action_type = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)
    success = Column(Boolean, default=False)
    
    # Action data (flexible JSON storage)
    action_data = Column(JSON, default=dict)  # Game-specific action details
    context = Column(JSON, default=dict)      # Environmental context
    
    # Performance metrics
    reaction_time = Column(Float, nullable=True)
    difficulty_level = Column(Float, default=0.5)
    effectiveness_score = Column(Float, nullable=True)  # 0.0 - 1.0
    
    # Fighting-specific fields
    move_type = Column(String, nullable=True)      # attack, block, move
    damage_dealt = Column(Float, nullable=True)
    combo_count = Column(Integer, default=0)
    position_x = Column(Float, nullable=True)
    position_y = Column(Float, nullable=True)
    
    # Badminton-specific fields
    shot_type = Column(String, nullable=True)      # clear, drop, smash, net, drive
    power_level = Column(Float, nullable=True)     # 0.0 - 1.0
    rally_position = Column(Integer, nullable=True)
    court_pos_x = Column(Float, nullable=True)
    court_pos_y = Column(Float, nullable=True)
    target_x = Column(Float, nullable=True)
    target_y = Column(Float, nullable=True)
    
    # Racing-specific fields
    speed = Column(Float, nullable=True)
    track_pos_x = Column(Float, nullable=True)
    track_pos_y = Column(Float, nullable=True)
    overtaking_attempt = Column(Boolean, default=False)
    crash_occurred = Column(Boolean, default=False)
    racing_line_deviation = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PersonalityProfile(Base):
    """
    Unified personality profile across all games
    """
    __tablename__ = "personality_profiles"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    
    # Core Personality Traits (0.0 - 1.0)
    aggression_level = Column(Float, default=0.5)
    risk_tolerance = Column(Float, default=0.5)
    analytical_thinking = Column(Float, default=0.5)
    patience_level = Column(Float, default=0.5)
    precision_focus = Column(Float, default=0.5)
    competitive_drive = Column(Float, default=0.5)
    strategic_thinking = Column(Float, default=0.5)
    adaptability = Column(Float, default=0.5)
    
    # Confidence and validation metrics
    trait_confidence = Column(JSON, default=dict)  # Per-trait confidence scores
    overall_confidence = Column(Float, default=0.0)
    consistency_score = Column(Float, default=0.0)  # Cross-game consistency
    
    # Game-specific personality manifestations
    fighting_profile = Column(JSON, default=dict)   # Fighting-specific traits
    badminton_profile = Column(JSON, default=dict)  # Badminton-specific traits  
    racing_profile = Column(JSON, default=dict)     # Racing-specific traits
    
    # Cross-game analysis data
    cross_game_correlations = Column(JSON, default=dict)
    transfer_learning_insights = Column(JSON, default=dict)
    behavioral_patterns = Column(JSON, default=dict)
    
    # Analysis metadata
    total_actions_analyzed = Column(Integer, default=0)
    fighting_actions_analyzed = Column(Integer, default=0)
    badminton_actions_analyzed = Column(Integer, default=0)
    racing_actions_analyzed = Column(Integer, default=0)
    
    # Update tracking
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    analysis_version = Column(String, default="1.0")

class AIStrategy(Base):
    """
    AI strategy selection and performance tracking
    """
    __tablename__ = "ai_strategies"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    game_type = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Strategy information
    strategy_name = Column(String, nullable=False)
    strategy_config = Column(JSON, default=dict)
    difficulty_level = Column(Float, default=0.5)
    
    # Decision reasoning
    personality_factors = Column(JSON, default=dict)  # Which traits influenced decision
    cross_game_reasoning = Column(Text)
    confidence_score = Column(Float, default=0.5)
    
    # Performance tracking
    player_response_time = Column(Float, nullable=True)
    strategy_effectiveness = Column(Float, nullable=True)  # 0.0 - 1.0
    player_adaptation_detected = Column(Boolean, default=False)
    
    # AI action details
    ai_action = Column(String, nullable=True)
    ai_action_success = Column(Boolean, nullable=True)
    player_counter_action = Column(String, nullable=True)
    
    # Learning and adaptation
    strategy_adjustment = Column(JSON, default=dict)
    next_strategy_hint = Column(String, nullable=True)

class CrossGameAnalytics(Base):
    """
    Advanced analytics and insights across all games
    """
    __tablename__ = "cross_game_analytics"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Cross-game behavioral insights
    personality_consistency = Column(Float, default=0.0)  # 0.0 - 1.0
    skill_transfer_score = Column(Float, default=0.0)     # How well skills transfer
    adaptation_speed = Column(Float, default=0.0)         # How quickly player adapts
    
    # Game-specific performance
    fighting_performance = Column(JSON, default=dict)
    badminton_performance = Column(JSON, default=dict)  
    racing_performance = Column(JSON, default=dict)
    
    # Behavioral patterns
    dominant_traits = Column(JSON, default=list)          # Top personality traits
    behavioral_clusters = Column(JSON, default=dict)      # Player type classification
    learning_patterns = Column(JSON, default=dict)        # How player improves
    
    # Predictive insights
    next_game_prediction = Column(String, nullable=True)  # Which game player will choose next
    difficulty_recommendations = Column(JSON, default=dict)
    coaching_suggestions = Column(JSON, default=dict)
    
    # Session summary
    total_playtime = Column(Float, default=0.0)
    games_mastered = Column(JSON, default=list)
    improvement_areas = Column(JSON, default=list)
    
    # Meta analysis
    analysis_quality_score = Column(Float, default=0.0)   # Confidence in analysis
    data_completeness = Column(Float, default=0.0)        # How complete is the data

class VoiceCommand(Base):
    """
    Voice command tracking and analysis (for Day 3+)
    """
    __tablename__ = "voice_commands"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Voice processing
    audio_duration = Column(Float, nullable=True)
    transcribed_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Game context
    game_type = Column(String, nullable=False)
    game_state_context = Column(JSON, default=dict)
    
    # Command processing
    detected_intent = Column(String, nullable=True)
    mapped_action = Column(String, nullable=True)
    command_success = Column(Boolean, default=False)
    
    # Performance
    processing_time_ms = Column(Integer, nullable=True)
    action_execution_time_ms = Column(Integer, nullable=True)

# ================================
# DATABASE INITIALIZATION AND UTILITIES
# ================================

def create_all_tables():
    """Create all database tables"""
    from databaseconn import engine
    Base.metadata.create_all(bind=engine)
    print("✓ All database tables created successfully")

def get_session_stats(db, session_id: str) -> dict:
    """Get comprehensive session statistics"""
    session = db.query(GameSession).filter(GameSession.session_id == session_id).first()
    if not session:
        return {"error": "Session not found"}
    
    # Get action counts by game
    action_counts = db.query(PlayerAction).filter(
        PlayerAction.session_id == session_id
    ).count()
    
    # Get personality profile
    personality = db.query(PersonalityProfile).filter(
        PersonalityProfile.session_id == session_id
    ).first()
    
    # Get latest analytics
    analytics = db.query(CrossGameAnalytics).filter(
        CrossGameAnalytics.session_id == session_id
    ).order_by(CrossGameAnalytics.analysis_timestamp.desc()).first()
    
    return {
        "session_info": {
            "session_id": session.session_id,
            "games_played": session.games_played,
            "total_actions": session.total_actions,
            "session_duration": session.session_duration,
            "is_active": session.is_active
        },
        "personality_summary": {
            "aggression": personality.aggression_level if personality else 0.5,
            "risk_tolerance": personality.risk_tolerance if personality else 0.5,
            "strategic_thinking": personality.strategic_thinking if personality else 0.5,
            "confidence": personality.overall_confidence if personality else 0.0
        } if personality else None,
        "analytics_summary": {
            "consistency": analytics.personality_consistency if analytics else 0.0,
            "adaptation_speed": analytics.adaptation_speed if analytics else 0.0,
            "dominant_traits": analytics.dominant_traits if analytics else []
        } if analytics else None,
        "action_count": action_counts
    }

if __name__ == "__main__":
    # Create database tables
    create_all_tables()