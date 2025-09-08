import sys
sys.stdout.reconfigure(encoding="utf-8")
from fastapi import FastAPI,HTTPException,Depends,WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import asyncio
from typing import List,Dict,Optional,Union
import json
from datetime import datetime
from backend.services.rulebased_ai import RuleBasedAIOpponent
import random
import time
from fastapi import Request
import pprint
from pydantic import ValidationError

from backend.config import settings
from backend.databaseconn import get_db,engine,Base
from backend.dbmodels.personality import(
    UniversalAnalysisRequest, UnifiedPersonality, 
    GameSpecificAnalysisRequest, CrossGameAIRequest, CrossGameAIResponse,
    MultiGameState, GameType
)
from backend.dbmodels.games import GameSession, PlayerAction, PersonalityProfile,VoiceCommand



from backend.services.personality_analyzer import MultiGameAnalyzer
from backend.services.crossgame_strategy import CrossGameStrategySelector

Base.metadata.create_all(bind=engine)

app=FastAPI(
    title="AI multi_game arena",
    description="Professional multi_game api",
    version="2.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

multi_game_analyzer=MultiGameAnalyzer()
strategy_selector=CrossGameStrategySelector()

# Add the missing get_session_stats function
def get_session_stats(db: Session, session_id: str) -> dict:
    """Get comprehensive session statistics"""
    
    # Get session info
    session = db.query(GameSession).filter(
        GameSession.session_id == session_id
    ).first()
    
    if not session:
        return {
            "session_info": {
                "session_id": session_id,
                "games_played": [],
                "total_actions": 0,
                "current_game": "unknown"
            },
            "game_breakdown": {},
            "error": "Session not found"
        }
    
    # Get all actions for this session
    all_actions = db.query(PlayerAction).filter(
        PlayerAction.session_id == session_id
    ).all()
    
    # Calculate game-specific stats
    game_breakdown = {}
    for game_type in ["fighting", "badminton", "racing"]:
        game_actions = [a for a in all_actions if a.game_type == game_type]
        if game_actions:
            success_count = sum(1 for a in game_actions if a.success)
            game_breakdown[game_type] = {
                "total_actions": len(game_actions),
                "success_rate": success_count / len(game_actions) if game_actions else 0,
                "last_played": max(a.created_at for a in game_actions).isoformat() if game_actions else None
            }
    
    return {
        "session_info": {
            "session_id": session.session_id,
            "games_played": session.games_played or [],
            "total_actions": session.total_actions,
            "current_game": session.current_game,
            "created_at": session.created_at.isoformat() if hasattr(session, 'created_at') else None
        },
        "game_breakdown": game_breakdown,
        "overall_stats": {
            "total_actions": len(all_actions),
            "success_rate": sum(1 for a in all_actions if a.success) / len(all_actions) if all_actions else 0,
            "games_played_count": len(set(a.game_type for a in all_actions))
        }
    }

@app.get("/")
async def root():
    """API Health Check"""
    return{
        "message":"AI Multi-Game Arena API is running",
        "version":"2.0.0",
        "games_supported":["fighting","badminton","racing"],
        "features": ["cross_game_analysis", "voice_commands", "unified_personality"]

    }

@app.post("/api/v1/player/analyze-universal")
async def analyze_universal_player(
    raw_request: Request,
    db: Session = Depends(get_db)
):
    """
    🔥 FIXED: Universal Player Analysis Endpoint
    Handles validation errors and ML pipeline issues
    """
    try:
        # Step 1: Log the raw request body
        body = await raw_request.json()
        print("\n=== 📨 Incoming Raw Body ===")
        print(f"Request keys: {list(body.keys())}")
        
        # Log action counts for debugging
        if 'fighting_actions' in body:
            print(f"Fighting actions: {len(body['fighting_actions'])}")
        if 'badminton_actions' in body:
            print(f"Badminton actions: {len(body['badminton_actions'])}")
            if body['badminton_actions']:
                first_action = body['badminton_actions'][0]
                print(f"First badminton context: {first_action.get('context', {})}")
        if 'racing_actions' in body:
            print(f"Racing actions: {len(body['racing_actions'])}")

        try:
            request = UniversalAnalysisRequest(**body)
            print("✅ Pydantic validation successful")
        except ValidationError as ve:
            print(f"❌ Pydantic validation failed:")
            for error in ve.errors():
                print(f"  - Field: {error['loc']}")
                print(f"  - Error: {error['msg']}")
                print(f"  - Input: {error['input']}")
            return {
                "error": "Validation failed",
                "details": ve.errors(),
                "suggestion": "Check that context fields match expected types (str, float, bool, or Any)"
            }

        # Get or create game session
        session = db.query(GameSession).filter(
            GameSession.session_id == request.session_id
        ).first()
        
        if not session:
            print(f"🆕 Creating new session: {request.session_id}")
            session = GameSession(
                session_id=request.session_id,
                current_game="fighting"
            )
            db.add(session)
        else:
            print(f"📂 Found existing session: {request.session_id}")
        
        # Process all actions from all games
        all_actions = []
        games_played = set()
        
        for fighting_action in request.fighting_actions:
            all_actions.append(fighting_action)
            games_played.add("fighting")
        for badminton_action in request.badminton_actions:
            all_actions.append(badminton_action)
            games_played.add("badminton")
        for racing_action in request.racing_actions:
            all_actions.append(racing_action)
            games_played.add("racing")
        
        print(f"📊 Total actions to analyze: {len(all_actions)} from {len(games_played)} games")
        
        # Store actions in database with better error handling
        for action in all_actions:
            try:
                db_action = PlayerAction(
                    session_id=request.session_id,
                    game_type=action.game_type.value,
                    action_type=action.action_type,
                    timestamp=action.timestamp,
                    success=action.success,
                    action_data=action.dict(),
                    context=action.context
                )
                db.add(db_action)
            except Exception as action_error:
                print(f"❌ Failed to store action: {action_error}")
                continue
        
        # ------- FIX: Safely handle None for session.total_actions -------
        if session.total_actions is None:
            session.total_actions = 0
        session.total_actions += len(all_actions)
        session.games_played = list(games_played)
        # ---------------------------------------------------------------

        print(f"🧠 Starting cross-game personality analysis...")
        
        try:
            unified_personality = await multi_game_analyzer.analyze_universal_behavior({
                "fighting": request.fighting_actions,
                "badminton": request.badminton_actions,
                "racing": request.racing_actions
            })
            print(f"✅ Analysis complete: {unified_personality.personality_archetype}")
        except Exception as analysis_error:
            print(f"❌ Analysis failed: {analysis_error}")
            import traceback
            traceback.print_exc()
            return {
                "error": "Analysis failed",
                "details": str(analysis_error),
                "fallback": "Using default personality profile"
            }
        
        # Get or update personality profile in database
        personality_db = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == request.session_id
        ).first()
        
        if not personality_db:
            print(f"🆕 Creating new personality profile")
            personality_db = PersonalityProfile(session_id=request.session_id)
            db.add(personality_db)
        else:
            print(f"📝 Updating existing personality profile")

        personality_db.aggression_level = getattr(unified_personality, 'aggression_level', 0.5)
        personality_db.risk_tolerance = getattr(unified_personality, 'risk_tolerance', 0.5)
        personality_db.analytical_thinking = getattr(unified_personality, 'analytical_thinking', 0.5)
        personality_db.patience_level = getattr(unified_personality, 'patience_level', 0.5)
        personality_db.precision_focus = getattr(unified_personality, 'precision_focus', 0.5)
        personality_db.competitive_drive = getattr(unified_personality, 'competitive_drive', 0.5)
        personality_db.strategic_thinking = getattr(unified_personality, 'strategic_thinking', 0.5)

        # ------- FIX: Safely handle None for personality_db.total_actions_analyzed -------
        if personality_db.total_actions_analyzed is None:
            personality_db.total_actions_analyzed = 0
        personality_db.total_actions_analyzed += len(all_actions)
        # -------------------------------------------------------------------------------

        try:
            db.commit()
            print(f"💾 Database updated successfully")
        except Exception as db_error:
            print(f"❌ Database commit failed: {db_error}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        
        try:
            response_data = {
                "unified_personality": unified_personality.dict(),
                "cross_game_insights": multi_game_analyzer.get_cross_game_insights(),
                "session_stats": {
                    "total_actions": session.total_actions,
                    "games_played": session.games_played,
                    "session_id": session.session_id
                },
                "analysis_metadata": {
                    "ml_system_active": multi_game_analyzer.hybrid_system is not None and multi_game_analyzer.hybrid_system.is_trained,
                    "actions_analyzed": len(all_actions),
                    "games_included": list(games_played)
                }
            }
            print(f"✅ Returning successful analysis response")
            return response_data
        except Exception as response_error:
            print(f"❌ Response building failed: {response_error}")
            raise HTTPException(status_code=500, detail=f"Response error: {str(response_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in analyze-universal: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/player/analyze-fighting")
async def analyze_fighting_behavior(
    request: GameSpecificAnalysisRequest,
    db: Session = Depends(get_db)
):
    """Game-specific fighting analysis"""
    try:
        features = await multi_game_analyzer.fighting_analyzer.extract_features(request.actions)
        
        return {
            "game_type": "fighting",
            "behavioral_features": [f.dict() for f in features],
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/player/analyze-badminton")
async def analyze_badminton_behavior(
    request: GameSpecificAnalysisRequest,
    db: Session = Depends(get_db)
):
    """Game-specific badminton analysis"""
    try:
        features = await multi_game_analyzer.badminton_analyzer.extract_features(request.actions)
        
        return {
            "game_type": "badminton",
            "behavioral_features": [f.dict() for f in features],
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/player/analyze-racing")
async def analyze_racing_behavior(
    request: GameSpecificAnalysisRequest,
    db: Session = Depends(get_db)
):
    """Game-specific racing analysis"""
    try:
        features = await multi_game_analyzer.racing_analyzer.extract_features(request.actions)
        
        return {
            "game_type": "racing",
            "behavioral_features": [f.dict() for f in features],
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/v1/ai/get-action/{game_type}")
async def get_cross_game_ai_action(
    game_type: GameType,
    request: CrossGameAIRequest,
    db: Session = Depends(get_db)
):
    """
    DAY 2 DELIVERABLE: Cross-Game AI Strategy Selection
    Get AI action informed by cross-game personality analysis
    """
    try:
        # Get unified personality from database
        personality_db = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == request.session_id
        ).first()
        
        if not personality_db:
            raise HTTPException(status_code=404, detail="No personality profile found")
        
        # Convert to unified personality format
        unified_personality = UnifiedPersonality(
            aggression_level=personality_db.aggression_level,
            risk_tolerance=personality_db.risk_tolerance,
            analytical_thinking=personality_db.analytical_thinking,
            patience_level=personality_db.patience_level,
            precision_focus=personality_db.precision_focus,
            competitive_drive=personality_db.competitive_drive,
            strategic_thinking=personality_db.strategic_thinking
        )
        
        # Select optimal action using cross-game strategy
        optimal_action = await strategy_selector.select_action(
            game_type=game_type,
            game_state=request.game_state,
            unified_personality=unified_personality,
            cross_game_history=request.cross_game_history or []
        )
        
        return CrossGameAIResponse(
            current_game_action=optimal_action["action"],
            confidence=optimal_action["confidence"],
            strategy=optimal_action["strategy"],
            cross_game_reasoning=optimal_action["reasoning"],
            personality_insights=optimal_action.get("insights", {}),
            adaptation_notes=optimal_action.get("notes", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI action failed: {str(e)}")

# Remove lines 20-27 (the placeholder class)

@app.post("/api/v1/voice/process/{session_id}")
async def process_voice_command(
    session_id: str,
    audio_data: dict,
    current_game: str,
    db: Session = Depends(get_db)
):
    """Process voice commands (placeholder for future implementation)"""
    
    # Use the proper SQLAlchemy model
    voice_record = VoiceCommand(
        session_id=session_id,
        game_type=current_game,
        transcribed_text=audio_data.get('transcribed_text', ''),
        confidence_score=0.8,
        detected_intent='placeholder',
        command_success=True
    )
    
    db.add(voice_record)
    db.commit()
    
    return {
        "success": True,
        "message": "Voice processing will be implemented in Phase 2",
        "current_game": current_game,
        "mock_response": {
            "action": "voice_acknowledged",
            "game_response": f"Voice command received for {current_game} game"
        }
    }

@app.post("/api/v1/games/fighting/action")
async def process_fighting_action(
    session_id: str,
    action_data: dict,
    db: Session = Depends(get_db)
):
    """Process fighting game action and get AI response"""
    try:
        ai_opponent = RuleBasedAIOpponent()
        
        # Get current game state
        game_state = {
            'player_health': action_data.get('player_health', 100),
            'ai_health': action_data.get('ai_health', 100),
            'distance_to_player': action_data.get('distance', 50),
            'player_position': action_data.get('player_position', {'x': 0, 'y': 0})
        }
        
        # Get personality for context
        personality_db = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == session_id
        ).first()
        
        personality = None
        if personality_db:
            personality = UnifiedPersonality(
                aggression_level=personality_db.aggression_level,
                risk_tolerance=personality_db.risk_tolerance,
                analytical_thinking=personality_db.analytical_thinking,
                patience_level=personality_db.patience_level,
                precision_focus=personality_db.precision_focus,
                competitive_drive=personality_db.competitive_drive,
                strategic_thinking=personality_db.strategic_thinking
            )
        
        # Get AI action
        ai_response = ai_opponent.get_action(GameType.FIGHTING, game_state, personality)
        
        # Store player action
        db_action = PlayerAction(
            session_id=session_id,
            game_type="fighting",
            action_type=action_data.get('move_type', 'unknown'),
            timestamp=time.time(),
            success=action_data.get('success', False),
            move_type=action_data.get('move_type'),
            damage_dealt=action_data.get('damage', 0),
            combo_count=action_data.get('combo', 0),
            position_x=action_data.get('player_position', {}).get('x', 0),
            position_y=action_data.get('player_position', {}).get('y', 0)
        )
        db.add(db_action)
        db.commit()
        
        return {
            "ai_action": ai_response,
            "game_state": game_state,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/games/badminton/action")
async def process_badminton_action(
    session_id: str,
    action_data: dict,
    db: Session = Depends(get_db)
):
    """Process badminton game action and get AI response"""
    try:
        ai_opponent = RuleBasedAIOpponent()
        
        # Get current game state
        game_state = {
            'shuttlecock_position': action_data.get('shuttlecock_position', {'x': 0.5, 'y': 0.5}),
            'player_position': action_data.get('player_position', {'x': 0.3, 'y': 0.8}),
            'ai_position': action_data.get('ai_position', {'x': 0.7, 'y': 0.2}),
            'rally_count': action_data.get('rally_count', 1),
            'score_difference': action_data.get('score_difference', 0)
        }
        
        # Get personality for context
        personality_db = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == session_id
        ).first()
        
        personality = None
        if personality_db:
            personality = UnifiedPersonality(
                aggression_level=personality_db.aggression_level,
                risk_tolerance=personality_db.risk_tolerance,
                analytical_thinking=personality_db.analytical_thinking,
                patience_level=personality_db.patience_level,
                precision_focus=personality_db.precision_focus,
                competitive_drive=personality_db.competitive_drive,
                strategic_thinking=personality_db.strategic_thinking
            )
        
        # Get AI action
        ai_response = ai_opponent.get_action(GameType.BADMINTON, game_state, personality)
        
        # Store player action
        db_action = PlayerAction(
            session_id=session_id,
            game_type="badminton",
            action_type=action_data.get('shot_type', 'unknown'),
            timestamp=time.time(),
            success=action_data.get('success', False),
            shot_type=action_data.get('shot_type'),
            power_level=action_data.get('power_level', 0.5),
            rally_position=action_data.get('rally_count', 1),
            court_pos_x=action_data.get('player_position', {}).get('x', 0),
            court_pos_y=action_data.get('player_position', {}).get('y', 0),
            target_x=action_data.get('target', {}).get('x', 0),
            target_y=action_data.get('target', {}).get('y', 0)
        )
        db.add(db_action)
        db.commit()
        
        return {
            "ai_action": ai_response,
            "game_state": game_state,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/games/racing/action")
async def process_racing_action(
    session_id: str,
    action_data: dict,
    db: Session = Depends(get_db)
):
    """Process racing game action and get AI response"""
    try:
        ai_opponent = RuleBasedAIOpponent()
        
        # Get current game state
        game_state = {
            'player_position': action_data.get('player_position', {'x': 0.5, 'y': 0.5}),
            'ai_position': action_data.get('ai_position', {'x': 0.5, 'y': 0.3}),
            'track_section': action_data.get('track_section', 'straight'),
            'ai_speed': action_data.get('ai_speed', 60),
            'lap_progress': action_data.get('lap_progress', 0.0),
            'position': action_data.get('position', 2)
        }
        
        # Get personality for context
        personality_db = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == session_id
        ).first()
        
        personality = None
        if personality_db:
            personality = UnifiedPersonality(
                aggression_level=personality_db.aggression_level,
                risk_tolerance=personality_db.risk_tolerance,
                analytical_thinking=personality_db.analytical_thinking,
                patience_level=personality_db.patience_level,
                precision_focus=personality_db.precision_focus,
                competitive_drive=personality_db.competitive_drive,
                strategic_thinking=personality_db.strategic_thinking
            )
        
        # Get AI action
        ai_response = ai_opponent.get_action(GameType.RACING, game_state, personality)
        
        # Store player action
        db_action = PlayerAction(
            session_id=session_id,
            game_type="racing",
            action_type=action_data.get('action_type', 'unknown'),
            timestamp=time.time(),
            success=action_data.get('success', False),
            speed=action_data.get('speed', 0),
            track_pos_x=action_data.get('player_position', {}).get('x', 0),
            track_pos_y=action_data.get('player_position', {}).get('y', 0),
            overtaking_attempt=action_data.get('overtaking_attempt', False),
            crash_occurred=action_data.get('crash_occurred', False),
            racing_line_deviation=action_data.get('racing_line_deviation', 0.0)
        )
        db.add(db_action)
        db.commit()
        
        return {
            "ai_action": ai_response,
            "game_state": game_state,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Real-time personality update endpoint
@app.get("/api/v1/personality/{session_id}")
async def get_personality_profile(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get current personality profile for display"""
    try:
        personality_db = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == session_id
        ).first()
        
        if not personality_db:
            return {
                "personality": None,
                "message": "No personality profile found"
            }
        
        # Get impressive display data
        display_data = multi_game_analyzer.get_personality_display_data(
            UnifiedPersonality(
                aggression_level=personality_db.aggression_level,
                risk_tolerance=personality_db.risk_tolerance,
                analytical_thinking=personality_db.analytical_thinking,
                patience_level=personality_db.patience_level,
                precision_focus=personality_db.precision_focus,
                competitive_drive=personality_db.competitive_drive,
                strategic_thinking=personality_db.strategic_thinking,
                confidence_score=personality_db.overall_confidence,
                total_actions_analyzed=personality_db.total_actions_analyzed,
                games_played=personality_db.fighting_actions_analyzed and ["fighting"] or [] + 
                           personality_db.badminton_actions_analyzed and ["badminton"] or [] +
                           personality_db.racing_actions_analyzed and ["racing"] or []
            )
        )
        
        return {
            "personality": display_data,
            "raw_scores": {
                "aggression_level": personality_db.aggression_level,
                "risk_tolerance": personality_db.risk_tolerance,
                "analytical_thinking": personality_db.analytical_thinking,
                "patience_level": personality_db.patience_level,
                "precision_focus": personality_db.precision_focus,
                "competitive_drive": personality_db.competitive_drive,
                "strategic_thinking": personality_db.strategic_thinking
            },
            "meta": {
                "confidence": personality_db.overall_confidence,
                "total_actions": personality_db.total_actions_analyzed,
                "last_updated": personality_db.last_updated.isoformat() if personality_db.last_updated else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/{session_id}")
async def get_session_analytics(session_id: str, db: Session = Depends(get_db)):
    """Get comprehensive session analytics"""
    
    # Fixed: Now using the defined get_session_stats function
    stats = get_session_stats(db, session_id)
    
    # Get action patterns
    recent_actions = db.query(PlayerAction).filter(
        PlayerAction.session_id == session_id
    ).order_by(PlayerAction.created_at.desc()).limit(50).all()
    
    # Calculate performance metrics
    success_rate = sum(1 for a in recent_actions if a.success) / len(recent_actions) if recent_actions else 0
    
    return {
        "session_stats": stats,
        "performance_metrics": {
            "overall_success_rate": success_rate,
            "actions_per_minute": len(recent_actions) / 10,  # Rough estimate
            "dominant_game": max(stats["session_info"]["games_played"], key=lambda x: x, default="fighting")
        },
        "recent_patterns": [
            {
                "game": a.game_type,
                "action": a.action_type,
                "success": a.success,
                "timestamp": a.created_at.isoformat()
            }
            for a in recent_actions[:10]
        ]
    }

class EnhancedMultiGameSession:
    def __init__(self, session_id: str, db: Session):
        self.session_id = session_id
        self.db = db
        self.current_game = "fighting"
        self.ai_opponent = RuleBasedAIOpponent()
        self.personality_profile = None
        self.action_count = 0
        self.game_states = {
            'fighting': {'player_health': 100, 'ai_health': 100},
            'badminton': {'score_player': 0, 'score_ai': 0, 'rally_count': 0},
            'racing': {'lap': 1, 'position': 2, 'speed': 60}
        }
        
        # Initialize or get session from database
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize or retrieve session from database"""
        session = self.db.query(GameSession).filter(
            GameSession.session_id == self.session_id
        ).first()
        
        if not session:
            session = GameSession(
                session_id=self.session_id,
                current_game=self.current_game
            )
            self.db.add(session)
            self.db.commit()
    
    async def switch_game(self, new_game: str):
        """Handle game switching with personality carryover"""
        self.current_game = new_game
        
        # Update database session
        session = self.db.query(GameSession).filter(
            GameSession.session_id == self.session_id
        ).first()
        if session:
            session.current_game = new_game
            if new_game not in session.games_played:
                session.games_played = session.games_played + [new_game]
            self.db.commit()
        
        return {
            "switched_to": new_game,
            "personality_carried": self.personality_profile is not None,
            "game_state": self.game_states.get(new_game, {})
        }
    
    async def process_player_action(self, action_data: dict):
        """Process player action and generate AI response"""
        
        # Store player action in database
        db_action = PlayerAction(
            session_id=self.session_id,
            game_type=self.current_game,
            action_type=action_data.get('action_type', 'unknown'),
            timestamp=action_data.get('timestamp', time.time()),
            success=action_data.get('success', False),
            action_data=action_data,
            context=action_data.get('context', {})
        )
        self.db.add(db_action)
        self.action_count += 1
        
        # Update personality analysis
        await self._update_personality_analysis(action_data)
        
        # Generate AI response
        ai_response = self.ai_opponent.get_action(
            GameType(self.current_game),
            self.game_states[self.current_game],
            self.personality_profile
        )
        
        # Update AI opponent's learning
        self.ai_opponent.update_player_pattern(action_data)
        
        # Update game state based on actions
        self._update_game_state(action_data, ai_response)
        
        self.db.commit()
        
        return {
            'updated_personality': self.personality_profile.dict() if self.personality_profile else None,
            'ai_response': ai_response,
            'game_state': self.game_states[self.current_game],
            'insights': self._get_current_insights()
        }
    
    async def _update_personality_analysis(self, action_data: dict):
        """Update personality analysis with new action"""
        try:
            # Create mock action for analysis (adapt this to your action structure)
            mock_actions = {
                self.current_game: [action_data]  # Simplified - you'll need proper action objects
            }
            
            # Use your existing analyzer
            self.personality_profile = await multi_game_analyzer.analyze_universal_behavior(mock_actions)
            
            # Update database personality profile
            personality_db = self.db.query(PersonalityProfile).filter(
                PersonalityProfile.session_id == self.session_id
            ).first()
            
            if not personality_db:
                personality_db = PersonalityProfile(session_id=self.session_id)
                self.db.add(personality_db)
            
            # Update personality traits
            personality_db.aggression_level = self.personality_profile.aggression_level
            personality_db.risk_tolerance = self.personality_profile.risk_tolerance
            personality_db.analytical_thinking = self.personality_profile.analytical_thinking
            personality_db.patience_level = self.personality_profile.patience_level
            personality_db.precision_focus = self.personality_profile.precision_focus
            personality_db.competitive_drive = self.personality_profile.competitive_drive
            personality_db.strategic_thinking = self.personality_profile.strategic_thinking
            personality_db.total_actions_analyzed += 1
            
        except Exception as e:
            print(f"Personality analysis error: {e}")
    
    def _update_game_state(self, player_action: dict, ai_action: dict):
        """Update game state based on player and AI actions"""
        if self.current_game == "fighting":
            # Fighting game state updates
            if player_action.get('action_type') == 'attack' and player_action.get('success'):
                self.game_states['fighting']['ai_health'] -= 10
            if ai_action.get('action') in ['aggressive_combo', 'quick_jab'] and random.random() > 0.5:
                self.game_states['fighting']['player_health'] -= 8
                
        elif self.current_game == "badminton":
            # Badminton state updates
            rally_count = self.game_states['badminton'].get('rally_count', 0)
            self.game_states['badminton']['rally_count'] = rally_count + 1
            
            # Simple scoring logic
            if rally_count > 5 and random.random() > 0.7:
                if random.random() > 0.5:
                    self.game_states['badminton']['score_player'] += 1
                else:
                    self.game_states['badminton']['score_ai'] += 1
                self.game_states['badminton']['rally_count'] = 0
                
        elif self.current_game == "racing":
            # Racing state updates
            current_speed = self.game_states['racing'].get('speed', 60)
            if ai_action.get('speed_adjustment'):
                new_speed = current_speed + ai_action['speed_adjustment']
                self.game_states['racing']['speed'] = max(20, min(120, new_speed))
    
    def _get_current_insights(self):
        """Get current cross-game insights"""
        return {
            "actions_processed": self.action_count,
            "current_game": self.current_game,
            "ai_difficulty": self.ai_opponent.difficulty.name,
            "personality_available": self.personality_profile is not None,
            "cross_game_learning": "Active pattern recognition"
        }

# Update your WebSocket endpoint in main.py
@app.websocket("/ws/multi-game/{session_id}")
async def enhanced_multi_game_websocket(websocket: WebSocket, session_id: str, db: Session = Depends(get_db)):
    await websocket.accept()
    session = EnhancedMultiGameSession(session_id, db)
    
    try:
        await websocket.send_json({
            "type": "connection_established",
            "message": "Enhanced multi-game session started",
            "session_id": session_id,
            "supported_games": ["fighting", "badminton", "racing"],
            "current_game": session.current_game
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "game_switch":
                result = await session.switch_game(data["new_game"])
                await websocket.send_json({
                    "type": "game_switched",
                    "data": result
                })
                
            elif data["type"] == "player_action":
                result = await session.process_player_action(data["action"])
                await websocket.send_json({
                    "type": "analysis_update",
                    "data": result
                })
                
            elif data["type"] == "get_status":
                await websocket.send_json({
                    "type": "session_status",
                    "data": {
                        "current_game": session.current_game,
                        "action_count": session.action_count,
                        "game_states": session.game_states,
                        "has_personality": session.personality_profile is not None,
                        "ai_info": session.ai_opponent.get_difficulty_info()
                    }
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
        
# Include API version routing
from fastapi import APIRouter

# V1 API Router
api_v1 = APIRouter(prefix="/api/v1")

# Health check for API versioning
@api_v1.get("/health")
async def api_health():
    return {
        "api_version": "v1",
        "status": "healthy",
        "features": {
            "multi_game_analysis": True,
            "cross_game_ai": True,
            "real_time_websocket": True,
            "voice_processing": False  # Will be True in Week 3
        }
    }

# Include the router
app.include_router(api_v1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)