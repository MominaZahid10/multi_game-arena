from fastapi import FastAPI,HTTPException,Depends,WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import asyncio
from typing import List,Dict,Optional,Union
import json
from datetime import datetime

from config import settings
from databaseconn import get_db,engine,Base
from models.personality import(
    UniversalAnalysisRequest, UnifiedPersonality, 
    GameSpecificAnalysisRequest, CrossGameAIRequest, CrossGameAIResponse,
    MultiGameState, GameType
)
from models.games import GameSession, PlayerAction, PersonalityProfile

from services.personality_analyzer import MultiGameAnalyzer
from services.crossgame_strategy import CrossGameStrategySelector

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
    request: UniversalAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    DAY 1 DELIVERABLE: Universal Player Analysis Endpoint
    Analyze player behavior across Fighting, Badminton, and Car Racing games
    """
    try:
        # Get or create game session
        session = db.query(GameSession).filter(
            GameSession.session_id == request.session_id
        ).first()
        
        if not session:
            session = GameSession(
                session_id=request.session_id,
                current_game="fighting"
            )
            db.add(session)
        
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
        
        # Store actions in database
        for action in all_actions:
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
        
        # Update session
        session.total_actions += len(all_actions)
        session.games_played = list(games_played)
        
        # Perform cross-game analysis
        unified_personality = await multi_game_analyzer.analyze_universal_behavior({
            "fighting": request.fighting_actions,
            "badminton": request.badminton_actions,
            "racing": request.racing_actions
        })
        
        # Get or update personality profile
        personality_db = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == request.session_id
        ).first()
        
        if not personality_db:
            personality_db = PersonalityProfile(session_id=request.session_id)
            db.add(personality_db)
        
        # Update personality traits
        personality_db.aggression_level = unified_personality.aggression_level
        personality_db.risk_tolerance = unified_personality.risk_tolerance
        personality_db.analytical_thinking = unified_personality.analytical_thinking
        personality_db.patience_level = unified_personality.patience_level
        personality_db.precision_focus = unified_personality.precision_focus
        personality_db.competitive_drive = unified_personality.competitive_drive
        personality_db.strategic_thinking = unified_personality.strategic_thinking
        personality_db.total_actions_analyzed += len(all_actions)
        
        db.commit()
        
        return {
            "unified_personality": unified_personality.dict(),
            "cross_game_insights": multi_game_analyzer.get_cross_game_insights(),
            "session_stats": {
                "total_actions": session.total_actions,
                "games_played": session.games_played,
                "session_id": session.session_id
            }
        }
        
    except Exception as e:
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

@app.post("/api/v1/voice/process")
async def process_voice_command(voice_data: dict):
    """
    Voice Processing Endpoint (Placeholder for Day 3)
    """
    return {
        "status": "placeholder",
        "message": "Voice processing will be implemented in Week 3",
        "input_received": voice_data.get("current_game", "unknown")
    }


class MultiGameSession:
    def __init__(self):
        self.current_game = None
        self.personality_profile = None
        self.session_data = {}
    
    async def switch_game(self, new_game: str):
        """Handle game switching with personality carryover"""
        self.current_game = new_game
        return {
            "switched_to": new_game,
            "personality_carried": self.personality_profile is not None
        }
    
    async def process_action(self, action_data: dict):
        """Process action and update analysis"""
        # Placeholder for real-time action processing
        return {
            "updated_personality": self.personality_profile,
            "ai_response": {"action": "placeholder", "reasoning": "Processing..."},
            "insights": {"message": "Real-time processing active"}
        }
        

@app.websocket("/ws/multi-game")
async def multi_game_websocket(websocket: WebSocket):
    await websocket.accept()
    session = MultiGameSession()
    
    try:
        await websocket.send_json({
            "type": "connection_established",
            "message": "Multi-game session started",
            "supported_games": ["fighting", "badminton", "racing"]
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "game_switch":
                result = await session.switch_game(data["new_game"])
                await websocket.send_json({
                    "type": "game_switched",
                    "new_game": data["new_game"],
                    "personality_carryover": result
                })
                
            elif data["type"] == "action":
                result = await session.process_action(data["action"])
                await websocket.send_json({
                    "type": "analysis_update",
                    "unified_personality": result["updated_personality"],
                    "ai_response": result["ai_response"],
                    "cross_game_insights": result["insights"]
                })
                
            elif data["type"] == "get_status":
                await websocket.send_json({
                    "type": "session_status",
                    "current_game": session.current_game,
                    "has_personality_profile": session.personality_profile is not None,
                    "timestamp": datetime.now().isoformat()
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
