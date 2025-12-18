import sys
sys.stdout.reconfigure(encoding="utf-8")
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
import numpy as np
ENABLED_GAMES = {
    "fighting": True,
    "badminton": True,  
    "racing": True      
}
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
import asyncio
from typing import List,Dict,Optional,Union
import json
from datetime import datetime
from backend.services.rulebased_ai import MLPoweredAIOpponent
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

global_ai_opponent = MLPoweredAIOpponent()
print("✅ Global AI opponent initialized (model loaded once)")

multi_game_analyzer = MultiGameAnalyzer(
    shared_ml_model=global_ai_opponent.ml_classifier if hasattr(global_ai_opponent, 'ml_classifier') else None
)

strategy_selector = CrossGameStrategySelector()
def get_session_stats_lightweight(db: Session, session_id: str) -> dict:
    """
    ✅ FIXED: Handles None values properly
    """
    session = (
        db.query(GameSession)
        .filter(GameSession.session_id == session_id)
        .first()
    )

    total_actions = (
        db.query(func.count(PlayerAction.id))
        .filter(PlayerAction.session_id == session_id)
        .scalar()
    ) or 0  # ✅ Handle None

    # ✅ FIXED: Return valid data even if session doesn't exist
    if not session and total_actions == 0:
        return {
            "session_info": {
                "session_id": session_id,
                "games_played": [],
                "total_actions": 0,
                "current_game": "unknown"
            },
            "game_breakdown": {},
            "overall_stats": {
                "total_actions": 0,
                "success_rate": 0,
                "games_played_count": 0
            }
        }

    
    # ✅ NEW (FIXED):
    games_played = []
    if session:
        if session.games_played is None:
            games_played = []
        elif isinstance(session.games_played, list):
            games_played = session.games_played
        elif isinstance(session.games_played, str):
            try:
                games_played = json.loads(session.games_played)  # Use global json
            except:
                games_played = []
        else:
            games_played = []
    
    current_game = session.current_game if session and session.current_game else "unknown"

    game_breakdown = {}
    for game_type in ["fighting", "badminton", "racing"]:
        count = (
            db.query(func.count(PlayerAction.id))
            .filter(
                PlayerAction.session_id == session_id,
                PlayerAction.game_type == game_type
            )
            .scalar()
        ) or 0  # ✅ Handle None

        if count > 0:
            game_breakdown[game_type] = {
                "total_actions": count,
                "success_rate": 0.5,
                "last_played": datetime.now().isoformat()
            }

    return {
        "session_info": {
            "session_id": session_id,
            "games_played": games_played,
            "total_actions": total_actions,
            "current_game": current_game,
        },
        "game_breakdown": game_breakdown,
        "overall_stats": {
            "total_actions": total_actions,
            "success_rate": 0.5,
            "games_played_count": len(game_breakdown)
        }
    }

@app.on_event("startup")
async def warmup_model():
    """Model is already loaded in global_ai_opponent - warmup not needed"""
    print("✅ Warmup skipped - model loaded at initialization")

@app.get("/")
async def root():
    return {"message":"AI Multi-Game Arena API is running", "version":"2.0.0"}

from pydantic import BaseModel

class FightingMLInput(BaseModel):
    aggression_rate: float
    defense_ratio: float
    combo_preference: float
    reaction_time: float


@app.post("/api/v1/ml/fighting/predict")
def ml_fighting_predict(data: FightingMLInput):
    try:
        features = [[
            data.aggression_rate,
            data.defense_ratio,
            data.combo_preference,
            data.reaction_time
        ]]
        if not (hasattr(global_ai_opponent, 'ml_classifier') and global_ai_opponent.ml_classifier):
            raise HTTPException(status_code=503, detail="ML model not loaded")
        ml_clf = global_ai_opponent.ml_classifier
        if not hasattr(ml_clf, 'personality_classifier'):
            raise HTTPException(status_code=503, detail="Personality classifier not found")
    
        game_features = {'fighting': features[0]}
        result = ml_clf.predict_personality(game_features)
        # Map archetype name to ID
        archetype_to_id = {
            "🔥 Aggressive Dominator": 0,
            "🧠 Strategic Analyst": 1,
            "⚡ Risk-Taking Maverick": 2,
            "🛡️ Defensive Tactician": 3,
            "🎯 Precision Master": 4,
            "🌪️ Chaos Creator": 5,
            "📊 Data-Driven Player": 6,
            "🏆 Victory Seeker": 7
        }
        archetype_name = result.get('personality_archetype', '🔥 Aggressive Dominator')
        archetype_id = archetype_to_id.get(archetype_name, 0)
        return {
            "prediction": archetype_id,
            "archetype_name": archetype_name,
            "confidence": result.get('category_confidence', 0.0)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.post("/api/v1/player/analyze-universal")
async def analyze_universal_player(request: Request, db: Session = Depends(get_db)):
    return {"status": "Analysis queued"}

@app.post("/api/v1/games/fighting/action")
async def process_fighting_action(
    session_id: str,
    action_request: dict,
    db: Session = Depends(get_db)
):
    """
    ✅ FIXED: Now saves actions to database properly
    """
    if not hasattr(process_fighting_action, '_req_counter'):
        process_fighting_action._req_counter = 0
    process_fighting_action._req_counter += 1
    should_log = process_fighting_action._req_counter % 5 == 0
    
    try:
        start_time = time.time()
        
        action_data = action_request.get('action_data', action_request)
        context = action_data.get('context', {})
        if not isinstance(context, dict): 
            context = {}
        
        player_pos = action_data.get('position', [0, 0])
        if isinstance(player_pos, dict):
            player_pos = [player_pos.get('x', 0), player_pos.get('z', player_pos.get('y', 0))]
        
        ai_pos = context.get('ai_position', {'x': 4.5, 'z': 0})
        if isinstance(ai_pos, list):
            ai_pos = {'x': ai_pos[0], 'z': ai_pos[2] if len(ai_pos) > 2 else 0}
            
        game_state = {
            'player_health': context.get('player_health', 100),
            'ai_health': context.get('ai_health', 100),
            'distance_to_player': context.get('distance_to_opponent', 5.0),
            'player_position': {'x': player_pos[0], 'z': player_pos[1] if len(player_pos) > 1 else 0},
            'ai_position': ai_pos  
        }
        if should_log:
            print(f"🔥 Request #{process_fighting_action._req_counter}: player=({player_pos[0]:.1f},{player_pos[1] if len(player_pos)>1 else 0:.1f}), ai=({ai_pos.get('x',0):.1f},{ai_pos.get('z',0):.1f})")

        def get_personality_sync():
            return db.query(PersonalityProfile).filter(
                PersonalityProfile.session_id == session_id
            ).first()
        
        p_db = await asyncio.to_thread(get_personality_sync)

        p_obj = None
        if p_db:
            p_obj = UnifiedPersonality(
                aggression_level=p_db.aggression_level,
                risk_tolerance=p_db.risk_tolerance,
                strategic_thinking=p_db.strategic_thinking,
                patience_level=p_db.patience_level,
                precision_focus=p_db.precision_focus,
                competitive_drive=p_db.competitive_drive,
                analytical_thinking=p_db.analytical_thinking,
                adaptability=p_db.adaptability
            )
        
        ai_action_result = global_ai_opponent.get_action(GameType.FIGHTING, game_state, p_obj)
        using_ml = False
        ml_archetype = None
        if isinstance(ai_action_result, dict):
            ai_action_str = ai_action_result.get('action', 'idle')
            raw_pos = ai_action_result.get('position', {'x': 4.0, 'y': 0.0, 'z': 0.0})
            ai_position = {
                'x': float(raw_pos.get('x', 0) if isinstance(raw_pos, dict) else raw_pos[0]),
                'y': float(raw_pos.get('y', 0) if isinstance(raw_pos, dict) else (raw_pos[1] if len(raw_pos)>1 else 0)),
                'z': float(raw_pos.get('z', 0) if isinstance(raw_pos, dict) else (raw_pos[2] if len(raw_pos)>2 else 0))
            }
            using_ml = ai_action_result.get('using_ml', False)
            ml_archetype = ai_action_result.get('ml_archetype', None)
        else:
            ai_action_str = ai_action_result
            ai_position = {'x': 4.0, 'y': 0.0, 'z': 0.0}

        # 🚀 CRITICAL FIX: Save action to database
        def save_action_sync():
            try:
                # Extract move type from action string
                move_type = ai_action_str.split('_')[0] if '_' in ai_action_str else ai_action_str
                
                db_action = PlayerAction(
                    session_id=session_id,
                    game_type="fighting",
                    action_type=ai_action_str,
                    move_type=move_type,  # e.g., 'punch', 'kick', 'block'
                    timestamp=time.time(),
                    success=True,  # AI actions are considered successful
                    action_data=action_request,  # Store full context
                    combo_count=0  # Can be updated based on action analysis
                )
                db.add(db_action)
                db.commit()
                
                # Also ensure GameSession exists
                game_session = db.query(GameSession).filter(
                    GameSession.session_id == session_id
                ).first()
                
                if not game_session:
                    game_session = GameSession(
                        session_id=session_id,
                        current_game="fighting",
                        games_played=["fighting"]
                    )
                    db.add(game_session)
                else:
                    game_session.current_game = "fighting"
                    if game_session.games_played is None:
                        game_session.games_played = ["fighting"]
                    elif "fighting" not in game_session.games_played:
                        if isinstance(game_session.games_played, str):
                            try:
                                games = json.loads(game_session.games_played)
                                games.append("fighting")
                                game_session.games_played = games
                            except:
                                game_session.games_played = ["fighting"]
                        else:
                            game_session.games_played.append("fighting")
                
                db.commit()
                return True
            except Exception as e:
                db.rollback()
                print(f"❌ Database save error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Save to database asynchronously
        save_success = await asyncio.to_thread(save_action_sync)

        stats = {"actions_today": process_fighting_action._req_counter}
        response = {
            "success": True,
            "ai_action": {
                "action": ai_action_str,
                "position": ai_position,
                "timestamp": time.time() * 1000,
                "confidence": 0.85,
                "strategy": "adaptive_combat",
                "using_ml": using_ml,  
                "ml_archetype": ml_archetype 
            },
            "game_state": game_state,
            "session_stats": stats,
            "personality": None,
            "analytics_updated": save_success,
            "database_saved": save_success
        }
        
        if should_log:
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"📤 Response #{process_fighting_action._req_counter}: action={ai_action_str}, pos=({ai_position['x']:.1f},{ai_position['z']:.1f}), time={elapsed_ms:.0f}ms, DB_saved={save_success}")
        
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Ensure rollback on error
        try:
            db.rollback()
        except:
            pass
        return {
            "success": False,
            "ai_action": {
                "action": "idle", 
                "position": {"x": 5, "y": 0, "z": 0}, 
                "timestamp": time.time() * 1000
            },
            "error": str(e),
            "database_saved": False
        }

@app.post("/api/v1/games/badminton/action")
async def process_badminton_action(session_id: str, action_data: dict, db: Session = Depends(get_db)):
    if not ENABLED_GAMES["badminton"]:
        raise HTTPException(
            status_code=503,
            detail="Badminton pipeline implemented but disabled in current deployment"
        )
    try:
        def process_sync():
            payload = action_data.get('action_data', action_data)
            game_state = {
                'shuttlecock_position': payload.get('context', {}).get('shuttlecock_position', {'x': 0, 'y': 2}),
                'rally_count': payload.get('context', {}).get('rally_count', 0),
            }
            ai_action = global_ai_opponent.get_action(GameType.BADMINTON, game_state, None)
            shuttle_pos = game_state['shuttlecock_position']
            ai_target = {
                'x': 5 + (1 if 'smash' in str(ai_action) else -0.5),
                'z': shuttle_pos.get('y', 0) + 0.5 if isinstance(shuttle_pos, dict) else 0.5
            }
            db_action = PlayerAction(
                session_id=session_id,
                game_type="badminton",
                action_type=payload.get('shot_type', 'shot'),
                timestamp=time.time(),
                success=payload.get('success', False),
                action_data=action_data
            )
            db.add(db_action)
            db.commit()
            return {
                "success": True,
                "ai_action": {
                    "action": ai_action,
                    "target": ai_target,
                    "timestamp": time.time() * 1000
                }
            }
        return await asyncio.to_thread(process_sync)
    except Exception as e:
        print(f"Badminton Error: {e}")
        return {"success": False, "ai_action": {"action": "clear", "target": {"x": 5, "z": 0}}}

@app.post("/api/v1/games/racing/action")
async def process_racing_action(session_id: str, action_data: dict, db: Session = Depends(get_db)):
    if not ENABLED_GAMES["racing"]:
        raise HTTPException(
            status_code=503,
            detail="Racing pipeline implemented but disabled in current deployment"
        )
    try:
        def process_sync():
            payload = action_data.get('action_data', action_data)
            track_pos = payload.get('position_on_track', [0, 0])
            if isinstance(track_pos, dict): track_pos = [track_pos.get('x', 0), track_pos.get('z', 0)]
            game_state = {
                'player_position': {'x': track_pos[0], 'z': track_pos[1]},
                'position': payload.get('context', {}).get('position_in_race', 2)
            }
            ai_action = global_ai_opponent.get_action(GameType.RACING, game_state, None)
            ai_position = {
                'x': track_pos[0] + (1.5 if 'overtake' in str(ai_action) else -0.5),
                'y': -1.75,
                'z': track_pos[1] - 3
            }
            db_action = PlayerAction(
                session_id=session_id,
                game_type="racing",
                action_type=payload.get('action_type', 'drive'),
                timestamp=time.time(),
                action_data=action_data
            )
            db.add(db_action)
            db.commit()
            return {
                "success": True,
                "ai_action": {
                    "action": ai_action,
                    "position": ai_position,
                    "speed": 70 if 'overtake' in str(ai_action) else 60
                }
            }
        return await asyncio.to_thread(process_sync)
    except Exception as e:
        print(f"Racing Error: {e}")
        return {"success": False, "ai_action": {"action": "maintain_speed", "position": {"x": 0, "y": 0, "z": 0}}}

@app.get("/api/v1/analytics/{session_id}")
async def get_session_analytics(session_id: str, db: Session = Depends(get_db)):
    return await asyncio.to_thread(get_session_stats_lightweight, db, session_id)

@app.get("/api/v1/personality/{session_id}")
async def get_personality_profile(session_id: str, db: Session = Depends(get_db)):
    """
    ✅ FIXED: Get REAL personality profile from ML model + database
    """
    def get_profile_sync():
        # 1. Get personality from database
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == session_id
        ).first()
        
        # 2. Get recent actions for ML prediction
        recent_actions = db.query(PlayerAction).filter(
            PlayerAction.session_id == session_id
        ).order_by(PlayerAction.timestamp.desc()).limit(50).all()
        
        if not profile or not recent_actions:
            return {
                "status": "no_data",
                "message": "No personality data yet - keep playing!",
                "personality": {
                    "aggression": 0.5,
                    "patience": 0.5,
                    "strategic_thinking": 0.5,
                    "risk_tolerance": 0.5,
                    "precision_focus": 0.5,
                    "adaptability": 0.5,
                    "competitive_drive": 0.5,
                    "analytical_thinking": 0.5
                },
                "archetype": "🎮 Balanced Player",
                "playstyle": "🎯 Adaptive Gamer",
                "description": "Play more to reveal your personality!",
                "confidence": 0.3
            }
        
        # 3. ✅ USE ML MODEL if available
        ml_prediction = None
        try:
            if hasattr(global_ai_opponent, 'ml_classifier') and global_ai_opponent.ml_classifier:
                ml_clf = global_ai_opponent.ml_classifier
                
                # Extract features from recent actions
                fighting_actions = [a for a in recent_actions if a.game_type == 'fighting']
                badminton_actions = [a for a in recent_actions if a.game_type == 'badminton']
                racing_actions = [a for a in recent_actions if a.game_type == 'racing']
                
                game_features = {}
                
                # Fighting features
                if fighting_actions:
                    game_features['fighting'] = extract_fighting_features(fighting_actions)
                
                # Badminton features  
                if badminton_actions:
                    game_features['badminton'] = extract_badminton_features(badminton_actions)
                
                # Racing features
                if racing_actions:
                    game_features['racing'] = extract_racing_features(racing_actions)
                
                # Get ML prediction
                if game_features:
                    ml_prediction = ml_clf.predict_personality(game_features)
                    print(f"✅ ML Prediction: {ml_prediction.get('personality_archetype')}")
        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. ✅ USE ML RESULTS if available, otherwise fall back to DB
        if ml_prediction:
            personality_scores = ml_prediction['personality_scores']
            archetype = ml_prediction.get('personality_archetype', '🎮 Multi-Game Player')
            playstyle = ml_prediction.get('playstyle_category', '🎯 Adaptive Gamer')
            confidence = ml_prediction.get('confidence_score', 0.7)
            
            return {
                "status": "success",
                "personality": {
                    "aggression": personality_scores['aggression_level'],
                    "patience": personality_scores['patience_level'],
                    "strategic_thinking": personality_scores['strategic_thinking'],
                    "risk_tolerance": personality_scores['risk_tolerance'],
                    "precision_focus": personality_scores['precision_focus'],
                    "adaptability": personality_scores.get('adaptability', 0.5),
                    "competitive_drive": personality_scores['competitive_drive'],
                    "analytical_thinking": personality_scores['analytical_thinking']
                },
                "raw_scores": personality_scores,
                "archetype": archetype,
                "personality_type": archetype,
                "playstyle": playstyle,
                "description": f"You are a {archetype} with {int(personality_scores['aggression_level']*100)}% aggression and {int(personality_scores['patience_level']*100)}% patience.",
                "confidence": confidence,
                "ml_powered": True,
                "actions_analyzed": len(recent_actions)
            }
        else:
            # Fallback to database values
            aggression = profile.aggression_level or 0.5
            patience = profile.patience_level or 0.5
            strategic = profile.strategic_thinking or 0.5
            risk = profile.risk_tolerance or 0.5
            
            # Use DB archetype if available
            archetype = profile.personality_archetype or "🎮 Balanced Player"
            playstyle = profile.playstyle_category or "🎯 Adaptive Gamer"
            
            return {
                "status": "success",
                "personality": {
                    "aggression": aggression,
                    "patience": patience,
                    "strategic_thinking": strategic,
                    "risk_tolerance": risk,
                    "precision_focus": profile.precision_focus or 0.5,
                    "adaptability": profile.adaptability or 0.5,
                    "competitive_drive": profile.competitive_drive or 0.5,
                    "analytical_thinking": profile.analytical_thinking or 0.5
                },
                "archetype": archetype,
                "personality_type": archetype,
                "playstyle": playstyle,
                "description": f"You are a {archetype} with {aggression*100:.0f}% aggression and {patience*100:.0f}% patience.",
                "raw_scores": {
                    "aggression_level": aggression,
                    "patience_level": patience,
                    "strategic_thinking": strategic,
                    "risk_tolerance": risk,
                    "precision_focus": profile.precision_focus or 0.5,
                    "competitive_drive": profile.competitive_drive or 0.5,
                    "analytical_thinking": profile.analytical_thinking or 0.5,
                    "adaptability": profile.adaptability or 0.5
                },
                "confidence": profile.overall_confidence or 0.5,
                "ml_powered": False,
                "actions_analyzed": len(recent_actions)
            }
    
    return await asyncio.to_thread(get_profile_sync)
# ============================================================================
# 🧠 QUICK PERSONALITY ANALYSIS - Every 10 actions
# ============================================================================

def extract_fighting_features(actions):
    """Extract fighting ML features from recent actions"""
    import numpy as np
    if not actions:
        return [0.5, 0.5, 0.5, 0.5]
    
    attack_count = sum(1 for a in actions if a.move_type in ['attack', 'punch', 'kick', 'combo'])
    defend_count = sum(1 for a in actions if a.move_type in ['block', 'dodge'])
    combo_total = sum(a.combo_count or 0 for a in actions)
    success_count = sum(1 for a in actions if a.success)
    
    total = len(actions)
    aggression_rate = attack_count / total if total > 0 else 0.5
    defense_ratio = defend_count / total if total > 0 else 0.5
    combo_preference = combo_total / (total * 3) if total > 0 else 0.5
    reaction_time = success_count / total if total > 0 else 0.5
    
    return [aggression_rate, defense_ratio, combo_preference, reaction_time]


def extract_badminton_features(actions):
    """Extract badminton ML features"""
    import numpy as np
    if not actions:
        return [0.5, 0.5, 0.5, 0.5]
    
    shot_types = set(a.shot_type for a in actions if hasattr(a, 'shot_type') and a.shot_type)
    power_levels = [a.power_level for a in actions if hasattr(a, 'power_level') and a.power_level is not None]
    rally_positions = [a.rally_position for a in actions if hasattr(a, 'rally_position') and a.rally_position]
    
    shot_variety = len(shot_types) / 5.0 if shot_types else 0.5
    power_control = 1.0 - np.var(power_levels) if power_levels else 0.5
    court_positioning = 0.5  # Placeholder
    rally_patience = np.mean(rally_positions) / 10.0 if rally_positions else 0.5
    
    return [shot_variety, power_control, court_positioning, rally_patience]


def extract_racing_features(actions):
    """Extract racing ML features"""
    import numpy as np
    if not actions:
        return [0.5, 0.5, 0.5, 0.5]
    
    speeds = [a.speed for a in actions if hasattr(a, 'speed') and a.speed and a.speed > 0]
    crashes = sum(1 for a in actions if hasattr(a, 'crash_occurred') and a.crash_occurred)
    overtakes = sum(1 for a in actions if hasattr(a, 'overtaking_attempt') and a.overtaking_attempt)
    
    total = len(actions)
    speed_preference = np.mean(speeds) / 120.0 if speeds else 0.5
    precision_level = 1.0 - (crashes / total) if total > 0 else 0.5
    overtaking_aggression = overtakes / total if total > 0 else 0.5
    consistency = 1.0 - np.var(speeds) / 400 if len(speeds) > 1 else 0.5
    
    return [speed_preference, precision_level, overtaking_aggression, consistency]


@app.post("/api/v1/player/quick-analyze")
async def quick_personality_analysis(session_id: str, db: Session = Depends(get_db)):
    """
    ✅ FIXED: Actually run personality analysis using ML model
    """
    def analyze_sync():
        try:
            recent_actions = db.query(PlayerAction).filter(
                PlayerAction.session_id == session_id
            ).order_by(PlayerAction.timestamp.desc()).limit(100).all()
            
            if len(recent_actions) < 10:
                return {
                    "status": "insufficient_data",
                    "message": f"Need {10 - len(recent_actions)} more actions for analysis",
                    "actions_count": len(recent_actions)
                }
            
            # Use ML model for analysis
            if hasattr(global_ai_opponent, 'ml_classifier') and global_ai_opponent.ml_classifier:
                ml_clf = global_ai_opponent.ml_classifier
                
                fighting_actions = [a for a in recent_actions if a.game_type == 'fighting']
                badminton_actions = [a for a in recent_actions if a.game_type == 'badminton']
                racing_actions = [a for a in recent_actions if a.game_type == 'racing']
                
                game_features = {}
                
                if fighting_actions:
                    game_features['fighting'] = extract_fighting_features(fighting_actions)
                
                if badminton_actions:
                    game_features['badminton'] = extract_badminton_features(badminton_actions)
                
                if racing_actions:
                    game_features['racing'] = extract_racing_features(racing_actions)
                
                if not game_features:
                    return {
                        "status": "no_features",
                        "message": "No valid game features extracted"
                    }
                
                ml_prediction = ml_clf.predict_personality(game_features)
                personality_scores = ml_prediction['personality_scores']
                
                profile = db.query(PersonalityProfile).filter(
                    PersonalityProfile.session_id == session_id
                ).first()
                
                if not profile:
                    profile = PersonalityProfile(session_id=session_id)
                    db.add(profile)
                
                profile.aggression_level = float(personality_scores['aggression_level'])
                profile.risk_tolerance = float(personality_scores['risk_tolerance'])
                profile.analytical_thinking = float(personality_scores['analytical_thinking'])
                profile.patience_level = float(personality_scores['patience_level'])
                profile.precision_focus = float(personality_scores['precision_focus'])
                profile.competitive_drive = float(personality_scores['competitive_drive'])
                profile.strategic_thinking = float(personality_scores['strategic_thinking'])
                profile.adaptability = float(np.mean([
                    personality_scores['strategic_thinking'],
                    personality_scores['analytical_thinking']
                ]))
                # Update archetype and playstyle
                profile.personality_archetype = ml_prediction.get('personality_archetype', '🎮 Multi-Game Player')
                profile.playstyle_category = ml_prediction.get('playstyle_category', '🎯 Adaptive Gamer')
                profile.category_confidence = float(ml_prediction.get('category_confidence', 0.7))
                profile.overall_confidence = float(ml_prediction.get('confidence_score', 0.7))
                profile.total_actions_analyzed = len(recent_actions)
                db.commit()
                
                print(f"✅ Personality updated: {profile.personality_archetype}")
                
                return {
                    "status": "success",
                    "message": "Personality analysis complete",
                    "personality_type": profile.personality_archetype,
                    "playstyle": profile.playstyle_category,
                    "confidence": profile.overall_confidence,
                    "actions_analyzed": len(recent_actions),
                    "traits": {
                        "aggression_level": profile.aggression_level,
                        "risk_tolerance": profile.risk_tolerance,
                        "patience_level": profile.patience_level,
                        "strategic_thinking": profile.strategic_thinking,
                        "precision_focus": profile.precision_focus,
                        "competitive_drive": profile.competitive_drive,
                        "analytical_thinking": profile.analytical_thinking,
                        "adaptability": profile.adaptability
                    }
                }
            else:
                return {
                    "status": "ml_unavailable",
                    "message": "ML model not loaded - using rule-based analysis",
                    "actions_count": len(recent_actions)
                }
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e)
            }
    
    return await asyncio.to_thread(analyze_sync)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)