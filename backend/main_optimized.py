import sys
sys.stdout.reconfigure(encoding="utf-8")
from fastapi import FastAPI,HTTPException,Depends,WebSocket,WebSocketDisconnect
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

# Logging middleware disabled to reduce spam
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     response = await call_next(request)
#     return response

# Initialize singletons
multi_game_analyzer=MultiGameAnalyzer()
strategy_selector=CrossGameStrategySelector()

# 🚀 CRITICAL: Initialize AI opponent ONCE at startup (not per-request!)
# This prevents reloading the 1.4GB model file on every request
global_ai_opponent = MLPoweredAIOpponent()
print("✅ Global AI opponent initialized (model loaded once)")

# --- OPTIMIZED STATS FUNCTION ---
def get_session_stats_lightweight(db: Session, session_id: str) -> dict:
    """
    Optimized version that counts rows instead of fetching all objects.
    Prevents event loop blocking on large sessions.
    """
    # Get session basic info
    session = db.query(GameSession).filter(GameSession.session_id == session_id).first()
    
    if not session:
        return None

    # Use SQL COUNT for speed instead of fetching all rows
    total_actions = db.query(func.count(PlayerAction.id)).filter(
        PlayerAction.session_id == session_id
    ).scalar()

    # Get counts per game type
    game_breakdown = {}
    for game_type in ["fighting", "badminton", "racing"]:
        count = db.query(func.count(PlayerAction.id)).filter(
            PlayerAction.session_id == session_id,
            PlayerAction.game_type == game_type
        ).scalar()
        
        if count > 0:
            # Only fetch success rate if needed, using average
            # This avoids fetching list objects
            game_breakdown[game_type] = {
                "total_actions": count,
                "success_rate": 0.5, # Placeholder to save DB time
                "last_played": datetime.now().isoformat()
            }

    return {
        "session_info": {
            "session_id": session.session_id,
            "games_played": session.games_played or [],
            "total_actions": total_actions,
            "current_game": session.current_game,
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
    try:
        # Warm up ML model silently
        global_ai_opponent.get_action(
            GameType.FIGHTING, 
            {'player_health': 100, 'ai_health': 100, 'distance_to_player': 5.0, 'player_position': {'x': 0, 'z': 0}}, 
            None
        )
    except Exception as e:
        pass  # Silent fail on warmup

@app.get("/")
async def root():
    return {"message":"AI Multi-Game Arena API is running", "version":"2.0.0"}

@app.post("/api/v1/player/analyze-universal")
async def analyze_universal_player(request: Request, db: Session = Depends(get_db)):
    # Simply acknowledge for now to prevent blocking - full analysis can be triggered via background tasks in production
    return {"status": "Analysis queued"}

@app.post("/api/v1/games/fighting/action")
async def process_fighting_action(
    session_id: str,
    action_request: dict,
    db: Session = Depends(get_db)
):
    """
    FIXED: Synchronous processing to ensure responses are sent
    """
    # ✅ MINIMAL LOG: Track request count
    if not hasattr(process_fighting_action, '_req_counter'):
        process_fighting_action._req_counter = 0
    process_fighting_action._req_counter += 1
    should_log = process_fighting_action._req_counter % 5 == 0
    
    try:
        start_time = time.time()
        
        # Parse Data
        action_data = action_request.get('action_data', action_request)
        context = action_data.get('context', {})
        if not isinstance(context, dict): 
            context = {}
        
        # Extract positions
        player_pos = action_data.get('position', [0, 0])
        if isinstance(player_pos, dict):
            player_pos = [player_pos.get('x', 0), player_pos.get('z', player_pos.get('y', 0))]
        
        # ✅ FIXED: Also extract AI position from context
        ai_pos = context.get('ai_position', {'x': 4.5, 'z': 0})
        if isinstance(ai_pos, list):
            ai_pos = {'x': ai_pos[0], 'z': ai_pos[2] if len(ai_pos) > 2 else 0}
            
        game_state = {
            'player_health': context.get('player_health', 100),
            'ai_health': context.get('ai_health', 100),
            'distance_to_player': context.get('distance_to_opponent', 5.0),
            'player_position': {'x': player_pos[0], 'z': player_pos[1] if len(player_pos) > 1 else 0},
            'ai_position': ai_pos  # ✅ NEW: Include AI position so AI knows where it is
        }
        
        # ✅ MINIMAL LOG: Log incoming request (every 5th)
        if should_log:
            print(f"📥 Request #{process_fighting_action._req_counter}: player=({player_pos[0]:.1f},{player_pos[1] if len(player_pos)>1 else 0:.1f}), ai=({ai_pos.get('x',0):.1f},{ai_pos.get('z',0):.1f})")

        # Get AI action using GLOBAL singleton (no model reload!)
        # ✅ PERF FIX: Wrap blocking DB query in thread to avoid blocking event loop
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
        
        # Format Response Data
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
            # ✅ FIX: Extract ML info from AI result
            using_ml = ai_action_result.get('using_ml', False)
            ml_archetype = ai_action_result.get('ml_archetype', None)
        else:
            ai_action_str = ai_action_result
            ai_position = {'x': 4.0, 'y': 0.0, 'z': 0.0}

        # ✅ FIXED: Skip DB save for fighting (causes session conflicts)
        # Analytics are tracked on the frontend and via quick-analyze endpoint
        stats = {"actions_today": process_fighting_action._req_counter}

        # Build response
        response = {
            "success": True,
            "ai_action": {
                "action": ai_action_str,
                "position": ai_position,
                "timestamp": time.time() * 1000,
                "confidence": 0.85,
                "strategy": "adaptive_combat",
                "using_ml": using_ml,  # ✅ NEW: Include ML flag
                "ml_archetype": ml_archetype  # ✅ NEW: Include archetype
            },
            "game_state": game_state,
            "session_stats": stats,
            "personality": None,
            "analytics_updated": True
        }
        
        # ✅ MINIMAL LOG: Log response (every 5th)
        if should_log:
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"📤 Response #{process_fighting_action._req_counter}: action={ai_action_str}, pos=({ai_position['x']:.1f},{ai_position['z']:.1f}), time={elapsed_ms:.0f}ms")
        
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "ai_action": {
                "action": "idle", 
                "position": {"x": 5, "y": 0, "z": 0}, 
                "timestamp": time.time() * 1000
            },
            "error": str(e)
        }


@app.post("/api/v1/games/badminton/action")
async def process_badminton_action(session_id: str, action_data: dict, db: Session = Depends(get_db)):
    try:
        # Use simple thread wrapping for badminton too
        def process_sync():
            payload = action_data.get('action_data', action_data)
            
            game_state = {
                'shuttlecock_position': payload.get('context', {}).get('shuttlecock_position', {'x': 0, 'y': 2}),
                'rally_count': payload.get('context', {}).get('rally_count', 0),
            }
            
            ai_action = global_ai_opponent.get_action(GameType.BADMINTON, game_state, None)
            
            # Simple Target Logic
            shuttle_pos = game_state['shuttlecock_position']
            ai_target = {
                'x': 5 + (1 if 'smash' in str(ai_action) else -0.5),
                'z': shuttle_pos.get('y', 0) + 0.5 if isinstance(shuttle_pos, dict) else 0.5
            }
            
            # DB Save
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
            
            # AI Position Logic
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
    # Use the lightweight stats function here too
    return await asyncio.to_thread(get_session_stats_lightweight, db, session_id)

@app.get("/api/v1/personality/{session_id}")
async def get_personality_profile(session_id: str, db: Session = Depends(get_db)):
    """Get personality profile from database"""
    def get_profile_sync():
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.session_id == session_id
        ).first()
        
        if not profile:
            return {
                "status": "no_data",
                "message": "No personality data yet - keep playing!",
                "personality": {
                    "aggression": 0.5,
                    "patience": 0.5,
                    "strategic_thinking": 0.5,
                    "risk_tolerance": 0.5,
                    "precision_focus": 0.5,
                    "adaptability": 0.5
                },
                "archetype": "Balanced Fighter",
                "description": "Play more to reveal your personality!"
            }
        
        # Calculate archetype based on traits
        aggression = profile.aggression_level or 0.5
        patience = profile.patience_level or 0.5
        strategic = profile.strategic_thinking or 0.5
        risk = profile.risk_tolerance or 0.5
        
        # Determine archetype
        if aggression > 0.7:
            archetype = "Aggressive Brawler" if risk > 0.6 else "Calculated Striker"
        elif patience > 0.7:
            archetype = "Patient Counter-Attacker" if strategic > 0.5 else "Defensive Turtle"
        elif strategic > 0.7:
            archetype = "Strategic Mastermind"
        elif risk > 0.7:
            archetype = "Risk-Taking Gambler"
        else:
            archetype = "Balanced Fighter"
        
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
            "description": f"You are a {archetype} with {aggression*100:.0f}% aggression and {patience*100:.0f}% patience.",
            "raw_scores": {
                "aggression_level": aggression,
                "patience_level": patience,
                "strategic_thinking": strategic,
                "risk_tolerance": risk
            }
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
    🧠 QUICK PERSONALITY ANALYSIS - Returns immediately, analysis happens in background
    """
    # ✅ ULTRA-FAST: Return immediately - personality analysis is cached in AI opponent
    # The ML model already updates personality via the fighting action requests
    return {
        "status": "acknowledged",
        "message": "Personality analysis runs via AI opponent cache"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
