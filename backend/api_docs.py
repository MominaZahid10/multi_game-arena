from fastapi.openapi.utils import get_openapi
from backend.main import app
from backend.dbmodels.personality import (
    GameType, UniversalAction, FightingAction, BadmintonAction, RacingAction,
    UnifiedPersonality, BehavioralFeature, UniversalAnalysisRequest,
    GameSpecificAnalysisRequest, CrossGameAIRequest, CrossGameAIResponse, MultiGameState
)

def custom_openapi():
    """Generate custom OpenAPI documentation for the AI Battle Arena API"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AI Battle Arena API",
        version="1.0.0",
        description="""## AI Battle Arena Backend API
        
        This API powers the AI Battle Arena game, providing cross-game personality analysis,
        adaptive AI opponents, and real-time gameplay interactions across fighting, badminton,
        and racing games.
        
        ### Key Features
        
        - **Cross-Game Personality Analysis**: Analyzes player behavior across different games
        - **Adaptive AI Opponents**: AI that adapts to player personality and playstyle
        - **Real-Time WebSocket Communication**: For continuous gameplay and personality updates
        - **Game-Specific Action Processing**: Specialized endpoints for each game type
        - **Session Analytics**: Track performance metrics and action patterns
        
        ### Authentication
        
        Most endpoints require a valid session_id which is created when starting a new game session.
        """,
        routes=app.routes,
    )
    
    # Add custom tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Player Analysis",
            "description": "Endpoints for analyzing player behavior and personality"
        },
        {
            "name": "AI Opponents",
            "description": "Endpoints for generating AI actions and strategies"
        },
        {
            "name": "Game Actions",
            "description": "Endpoints for processing game-specific actions"
        },
        {
            "name": "Session Management",
            "description": "Endpoints for managing game sessions and analytics"
        },
        {
            "name": "WebSocket",
            "description": "Real-time communication for continuous gameplay"
        },
        {
            "name": "System",
            "description": "System health and status endpoints"
        }
    ]
    
    # Add security schemes if needed
    # openapi_schema["components"]["securitySchemes"] = {...}
    
    # Add custom examples for request bodies
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    if "examples" not in openapi_schema["components"]:
        openapi_schema["components"]["examples"] = {}
    
    # Example for universal analysis request
    openapi_schema["components"]["examples"]["universal_analysis_request"] = {
        "summary": "Universal Analysis Request",
        "value": {
            "session_id": "abc123",
            "actions": {
                "fighting": [
                    {
                        "move_type": "attack",
                        "combo_count": 3,
                        "success": True
                    }
                ],
                "badminton": [
                    {
                        "shot_type": "smash",
                        "power_level": 0.8,
                        "court_position": [0.3, 0.7],
                        "rally_position": 5
                    }
                ],
                "racing": [
                    {
                        "speed": 95,
                        "overtaking_attempt": True,
                        "crash_occurred": False
                    }
                ]
            }
        }
    }
    
    # Example for fighting game action
    openapi_schema["components"]["examples"]["fighting_action"] = {
        "summary": "Fighting Game Action",
        "value": {
            "session_id": "abc123",
            "player_action": {
                "move_type": "attack",
                "combo_count": 3,
                "success": True
            },
            "game_state": {
                "player_health": 80,
                "ai_health": 65,
                "player_position": {"x": 0.3, "y": 0.5},
                "distance_to_player": 40
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Update the app to use the custom OpenAPI schema
app.openapi = custom_openapi