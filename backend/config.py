from typing import Optional
from pydantic_settings import BaseSettings
from pathlib import Path
from pydantic import field_validator

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Multi-Game Arena"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    DATABASE_URL: Optional[str] = None
    
    REDIS_URL: str = "redis://localhost:6379"
    
    MODEL_UPDATE_INTERVAL: int = 30
    PERSONALITY_CONFIDENCE_THRESHOLD: float = 0.7
    CROSS_GAME_ANALYSIS_WINDOW: int = 100
    ML_MODEL_PATH: str = "./models/"
    
    GROQ_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    VOICE_PROCESSING_ENABLED: bool = False
    
    MAX_CONCURRENT_SESSIONS: int = 1000
    WEBSOCKET_TIMEOUT: int = 300
    
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/multi_game_arena.log"
    
    @field_validator("ML_MODEL_PATH", mode="before")
    def create_model_directory(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("LOG_FILE", mode="before")
    def create_log_directory(cls, v: str) -> str:
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
