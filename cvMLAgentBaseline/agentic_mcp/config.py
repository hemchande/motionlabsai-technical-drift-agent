"""
Configuration management for Agentic MCP Server.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

class Config:
    """Configuration class for MCP Server."""
    
    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "gymnastics_analytics")
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    
    # WebSocket
    WEBSOCKET_HOST: str = os.getenv("WEBSOCKET_HOST", "localhost")
    WEBSOCKET_PORT: int = int(os.getenv("WEBSOCKET_PORT", "8765"))
    
    # Cloudflare
    CLOUDFLARE_ACCOUNT_ID: Optional[str] = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    CLOUDFLARE_API_TOKEN: Optional[str] = os.getenv("CLOUDFLARE_API_TOKEN")
    
    # Queue Names
    RETRIEVAL_QUEUE: str = os.getenv("RETRIEVAL_QUEUE", "retrievalQueue")
    DRIFT_ALERTS_QUEUE: str = os.getenv("DRIFT_ALERTS_QUEUE", "drift_alerts_queue")
    COACH_FOLLOWUP_QUEUE: str = os.getenv("COACH_FOLLOWUP_QUEUE", "coach_followup_queue")
    
    # Agent Settings
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "50"))
    AGENT_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        required = [
            ("MONGODB_URI", cls.MONGODB_URI),
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
        ]
        
        missing = [name for name, value in required if not value]
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        return True

