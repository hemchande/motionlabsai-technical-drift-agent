"""
Redis Sub-Agent for Technical Drift Detection.

Specialized agent that handles all Redis queue operations.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import create_agent
from langchain.tools import tool

from config import Config

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


# Redis Tools
@tool
def redis_send_to_queue(
    queue_name: str,
    message: Dict[str, Any]
) -> str:
    """
    Send message to Redis queue for inter-service communication.
    
    Use this to send messages to queues like 'retrievalQueue', 'drift_alerts_queue', etc.
    """
    if not REDIS_AVAILABLE:
        return json.dumps({"error": "Redis not available", "success": False})
    
    try:
        redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True
        )
        
        message_json = json.dumps(message)
        redis_client.lpush(queue_name, message_json)
        
        return json.dumps({
            "success": True,
            "queue_name": queue_name,
            "message_sent": True
        })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


@tool
def redis_listen_to_queue(
    queue_name: str,
    timeout: int = 5,
    max_messages: int = 1
) -> str:
    """
    Listen to Redis queue and retrieve messages.
    
    Use this to receive messages from queues.
    Returns JSON with count and messages array.
    """
    if not REDIS_AVAILABLE:
        return json.dumps({"error": "Redis not available", "success": False})
    
    try:
        redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True
        )
        
        messages = []
        for _ in range(max_messages):
            result = redis_client.brpop(queue_name, timeout=timeout)
            if result:
                queue, message_json = result
                messages.append(json.loads(message_json))
            else:
                break
        
        return json.dumps({
            "success": True,
            "count": len(messages),
            "messages": messages
        })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


# Redis Sub-Agent Prompt
REDIS_AGENT_PROMPT = (
    "You are a Redis queue management assistant for technical drift detection. "
    "Your role is to send and receive messages via Redis queues. "
    "You handle inter-service communication through queues. "
    "Common queues: 'retrievalQueue', 'drift_alerts_queue', 'coach_followup_queue'. "
    "Always confirm what messages were sent or received in your final response."
)


def create_redis_agent(model):
    """Create the Redis sub-agent."""
    return create_agent(
        model,
        tools=[
            redis_send_to_queue,
            redis_listen_to_queue,
        ],
        prompt=REDIS_AGENT_PROMPT,
    )


