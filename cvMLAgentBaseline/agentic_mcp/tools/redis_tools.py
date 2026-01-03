"""
Redis Tools for Agentic MCP Server.
"""
import json
import os
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import Config

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class SendToQueueInput(BaseModel):
    """Input schema for send_to_queue tool."""
    queue_name: str = Field(..., description="Queue name (e.g., 'retrievalQueue', 'drift_alerts_queue')")
    message: Dict[str, Any] = Field(..., description="Message dictionary to send")


@tool(args_schema=SendToQueueInput)
def redis_send_to_queue(
    queue_name: str,
    message: Dict[str, Any]
) -> str:
    """
    Send message to Redis queue.
    
    Use this tool to send messages to Redis queues for inter-service communication.
    """
    if not REDIS_AVAILABLE:
        return json.dumps({"error": "Redis not available"})
    
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


class ListenToQueueInput(BaseModel):
    """Input schema for listen_to_queue tool."""
    queue_name: str = Field(..., description="Queue name to listen to")
    timeout: int = Field(5, description="Timeout in seconds")
    max_messages: int = Field(1, description="Maximum messages to retrieve")


@tool(args_schema=ListenToQueueInput)
def redis_listen_to_queue(
    queue_name: str,
    timeout: int = 5,
    max_messages: int = 1
) -> str:
    """
    Listen to Redis queue and retrieve messages.
    
    Use this tool to receive messages from Redis queues.
    """
    if not REDIS_AVAILABLE:
        return json.dumps({"error": "Redis not available"})
    
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


@tool
def redis_broadcast_message(
    channel: str,
    message: Dict[str, Any]
) -> str:
    """
    Broadcast message to Redis pub/sub channel.
    
    Use this tool to broadcast messages to multiple subscribers.
    """
    if not REDIS_AVAILABLE:
        return json.dumps({"error": "Redis not available"})
    
    try:
        redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True
        )
        
        message_json = json.dumps(message)
        redis_client.publish(channel, message_json)
        
        return json.dumps({
            "success": True,
            "channel": channel,
            "message_broadcast": True
        })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})

