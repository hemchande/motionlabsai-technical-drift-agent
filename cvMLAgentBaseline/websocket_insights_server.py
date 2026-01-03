#!/usr/bin/env python3
"""
WebSocket Server for Coach Follow-Up on Insights

Listens for WebSocket messages with coach follow-up decisions and updates MongoDB.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from videoAgent.mongodb_service import MongoDBService

# Try to import WebSocket and Redis libraries
try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsightsWebSocketServer:
    """
    WebSocket server for handling coach follow-up messages on insights.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.mongodb = MongoDBService()
        
        if not self.mongodb.connect():
            raise RuntimeError("Failed to connect to MongoDB")
        
        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", 6379))
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"⚠️  Redis not available: {e}")
                self.redis_client = None
        
        logger.info(f"✅ Initialized InsightsWebSocketServer on {host}:{port}")
    
    def validate_message(self, message: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate WebSocket message structure.
        
        Args:
            message: Message dictionary
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(message, dict):
            return False, "Message must be a JSON object"
        
        if "insight" not in message:
            return False, "Message must contain 'insight' field"
        
        if "coach_follow_up" not in message:
            return False, "Message must contain 'coach_follow_up' field"
        
        valid_follow_ups = ["Monitor", "Adjust Training", "Escalate to AT/PT", "Dismiss"]
        if message["coach_follow_up"] not in valid_follow_ups:
            return False, f"coach_follow_up must be one of: {valid_follow_ups}"
        
        return True, None
    
    async def handle_message(self, websocket, path):
        """
        Handle incoming WebSocket messages.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        logger.info(f"🔌 New WebSocket connection from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    # Parse JSON message
                    data = json.loads(message)
                    logger.info(f"📨 Received message: {data}")
                    
                    # Validate message
                    is_valid, error = self.validate_message(data)
                    if not is_valid:
                        error_response = {
                            "success": False,
                            "error": error,
                            "received": data
                        }
                        await websocket.send(json.dumps(error_response))
                        logger.warning(f"⚠️  Invalid message: {error}")
                        continue
                    
                    insight_text = data["insight"]
                    coach_follow_up = data["coach_follow_up"]
                    
                    # Get session_id from message or find it
                    session_id = data.get("session_id")
                    
                    if not session_id:
                        # Try to find session by insight text
                        session_id = self._find_session_by_insight(insight_text)
                    
                    if not session_id:
                        error_response = {
                            "success": False,
                            "error": "Could not find session for insight. Please provide session_id in message.",
                            "received": data
                        }
                        await websocket.send(json.dumps(error_response))
                        logger.warning(f"⚠️  Could not find session for insight: {insight_text}")
                        continue
                    
                    # Update MongoDB
                    success = self.mongodb.update_insight_coach_follow_up(
                        session_id=session_id,
                        insight_text=insight_text,
                        coach_follow_up=coach_follow_up
                    )
                    
                    if success:
                        # Send follow-up action to queue
                        self._send_followup_to_queue(session_id, insight_text, coach_follow_up)
                        # Store in Redis if available (for caching/real-time updates)
                        if self.redis_client:
                            try:
                                redis_key = f"insight:{session_id}:{insight_text}"
                                redis_data = {
                                    "insight": insight_text,
                                    "coach_follow_up": coach_follow_up,
                                    "is_monitored": (coach_follow_up == "Monitor"),
                                    "updated_at": datetime.utcnow().isoformat()
                                }
                                self.redis_client.setex(
                                    redis_key,
                                    3600,  # 1 hour TTL
                                    json.dumps(redis_data)
                                )
                                logger.info(f"✅ Cached insight update in Redis: {redis_key}")
                            except Exception as e:
                                logger.warning(f"⚠️  Failed to cache in Redis: {e}")
                        
                        response = {
                            "success": True,
                            "message": f"Updated insight '{insight_text}' with coach_follow_up: {coach_follow_up}",
                            "session_id": session_id,
                            "insight": insight_text,
                            "coach_follow_up": coach_follow_up,
                            "is_monitored": (coach_follow_up == "Monitor")
                        }
                        await websocket.send(json.dumps(response))
                        logger.info(f"✅ Successfully updated insight: {insight_text} -> {coach_follow_up}")
                    else:
                        error_response = {
                            "success": False,
                            "error": "Failed to update insight in MongoDB",
                            "received": data
                        }
                        await websocket.send(json.dumps(error_response))
                        logger.error(f"❌ Failed to update insight in MongoDB")
                
                except json.JSONDecodeError as e:
                    error_response = {
                        "success": False,
                    "error": f"Invalid JSON: {str(e)}",
                        "received": message[:100]
                    }
                    await websocket.send(json.dumps(error_response))
                    logger.error(f"❌ JSON decode error: {e}")
                
                except Exception as e:
                    error_response = {
                        "success": False,
                        "error": f"Unexpected error: {str(e)}"
                    }
                    await websocket.send(json.dumps(error_response))
                    logger.error(f"❌ Error handling message: {e}", exc_info=True)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 WebSocket connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}", exc_info=True)
    
    def _send_followup_to_queue(self, session_id: str, insight_text: str, coach_follow_up: str) -> bool:
        """
        Send coach follow-up action to Redis queue.
        
        Args:
            session_id: Session ID
            insight_text: Insight text
            coach_follow_up: Follow-up action
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            # Get athlete_id from session if available
            athlete_id = None
            try:
                sessions_collection = self.mongodb.get_sessions_collection()
                session = sessions_collection.find_one({"session_id": session_id})
                if not session:
                    # Try by _id
                    from bson import ObjectId
                    try:
                        session = sessions_collection.find_one({"_id": ObjectId(session_id)})
                    except:
                        pass
                if session:
                    athlete_id = session.get("athlete_id")
            except:
                pass
            
            # Prepare queue message
            queue_message = {
                "event_type": "insight_followup",
                "session_id": session_id,
                "athlete_id": athlete_id,
                "insight": insight_text,
                "coach_follow_up": coach_follow_up,
                "is_monitored": (coach_follow_up == "Monitor"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to coach_followup_queue
            queue_name = "coach_followup_queue"
            message_json = json.dumps(queue_message)
            self.redis_client.lpush(queue_name, message_json)
            
            logger.info(f"✅ Sent insight follow-up to queue: {queue_name}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to send follow-up to queue: {e}")
            return False
    
    def _find_session_by_insight(self, insight_text: str) -> Optional[str]:
        """
        Find session_id by insight text.
        
        Args:
            insight_text: Insight text to search for
            
        Returns:
            Session ID if found, None otherwise
        """
        try:
            collection = self.mongodb.get_insights_collection()
            
            # Search for document containing this insight
            # Handle both old (string) and new (object) formats
            query = {
                "$or": [
                    {"insights": insight_text},  # Old format: array of strings
                    {"insights.insight": insight_text}  # New format: array of objects
                ]
            }
            
            doc = collection.find_one(query)
            if doc:
                return doc.get("session_id")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding session by insight: {e}")
            return None
    
    async def start(self):
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets library not available. Install with: pip install websockets")
        
        logger.info(f"🚀 Starting WebSocket server on ws://{self.host}:{self.port}")
        
        async with serve(self.handle_message, self.host, self.port):
            logger.info(f"✅ WebSocket server running on ws://{self.host}:{self.port}")
            logger.info("   Waiting for connections...")
            await asyncio.Future()  # Run forever
    
    def close(self):
        """Close connections."""
        if self.mongodb:
            self.mongodb.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("✅ WebSocket server connections closed")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket server for coach follow-up on insights")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    
    args = parser.parse_args()
    
    try:
        server = InsightsWebSocketServer(host=args.host, port=args.port)
        await server.start()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down WebSocket server...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}", exc_info=True)
    finally:
        if 'server' in locals():
            server.close()


if __name__ == "__main__":
    asyncio.run(main())

