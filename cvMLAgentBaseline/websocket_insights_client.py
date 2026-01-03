#!/usr/bin/env python3
"""
WebSocket Client for Testing Coach Follow-Up Messages

Sends test messages to the insights WebSocket server.
"""

import asyncio
import json
import logging
import sys
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Try to import WebSocket library
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def send_coach_follow_up(
    uri: str,
    insight: str,
    coach_follow_up: str,
    session_id: Optional[str] = None
):
    """
    Send coach follow-up message to WebSocket server.
    
    Args:
        uri: WebSocket server URI (e.g., "ws://localhost:8765")
        insight: Insight text
        coach_follow_up: One of "Monitor", "Adjust Training", "Escalate to AT/PT", "Dismiss"
        session_id: Optional session ID
    """
    if not WEBSOCKETS_AVAILABLE:
        print("❌ websockets library not available. Install with: pip install websockets")
        return
    
    message = {
        "insight": insight,
        "coach_follow_up": coach_follow_up
    }
    
    if session_id:
        message["session_id"] = session_id
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"🔌 Connected to {uri}")
            
            # Send message
            logger.info(f"📤 Sending message: {message}")
            await websocket.send(json.dumps(message))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            
            logger.info(f"📥 Received response: {response_data}")
            
            if response_data.get("success"):
                print(f"✅ Success: {response_data.get('message')}")
            else:
                print(f"❌ Error: {response_data.get('error')}")
            
            return response_data
    
    except Exception as e:
        logger.error(f"❌ Error sending message: {e}", exc_info=True)
        return None


async def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Send coach follow-up message to WebSocket server")
    parser.add_argument("--uri", type=str, default="ws://localhost:8765", help="WebSocket server URI")
    parser.add_argument("--insight", type=str, required=True, help="Insight text")
    parser.add_argument("--follow-up", type=str, required=True, 
                       choices=["Monitor", "Adjust Training", "Escalate to AT/PT", "Dismiss"],
                       help="Coach follow-up action")
    parser.add_argument("--session-id", type=str, default=None, help="Optional session ID")
    
    args = parser.parse_args()
    
    await send_coach_follow_up(
        uri=args.uri,
        insight=args.insight,
        coach_follow_up=args.follow_up,
        session_id=args.session_id
    )


if __name__ == "__main__":
    asyncio.run(main())

