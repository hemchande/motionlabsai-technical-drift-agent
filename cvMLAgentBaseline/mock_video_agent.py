#!/usr/bin/env python3
"""
Mock Video Agent - Simulates video call ending and sends message to Redis queue

This simulates the video processing agent that sends a message when a video call ends.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Try to import Redis
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


class MockVideoAgent:
    """
    Mock video agent that simulates video call processing.
    """
    
    def __init__(self, queue_name: str = "retrievalQueue"):
        """Initialize mock video agent."""
        self.redis_client = None
        self.queue_name = queue_name
        
        if not REDIS_AVAILABLE:
            logger.warning("⚠️  Redis not available. Install with: pip install redis")
            return
        
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
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def simulate_video_call_ended(
        self,
        session_id: str,
        athlete_id: str,
        activity: str = "gymnastics",
        technique: str = "back_handspring",
        athlete_name: str = "Test Athlete",
        duration_seconds: float = 120.5,
        frame_count: int = 3600
    ) -> bool:
        """
        Simulate a video call ending and send message to Redis queue.
        
        Args:
            session_id: Session identifier
            athlete_id: Athlete identifier (REQUIRED for baseline/drift detection)
            activity: Activity type
            technique: Technique performed
            athlete_name: Name of athlete
            duration_seconds: Video duration in seconds
            frame_count: Number of frames processed
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.redis_client:
            logger.error("❌ Redis client not available")
            return False
        
        try:
            # Create message payload
            message = {
                "event_type": "video_call_ended",
                "session_id": session_id,
                "athlete_id": athlete_id,  # REQUIRED for baseline/drift detection
                "activity": activity,
                "technique": technique,
                "athlete_name": athlete_name,
                "duration_seconds": duration_seconds,
                "frame_count": frame_count,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            
            # Send to Redis queue (using LPUSH for queue)
            message_json = json.dumps(message)
            self.redis_client.lpush(self.queue_name, message_json)
            
            logger.info(f"✅ Sent video call ended message to queue: {session_id}")
            logger.debug(f"   Message: {message}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send message to queue: {e}", exc_info=True)
            return False
    
    def simulate_multiple_calls(self, count: int = 3, athlete_id: str = "test_athlete_001") -> None:
        """
        Simulate multiple video calls ending.
        
        Args:
            count: Number of calls to simulate
            athlete_id: Athlete ID (same for all calls)
        """
        logger.info(f"📹 Simulating {count} video calls ending for athlete: {athlete_id}...")
        
        for i in range(count):
            session_id = f"test_session_{int(time.time())}_{i}"
            success = self.simulate_video_call_ended(
                session_id=session_id,
                athlete_id=athlete_id,
                activity="gymnastics",
                technique=["back_handspring", "front_flip", "vault"][i % 3],
                athlete_name=f"Athlete {i+1}",
                duration_seconds=100 + (i * 10),
                frame_count=3000 + (i * 100)
            )
            
            if success:
                logger.info(f"   ✅ Call {i+1}/{count} processed")
            else:
                logger.error(f"   ❌ Call {i+1}/{count} failed")
            
            time.sleep(0.5)  # Small delay between calls
        
        logger.info(f"✅ Completed simulating {count} video calls")
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("✅ Redis connection closed")


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock Video Agent - Simulate video call ending")
    parser.add_argument("--session-id", type=str, default=None, help="Session ID (auto-generated if not provided)")
    parser.add_argument("--athlete-id", type=str, default="test_athlete_001", help="Athlete ID (REQUIRED)")
    parser.add_argument("--activity", type=str, default="gymnastics", help="Activity type")
    parser.add_argument("--technique", type=str, default="back_handspring", help="Technique")
    parser.add_argument("--athlete", type=str, default="Test Athlete", help="Athlete name")
    parser.add_argument("--count", type=int, default=1, help="Number of calls to simulate")
    parser.add_argument("--queue", type=str, default="retrievalQueue", help="Redis queue name (default: retrievalQueue)")
    
    args = parser.parse_args()
    
    agent = MockVideoAgent(queue_name=args.queue)
    
    if not agent.redis_client:
        print("❌ Redis not available. Please install and start Redis.")
        return
    
    if args.count > 1:
        agent.simulate_multiple_calls(count=args.count, athlete_id=args.athlete_id)
    else:
        session_id = args.session_id or f"test_session_{int(time.time())}"
        success = agent.simulate_video_call_ended(
            session_id=session_id,
            athlete_id=args.athlete_id,
            activity=args.activity,
            technique=args.technique,
            athlete_name=args.athlete
        )
        
        if success:
            print(f"✅ Video call ended message sent for session: {session_id}")
        else:
            print(f"❌ Failed to send message")
    
    agent.close()


if __name__ == "__main__":
    main()

