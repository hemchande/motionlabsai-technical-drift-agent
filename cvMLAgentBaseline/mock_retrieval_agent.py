#!/usr/bin/env python3
"""
Mock Retrieval Agent - Listens to Redis queue and recalculates insights

This simulates the retrieval agent that processes messages from the queue
and recalculates insights when a video call ends.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
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


class MockRetrievalAgent:
    """
    Mock retrieval agent that listens to Redis queue and processes messages.
    """
    
    def __init__(self):
        """Initialize mock retrieval agent."""
        self.redis_client = None
        self.queue_name = "video_call_ended_queue"
        self.running = False
        
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
    
    def process_message(self, message: Dict[str, Any]) -> bool:
        """
        Process a message from the queue.
        
        Args:
            message: Message dictionary from queue
            
        Returns:
            True if processed successfully, False otherwise
        """
        try:
            event_type = message.get("event_type")
            session_id = message.get("session_id")
            
            if event_type != "video_call_ended":
                logger.warning(f"⚠️  Unknown event type: {event_type}")
                return False
            
            logger.info(f"📨 Processing message for session: {session_id}")
            logger.info(f"   Activity: {message.get('activity')}")
            logger.info(f"   Technique: {message.get('technique')}")
            logger.info(f"   Athlete: {message.get('athlete_name')}")
            logger.info(f"   Duration: {message.get('duration_seconds')}s")
            logger.info(f"   Frames: {message.get('frame_count')}")
            
            # Simulate insight recalculation
            success = self.recalculate_insights(message)
            
            if success:
                logger.info(f"✅ Successfully processed session: {session_id}")
            else:
                logger.error(f"❌ Failed to process session: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}", exc_info=True)
            return False
    
    def recalculate_insights(self, message: Dict[str, Any]) -> bool:
        """
        Mock insight recalculation.
        
        In the real implementation, this would:
        1. Query MongoDB for session data
        2. Run form correction analysis
        3. Update insights collection
        4. Track trends
        
        Args:
            message: Message with session information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_id = message.get("session_id")
            activity = message.get("activity")
            technique = message.get("technique")
            
            logger.info(f"🔄 Recalculating insights for session: {session_id}")
            
            # Simulate processing time
            time.sleep(0.5)
            
            # Mock insights (in real implementation, these would come from analysis)
            mock_insights = [
                "Insufficient height off floor/beam",
                "Insufficient landing knee extension",
                "Not enough hip flexion"
            ]
            
            # Simulate finding form issues
            logger.info(f"   📊 Found {len(mock_insights)} form issues")
            for i, insight in enumerate(mock_insights, 1):
                logger.info(f"      {i}. {insight}")
            
            # Simulate trend analysis
            logger.info(f"   📈 Analyzing trends across sessions...")
            time.sleep(0.3)
            
            # Simulate MongoDB update
            logger.info(f"   💾 Updating insights in MongoDB...")
            time.sleep(0.2)
            
            # Mock result
            result = {
                "session_id": session_id,
                "insights_count": len(mock_insights),
                "insights": mock_insights,
                "activity": activity,
                "technique": technique,
                "processed_at": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            
            logger.info(f"✅ Insights recalculated successfully")
            logger.debug(f"   Result: {json.dumps(result, indent=2)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recalculating insights: {e}", exc_info=True)
            return False
    
    def listen_to_queue(self, timeout: int = 1, max_messages: Optional[int] = None) -> None:
        """
        Listen to Redis queue and process messages.
        
        Args:
            timeout: Blocking timeout in seconds (0 = no timeout, blocks forever)
            max_messages: Maximum number of messages to process (None = unlimited)
        """
        if not self.redis_client:
            logger.error("❌ Redis client not available")
            return
        
        self.running = True
        logger.info(f"👂 Listening to queue: {self.queue_name}")
        logger.info(f"   Timeout: {timeout}s")
        logger.info(f"   Max messages: {max_messages or 'unlimited'}")
        logger.info("   Press Ctrl+C to stop\n")
        
        message_count = 0
        
        try:
            while self.running:
                try:
                    # Blocking pop from queue (BRPOP blocks until message available)
                    # Returns tuple: (queue_name, message) or None if timeout
                    result = self.redis_client.brpop(self.queue_name, timeout=timeout)
                    
                    if result is None:
                        # Timeout - no message received
                        continue
                    
                    queue_name, message_json = result
                    
                    # Parse message
                    try:
                        message = json.loads(message_json)
                        message_count += 1
                        
                        logger.info(f"\n{'='*60}")
                        logger.info(f"📨 Message #{message_count} received")
                        logger.info(f"{'='*60}")
                        
                        # Process message
                        success = self.process_message(message)
                        
                        if success:
                            logger.info(f"✅ Message #{message_count} processed successfully")
                        else:
                            logger.error(f"❌ Message #{message_count} processing failed")
                        
                        logger.info(f"{'='*60}\n")
                        
                        # Check if we've reached max messages
                        if max_messages and message_count >= max_messages:
                            logger.info(f"✅ Processed {message_count} messages. Stopping.")
                            break
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse message JSON: {e}")
                        logger.error(f"   Message: {message_json[:100]}")
                        continue
                
                except KeyboardInterrupt:
                    logger.info("\n🛑 Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in queue listener: {e}", exc_info=True)
                    time.sleep(1)  # Wait before retrying
        
        finally:
            self.running = False
            logger.info("👋 Stopped listening to queue")
    
    def stop(self):
        """Stop listening to queue."""
        self.running = False
        logger.info("🛑 Stopping queue listener...")
    
    def close(self):
        """Close Redis connection."""
        self.stop()
        if self.redis_client:
            self.redis_client.close()
            logger.info("✅ Redis connection closed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock Retrieval Agent - Listen to queue and process messages")
    parser.add_argument("--timeout", type=int, default=1, help="Blocking timeout in seconds (0 = no timeout)")
    parser.add_argument("--max-messages", type=int, default=None, help="Maximum number of messages to process")
    
    args = parser.parse_args()
    
    agent = MockRetrievalAgent()
    
    if not agent.redis_client:
        print("❌ Redis not available. Please install and start Redis.")
        return
    
    try:
        agent.listen_to_queue(timeout=args.timeout, max_messages=args.max_messages)
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
    finally:
        agent.close()


if __name__ == "__main__":
    main()

